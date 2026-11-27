"""Phase 3: Fit individual user-wise low-rank BT models across ranks & weight decays vs.
shared mythical BT, compute fit/calibration/heterogeneity diagnostics, save checkpoints,
and generate diagnostic plots.

Usage:
  # Fast local CPU dry-run on a real slice of community_alignment_en.csv:
  python run_phase3_user_bt.py --mode local-test

  # Full-scale run (intended for cloud A100/H100 GPU, K > 20 -> 839 users, 79,323 pairs):
  python run_phase3_user_bt.py --mode full --device cuda --encode-batch-size 128
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import Turn1Dataset, extract_local_fixture_csv, load_turn1_dataset
from src.encoder import FrozenModernBERTEncoder
from src.mythical_bt import MythicalUserBT, MythicalUserMLPBT, fit_mythical_bt, fit_mythical_mlp_bt
from src.user_bt import LowRankUserBT, fit_user_bt, global_affine_normalize


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


def split_pairs_per_user(
    pairs: list[tuple[int, int, int, int]],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Stratified per-user, click-grouped train/val split (zero within-click leakage).

    Every 4-way annotator choice on `(user_idx, prompt_idx)` expands into up to 3 pairwise
    comparisons sharing the same chosen response. Grouping by `(user_idx, prompt_idx)` keeps
    all pairs from a given click together so `val` contains strictly unseen clicks.
    """
    rng = np.random.default_rng(seed)
    by_user_prompt: defaultdict[int, defaultdict[int, list[tuple[int, int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in pairs:
        by_user_prompt[p[0]][p[1]].append(p)

    train_pairs: list[tuple[int, int, int, int]] = []
    val_pairs: list[tuple[int, int, int, int]] = []
    for u_idx in sorted(by_user_prompt.keys()):
        p_keys = sorted(by_user_prompt[u_idx].keys())
        n_clicks = len(p_keys)
        perm = rng.permutation(n_clicks)
        n_val_clicks = max(1, int(round(n_clicks * val_frac))) if n_clicks >= 2 else 0
        val_p_set = {p_keys[i] for i in perm[:n_val_clicks].tolist()}
        for p_idx in p_keys:
            if p_idx in val_p_set:
                val_pairs.extend(by_user_prompt[u_idx][p_idx])
            else:
                train_pairs.extend(by_user_prompt[u_idx][p_idx])
    return train_pairs, val_pairs


@torch.no_grad()
def evaluate_margins(
    model: LowRankUserBT | MythicalUserBT | MythicalUserMLPBT,
    features_xy: torch.Tensor,
    pairs: list[tuple[int, int, int, int]],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute pairwise margins, log-loss, accuracy, and Brier score."""
    if len(pairs) == 0:
        return np.empty(0, dtype=np.float64), {"loss": float("nan"), "acc": float("nan"), "brier": float("nan")}

    feats = features_xy.to(device, dtype=torch.float32)
    u_arr = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    pos_arr = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=device)
    neg_arr = torch.tensor([p[3] for p in pairs], dtype=torch.long, device=device)

    if isinstance(model, LowRankUserBT):
        margins = model.forward_pairs(u_arr, feats[pos_arr], feats[neg_arr])
    else:
        margins = model.forward_pairs(feats[pos_arr], feats[neg_arr])

    loss = float(F.softplus(-margins).mean().item())
    probs = torch.sigmoid(margins)
    acc = float((margins > 0).float().mean().item())
    brier = float(((1.0 - probs) ** 2).mean().item())
    return margins.cpu().numpy().astype(np.float64), {"loss": loss, "acc": acc, "brier": brier}


def compute_heterogeneity_stats(
    u_norm: np.ndarray,
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
) -> dict:
    """Measure inter-user utility variance and top-1 disagreement across evaluation prompts."""
    response_stds: list[float] = []
    top1_distinct_counts: list[int] = []
    top1_agreement_fracs: list[float] = []

    for p_idx in eval_prompt_indices:
        g_idxs = ds.prompt_global_xy_indices[p_idx]
        if len(g_idxs) < 2:
            continue
        sub = u_norm[:, g_idxs]  # [N_users, |A(x)|]
        response_stds.extend(np.std(sub, axis=0).tolist())

        top1 = np.argmax(sub, axis=1)
        vals, counts = np.unique(top1, return_counts=True)
        top1_distinct_counts.append(int(len(vals)))
        top1_agreement_fracs.append(float(np.max(counts) / len(top1)))

    return {
        "num_eval_prompts": len(eval_prompt_indices),
        "mean_resp_user_std": float(np.mean(response_stds)) if response_stds else 0.0,
        "median_resp_user_std": float(np.median(response_stds)) if response_stds else 0.0,
        "mean_distinct_top1": float(np.mean(top1_distinct_counts)) if top1_distinct_counts else 0.0,
        "median_top1_majority_frac": float(np.median(top1_agreement_fracs)) if top1_agreement_fracs else 1.0,
        "response_stds": response_stds,
        "top1_majority_fracs": top1_agreement_fracs,
    }


def plot_phase3_diagnostics(
    best_per_family_df: pd.DataFrame,
    best_rank: int,
    best_wd: float,
    best_u_norm: np.ndarray,
    best_beta_nat: float,
    best_het: dict,
    best_cal_margins: np.ndarray,
    save_path: Path,
    mode_label: str,
) -> Path:
    """Create 4-panel Phase 3 diagnostic figure (using best weight decay per model family)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.5), dpi=160)

    # Panel 1: Train & Val Accuracy across best-WD models per family
    ax = axes[0, 0]
    labels = [
        f"{r['model']}\n(wd={r['weight_decay']:g})"
        for _, r in best_per_family_df.iterrows()
    ]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, best_per_family_df["full_acc"], width=w, color="#2563eb", label="Full-data Acc")
    b2 = ax.bar(x + w / 2, best_per_family_df["val_acc"], width=w, color="#059669", label="Held-out (20%) Acc")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1.2, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("Pairwise Accuracy")
    ax.set_title("Model Fit (Best Weight Decay per Rank)", fontweight="bold")
    ax.legend(fontsize=8.5, frameon=True)

    # Panel 2: Distribution of normalized utilities u_tilde in [0, 1]
    ax = axes[0, 1]
    u_flat = best_u_norm.ravel()
    ax.hist(u_flat, bins=40, color="#7c3aed", edgecolor="white", alpha=0.88)
    ax.axvline(float(np.mean(u_flat)), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(u_flat):.3f}")
    ax.axvline(float(np.median(u_flat)), color="#0f766e", linestyle=":", linewidth=1.7, label=f"Median = {np.median(u_flat):.3f}")
    p5, p95 = np.percentile(u_flat, [5, 95])
    ax.set_title(
        f"Normalized Utilities $\\tilde{{u}}_i(y|x)$ (Rank {best_rank}, wd={best_wd:g}, $\\beta_{{BT}}^{{nat}}={best_beta_nat:.2f}$)",
        fontweight="bold",
    )
    ax.set_xlabel(f"Global Affine Utility in [0, 1]  (5th–95th pct: [{p5:.2f}, {p95:.2f}])")
    ax.set_ylabel("Count of (user, x, y) entries")
    ax.legend(fontsize=8.5, frameon=True)

    # Panel 3: Inter-user heterogeneity across evaluation prompts
    ax = axes[1, 0]
    maj_fracs = best_het["top1_majority_fracs"]
    if len(maj_fracs) > 0:
        ax.hist(maj_fracs, bins=np.linspace(0.0, 1.0, 21), color="#d97706", edgecolor="white", alpha=0.88)
        ax.axvline(
            best_het["median_top1_majority_frac"],
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"Median majority share = {best_het['median_top1_majority_frac']:.2f}",
        )
    ax.set_title(
        f"User Disagreement on Shared Prompts (Rank {best_rank}, n={best_het['num_eval_prompts']} prompts)",
        fontweight="bold",
    )
    ax.set_xlabel("Fraction of Users Agreeing on Top-1 Response (lower = more heterogeneous)")
    ax.set_ylabel("Number of Prompts")
    ax.legend(fontsize=8.5, frameon=True)

    # Panel 4: Calibration curve (symmetrized pairs)
    ax = axes[1, 1]
    if best_cal_margins.size > 0:
        all_m = np.concatenate([best_cal_margins, -best_cal_margins])
        all_y = np.concatenate([np.ones_like(best_cal_margins), np.zeros_like(best_cal_margins)])
        all_p = 1.0 / (1.0 + np.exp(-np.clip(all_m, -50.0, 50.0)))

        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.clip(np.digitize(all_p, bins) - 1, 0, len(bins) - 2)
        pred_means, emp_means = [], []
        for b in range(len(bins) - 1):
            mask = bin_ids == b
            if np.any(mask):
                pred_means.append(float(np.mean(all_p[mask])))
                emp_means.append(float(np.mean(all_y[mask])))

        ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.3, label="Perfect calibration")
        ax.plot(pred_means, emp_means, marker="o", color="#2563eb", linewidth=1.8, label=f"Rank-{best_rank} (wd={best_wd:g})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted BT Win Probability")
    ax.set_ylabel("Empirical Win Frequency")
    ax.set_title(f"Held-Out BT Probability Calibration (Rank {best_rank}, wd={best_wd:g})", fontweight="bold")
    ax.legend(fontsize=8.5, frameon=True)

    fig.suptitle(
        f"Phase 3 Diagnostics: Individual User-Wise Low-Rank BT Models ({mode_label})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 user-wise low-rank BT training & diagnostics.")
    parser.add_argument("--mode", choices=["local-test", "full"], default="local-test")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ranks", type=int, nargs="+", default=None, help="Defaults to [2, 8] locally and [8] in full mode.")
    parser.add_argument("--mlp-hidden", type=int, default=8, help="Hidden bottleneck width for Shared MLP diagnostic (hidden_dim -> h -> 1).")
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.01, 0.1, 1.0, 5.0, 10.0, 20.0])
    parser.add_argument(
        "--prior",
        choices=["mean-centered", "zero-centered"],
        default="mean-centered",
        help="Hierarchical prior for UserBT: mean-centered shrinks delta_i = w_i - w_bar; zero-centered shrinks w_i -> 0.",
    )
    parser.add_argument(
        "--shared-weight-decay",
        type=float,
        default=0.1,
        help="Weight decay on shared projection W and population mean w_bar under mean-centered prior.",
    )
    parser.add_argument("--min-user-prefs-gt", type=int, default=20)
    parser.add_argument("--eval-min-users", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Do not save/load cached ModernBERT CLS features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = args.out_dir or (ROOT / "results" / ("phase3_local_test" if args.mode == "local-test" else "phase3"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Turn-1 dataset
    if args.mode == "local-test":
        ranks = args.ranks if args.ranks is not None else [2, 8]
        fixture_csv = out_dir / "local_fixture_en.csv"
        if not fixture_csv.exists():
            extract_local_fixture_csv(args.csv, fixture_csv, num_prompts=6, max_users=16)
        ds = load_turn1_dataset(fixture_csv, min_user_prefs_gt=1)
        eval_prompt_indices = ds.prompts_with_min_users(3)
        max_len = min(args.max_length, 256)
        enc_bs = args.encode_batch_size or 16
        epochs = args.epochs or 80
        train_bs = min(args.batch_size, 128)
        mode_label = f"Local CPU Slice: {ds.num_users} users, {ds.num_prompts} prompts, {ds.num_pairs} pairs"
    else:
        ranks = args.ranks if args.ranks is not None else [8]
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=args.min_user_prefs_gt)
        eval_prompt_indices = ds.prompts_with_min_users(args.eval_min_users)
        max_len = args.max_length
        enc_bs = args.encode_batch_size or (128 if device.type == "cuda" else 16)
        epochs = args.epochs or 120
        train_bs = args.batch_size
        mode_label = f"Full EN Turn-1 (K>{args.min_user_prefs_gt}): {ds.num_users} users, {ds.num_pairs:,} pairs"

    print(f"=== Phase 3 ({args.mode}) on device={device} ===")
    print(f"Users={ds.num_users:,}, Prompts={ds.num_prompts:,}, Eval prompts={len(eval_prompt_indices):,}")
    print(f"Unique (x, y)={ds.num_unique_xy:,}, Pairwise pairs={ds.num_pairs:,}")
    print(f"Sweeping ranks={ranks}, mlp_hidden={args.mlp_hidden}, weight_decays={args.weight_decays}")

    # 2. Frozen ModernBERT CLS features (load from cache if already computed!)
    cache_path = None if args.no_cache else (out_dir / "modernbert_cls_features.pt")
    t_enc = time.perf_counter()
    if cache_path is not None and cache_path.exists():
        feats = torch.load(cache_path, map_location="cpu", weights_only=True)
        if feats.shape[0] != ds.num_unique_xy:
            feats = None
        else:
            print(f"Loaded cached ModernBERT features: shape={tuple(feats.shape)} from {cache_path}")
    else:
        feats = None

    if feats is None:
        encoder = FrozenModernBERTEncoder(
            model_name="answerdotai/ModernBERT-base",
            max_length=max_len,
            pooling="cls",
            device=device,
        )
        feats = encoder.precompute_dataset_features(ds, batch_size=enc_bs, cache_path=cache_path)
        print(f"Encoded frozen ModernBERT features: shape={tuple(feats.shape)} in {time.perf_counter() - t_enc:.1f}s")
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_pairs, val_pairs = split_pairs_per_user(ds.pairs, val_frac=0.2, seed=args.seed)
    train_clicks = {(p[0], p[1]) for p in train_pairs}
    val_clicks = {(p[0], p[1]) for p in val_pairs}
    assert len(train_clicks & val_clicks) == 0, "Click leakage detected between train and val splits!"
    print(
        f"Click-grouped split (0 overlap): "
        f"train={len(train_pairs):,} pairs ({len(train_clicks):,} clicks), "
        f"val={len(val_pairs):,} pairs ({len(val_clicks):,} unseen clicks)\n"
    )

    all_rows = []
    best_per_family_rows = []

    # 3A-1. Shared mythical-user LINEAR baseline (hidden_dim -> 1) across weight_decays
    hidden_dim = int(feats.shape[1])
    best_myth_row = None
    best_myth_state = None
    for wd in args.weight_decays:
        myth_full, _ = fit_mythical_bt(
            feats, ds.pairs, lr=args.lr, weight_decay=wd, epochs=epochs, batch_size=train_bs, seed=args.seed, device=device
        )
        _, myth_full_m = evaluate_margins(myth_full, feats, ds.pairs, device)

        myth_tr, _ = fit_mythical_bt(
            feats, train_pairs, lr=args.lr, weight_decay=wd, epochs=epochs, batch_size=train_bs, seed=args.seed, device=device
        )
        _, myth_tr_m = evaluate_margins(myth_tr, feats, train_pairs, device)
        _, myth_val_m = evaluate_margins(myth_tr, feats, val_pairs, device)

        row = {
            "model": f"Shared Linear ({hidden_dim}->1)",
            "rank": 0,
            "weight_decay": wd,
            "full_loss": myth_full_m["loss"],
            "full_acc": myth_full_m["acc"],
            "train_loss": myth_tr_m["loss"],
            "train_acc": myth_tr_m["acc"],
            "val_loss": myth_val_m["loss"],
            "val_acc": myth_val_m["acc"],
            "val_brier": myth_val_m["brier"],
            "beta_bt_nat": float("nan"),
            "mean_resp_user_std": 0.0,
            "median_top1_majority_frac": 1.0,
        }
        all_rows.append(row)
        if best_myth_row is None or row["val_loss"] < best_myth_row["val_loss"]:
            best_myth_row = row
            best_myth_state = {"state_dict": myth_full.state_dict(), "hidden_dim": hidden_dim, "weight_decay": wd}

    assert best_myth_row is not None and best_myth_state is not None
    torch.save(best_myth_state, out_dir / "mythical_bt_best.pt")
    best_per_family_rows.append(best_myth_row)
    print(
        f"  Shared Linear ({hidden_dim}->1, best wd={best_myth_row['weight_decay']:g}): "
        f"train_acc={best_myth_row['train_acc']:.4f}, full_acc={best_myth_row['full_acc']:.4f}, "
        f"val_acc={best_myth_row['val_acc']:.4f}, val_loss={best_myth_row['val_loss']:.4f}, val_brier={best_myth_row['val_brier']:.4f}"
    )

    # 3A-2. Shared mythical-user MLP baseline (hidden_dim -> mlp_hidden -> 1) across weight_decays
    best_mlp_row = None
    best_mlp_state = None
    for wd in args.weight_decays:
        mlp_full, _ = fit_mythical_mlp_bt(
            feats,
            ds.pairs,
            bottleneck_dim=args.mlp_hidden,
            lr=args.lr,
            weight_decay=wd,
            epochs=epochs,
            batch_size=train_bs,
            seed=args.seed,
            device=device,
        )
        _, mlp_full_m = evaluate_margins(mlp_full, feats, ds.pairs, device)

        mlp_tr, _ = fit_mythical_mlp_bt(
            feats,
            train_pairs,
            bottleneck_dim=args.mlp_hidden,
            lr=args.lr,
            weight_decay=wd,
            epochs=epochs,
            batch_size=train_bs,
            seed=args.seed,
            device=device,
        )
        _, mlp_tr_m = evaluate_margins(mlp_tr, feats, train_pairs, device)
        _, mlp_val_m = evaluate_margins(mlp_tr, feats, val_pairs, device)

        row = {
            "model": f"Shared MLP ({hidden_dim}->{args.mlp_hidden}->1)",
            "rank": 0,
            "weight_decay": wd,
            "full_loss": mlp_full_m["loss"],
            "full_acc": mlp_full_m["acc"],
            "train_loss": mlp_tr_m["loss"],
            "train_acc": mlp_tr_m["acc"],
            "val_loss": mlp_val_m["loss"],
            "val_acc": mlp_val_m["acc"],
            "val_brier": mlp_val_m["brier"],
            "beta_bt_nat": float("nan"),
            "mean_resp_user_std": 0.0,
            "median_top1_majority_frac": 1.0,
        }
        all_rows.append(row)
        if best_mlp_row is None or row["val_loss"] < best_mlp_row["val_loss"]:
            best_mlp_row = row
            best_mlp_state = {
                "state_dict": mlp_full.state_dict(),
                "hidden_dim": hidden_dim,
                "bottleneck_dim": args.mlp_hidden,
                "weight_decay": wd,
            }

    assert best_mlp_row is not None and best_mlp_state is not None
    torch.save(best_mlp_state, out_dir / "mythical_mlp_bt_best.pt")
    best_per_family_rows.append(best_mlp_row)
    print(
        f"  Shared MLP ({hidden_dim}->{args.mlp_hidden}->1, best wd={best_mlp_row['weight_decay']:g}): "
        f"train_acc={best_mlp_row['train_acc']:.4f}, full_acc={best_mlp_row['full_acc']:.4f}, "
        f"val_acc={best_mlp_row['val_acc']:.4f}, val_loss={best_mlp_row['val_loss']:.4f}, val_brier={best_mlp_row['val_brier']:.4f}"
    )

    # 3B. User-wise low-rank BT models across ranks x weight_decays
    overall_best_val_loss = float("inf")
    overall_best: dict = {}

    for rank in ranks:
        best_rank_row = None
        best_rank_ckpt = None

        for wd in args.weight_decays:
            full_model, _ = fit_user_bt(
                feats,
                ds.pairs,
                num_users=ds.num_users,
                rank=rank,
                lr=args.lr,
                weight_decay=wd,
                epochs=epochs,
                batch_size=train_bs,
                seed=args.seed,
                device=device,
            )
            _, full_m = evaluate_margins(full_model, feats, ds.pairs, device)

            tr_model, _ = fit_user_bt(
                feats,
                train_pairs,
                num_users=ds.num_users,
                rank=rank,
                lr=args.lr,
                weight_decay=wd,
                epochs=epochs,
                batch_size=train_bs,
                seed=args.seed,
                device=device,
            )
            _, tr_m = evaluate_margins(tr_model, feats, train_pairs, device)
            val_margins, val_m = evaluate_margins(tr_model, feats, val_pairs, device)

            with torch.no_grad():
                raw_scores = full_model.score_all_users(feats.to(device)).cpu().numpy()
            u_norm, beta_nat, s_min, s_max = global_affine_normalize(raw_scores)
            het = compute_heterogeneity_stats(u_norm, ds, eval_prompt_indices)

            row = {
                "model": f"UserBT (d={rank})",
                "rank": rank,
                "weight_decay": wd,
                "full_loss": full_m["loss"],
                "full_acc": full_m["acc"],
                "train_loss": tr_m["loss"],
                "train_acc": tr_m["acc"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["acc"],
                "val_brier": val_m["brier"],
                "beta_bt_nat": beta_nat,
                "mean_resp_user_std": het["mean_resp_user_std"],
                "median_top1_majority_frac": het["median_top1_majority_frac"],
            }
            all_rows.append(row)

            ckpt = {
                "state_dict": full_model.state_dict(),
                "num_users": ds.num_users,
                "hidden_dim": hidden_dim,
                "rank": rank,
                "weight_decay": wd,
                "beta_bt_nat": beta_nat,
                "s_min": s_min,
                "s_max": s_max,
            }
            if best_rank_row is None or row["val_loss"] < best_rank_row["val_loss"]:
                best_rank_row = row
                best_rank_ckpt = ckpt

            if val_m["loss"] < overall_best_val_loss:
                overall_best_val_loss = val_m["loss"]
                overall_best = {
                    "rank": rank,
                    "weight_decay": wd,
                    "ckpt": ckpt,
                    "u_norm": u_norm,
                    "beta_nat": beta_nat,
                    "het": het,
                    "val_margins": val_margins,
                }

        assert best_rank_row is not None and best_rank_ckpt is not None
        torch.save(best_rank_ckpt, out_dir / f"user_bt_rank{rank}_best.pt")
        best_per_family_rows.append(best_rank_row)
        print(
            f"  UserBT (d={rank}, best wd={best_rank_row['weight_decay']:g}): "
            f"train_acc={best_rank_row['train_acc']:.4f}, full_acc={best_rank_row['full_acc']:.4f}, "
            f"val_acc={best_rank_row['val_acc']:.4f}, val_loss={best_rank_row['val_loss']:.4f}, "
            f"val_brier={best_rank_row['val_brier']:.4f}, beta_nat={best_rank_row['beta_bt_nat']:.2f}, "
            f"med_top1_maj={best_rank_row['median_top1_majority_frac']:.2f}"
        )

    torch.save(overall_best["ckpt"], out_dir / "user_bt_best.pt")

    all_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_per_family_rows)
    all_df.to_csv(out_dir / "phase3_metrics_full_grid.csv", index=False)
    best_df.to_csv(out_dir / "phase3_metrics.csv", index=False)

    print("\n=== Phase 3 Best-per-Family Summary ===")
    print(best_df.round(4).to_string(index=False))

    fig_path = plot_phase3_diagnostics(
        best_per_family_df=best_df,
        best_rank=overall_best["rank"],
        best_wd=overall_best["weight_decay"],
        best_u_norm=overall_best["u_norm"],
        best_beta_nat=overall_best["beta_nat"],
        best_het=overall_best["het"],
        best_cal_margins=overall_best["val_margins"],
        save_path=out_dir / "phase3_user_bt_diagnostics.png",
        mode_label=mode_label,
    )

    summary_json = out_dir / "phase3_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "split_strategy": "click_grouped_per_user",
                "num_users": ds.num_users,
                "num_prompts": ds.num_prompts,
                "num_eval_prompts": len(eval_prompt_indices),
                "num_unique_xy": ds.num_unique_xy,
                "num_pairs": ds.num_pairs,
                "num_train_clicks": len(train_clicks),
                "num_val_clicks": len(val_clicks),
                "mlp_hidden": args.mlp_hidden,
                "best_rank": overall_best["rank"],
                "best_weight_decay": overall_best["weight_decay"],
                "best_beta_bt_nat": overall_best["beta_nat"],
                "best_mythical_weight_decay": best_myth_row["weight_decay"],
                "best_mythical_mlp_weight_decay": best_mlp_row["weight_decay"],
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
                "best_per_family": best_per_family_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved full grid to: {out_dir / 'phase3_metrics_full_grid.csv'}")
    print(f"Saved summary to  : {out_dir / 'phase3_metrics.csv'}")
    print(f"Saved plot to     : {fig_path}")
    print(f"Total elapsed     : {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
