"""Sanity check: Refit Bradley-Terry utilities on ModernBERT features from beta_BT-resampled
datasets and check whether the intended beta_BT is recovered (and how L2 prior / weight decay
affects identifiability vs. unbounded MLE growth).

Usage:
  python check_beta_recovery.py --mode full --rank 32 --epochs 60
  python check_beta_recovery.py --mode local-test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import load_turn1_dataset
from src.mythical_bt import fit_mythical_bt, resample_preferences
from src.user_bt import (
    LowRankUserBT,
    fit_user_bt,
    global_affine_normalize,
    per_query_normalize,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


def _resolve_best_ckpt(phase3_dir: Path, rank: int | None) -> Path:
    if rank is not None:
        p = phase3_dir / f"user_bt_rank{rank}_best.pt"
        if p.exists():
            return p
    best = phase3_dir / "user_bt_best.pt"
    if best.exists():
        return best
    cands = sorted(
        phase3_dir.glob("user_bt_rank*_best.pt"),
        key=lambda p: int(p.stem.split("rank")[1].split("_")[0]),
    )
    return cands[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check: recover intended beta_BT from resampled datasets.")
    parser.add_argument("--mode", choices=["local-test", "full"], default="full")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--source-rank", type=int, default=None, help="Ground-truth UserBT rank to generate u_tilde.")
    parser.add_argument("--refit-rank", type=int, default=32, help="Rank for refitting LowRankUserBT on resampled data.")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-3, 1e-1, 1.0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    phase3_dir = args.phase3_dir or (ROOT / "results" / ("phase3_local_test" if args.mode == "local-test" else "phase3"))

    if args.mode == "local-test":
        ds = load_turn1_dataset(phase3_dir / "local_fixture_en.csv", min_user_prefs_gt=1)
        eval_ps = ds.prompts_with_min_users(3)
        bs = min(args.batch_size, 128)
    else:
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=20)
        eval_ps = ds.prompts_with_min_users(20)
        bs = args.batch_size

    feats = torch.load(phase3_dir / "modernbert_cls_features.pt", map_location="cpu", weights_only=True).float()
    ckpt_path = _resolve_best_ckpt(phase3_dir, args.source_rank)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    src_bt = LowRankUserBT(int(ckpt["num_users"]), int(ckpt["hidden_dim"]), int(ckpt["rank"]))
    src_bt.load_state_dict(ckpt["state_dict"])
    src_bt.eval()

    with torch.no_grad():
        raw_src = src_bt.score_all_users(feats).numpy()
    u_true, true_spans = per_query_normalize(raw_src, ds.prompt_global_xy_indices)

    # Precompute true pairwise utility diffs on ds.pairs (and a deterministic 80/20 split)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds.pairs))
    n_val = max(1, int(0.2 * len(ds.pairs)))
    val_mask = np.zeros(len(ds.pairs), dtype=bool)
    val_mask[perm[:n_val]] = True

    u_idx_arr = np.array([p[0] for p in ds.pairs], dtype=np.int64)
    a_idx_arr = np.array([p[2] for p in ds.pairs], dtype=np.int64)
    b_idx_arr = np.array([p[3] for p in ds.pairs], dtype=np.int64)
    du_true_all = u_true[u_idx_arr, a_idx_arr] - u_true[u_idx_arr, b_idx_arr]

    # Precompute eval-prompt flattened indices for fast per-query span recovery
    eval_prompt_xy = [ds.prompt_global_xy_indices[p] for p in eval_ps]
    print(f"=== beta_BT Recovery Sanity Check ({args.mode}, src={ckpt_path.name}, refit_rank={args.refit_rank}, device={device}) ===")
    print(f"Users={ds.num_users}, Pairs={len(ds.pairs)}, Eval prompts={len(eval_ps)}, True du std={np.std(du_true_all):.4f}")

    rows = []
    for beta_target in args.betas:
        # Theoretical Bayes-optimal accuracy & NLL of the resampled distribution at beta_target
        p_true = 1.0 / (1.0 + np.exp(-np.clip(beta_target * du_true_all, -60, 60)))
        bayes_acc = float(np.mean(np.maximum(p_true, 1.0 - p_true)))
        p_clip = np.clip(p_true, 1e-12, 1.0 - 1e-12)
        bayes_nll = float(-np.mean(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip)))

        resamp_pairs = resample_preferences(u_norm=u_true, pairs=ds.pairs, beta_bt=beta_target, seed=args.seed)
        train_pairs = [resamp_pairs[i] for i in range(len(resamp_pairs)) if not val_mask[i]]
        val_pairs = [resamp_pairs[i] for i in range(len(resamp_pairs)) if val_mask[i]]

        # Also check MythicalUserBT logit scale on resampled pairs (wd=0.1)
        myth_model, _ = fit_mythical_bt(
            feats, resamp_pairs, lr=args.lr, weight_decay=0.1, epochs=args.epochs, batch_size=bs, seed=args.seed, device=device
        )
        with torch.no_grad():
            r_myth = myth_model.score(feats.to(device)).cpu().numpy()
        myth_eval_spans = [float(np.ptp(r_myth[idxs])) for idxs in eval_prompt_xy]
        myth_med_span = float(np.median(myth_eval_spans))

        for wd in args.weight_decays:
            # 1. Fit LowRankUserBT on 80% train, evaluate val NLL/Acc, and inspect recovered beta metrics
            refit_bt, hist = fit_user_bt(
                features_xy=feats,
                pairs=train_pairs,
                num_users=ds.num_users,
                rank=args.refit_rank,
                lr=args.lr,
                weight_decay=wd,
                epochs=args.epochs,
                batch_size=bs,
                seed=args.seed,
                device=device,
            )
            refit_bt.eval()
            with torch.no_grad():
                scores_hat = refit_bt.score_all_users(feats.to(device)).cpu().numpy()

            # Held-out validation loss & accuracy on val_pairs
            vu = np.array([p[0] for p in val_pairs], dtype=np.int64)
            vw = np.array([p[2] for p in val_pairs], dtype=np.int64)
            vl = np.array([p[3] for p in val_pairs], dtype=np.int64)
            val_diff = scores_hat[vu, vw] - scores_hat[vu, vl]
            val_acc = float(np.mean(val_diff > 0))
            val_nll = float(np.mean(np.logaddexp(0.0, -val_diff)))

            # Direct regression slope of recovered score diff s_hat(a)-s_hat(b) onto true u_tilde(a)-u_tilde(b)
            # If s_hat_i(y) ~= beta_hat * u_tilde_i(y) + c_{i,x}, then OLS slope == beta_hat!
            ds_hat_val = scores_hat[u_idx_arr[val_mask], a_idx_arr[val_mask]] - scores_hat[u_idx_arr[val_mask], b_idx_arr[val_mask]]
            du_val = du_true_all[val_mask]
            beta_hat_slope = float(np.dot(ds_hat_val, du_val) / max(float(np.dot(du_val, du_val)), 1e-12))
            corr_du = float(np.corrcoef(ds_hat_val, du_val)[0, 1]) if np.std(ds_hat_val) > 1e-8 else 0.0

            # Per-query max span M_hat(x) across eval prompts (our per-query beta_BT(x) estimator!)
            _, hat_spans = per_query_normalize(scores_hat, ds.prompt_global_xy_indices)
            eval_hat_spans = hat_spans[eval_ps]
            med_span = float(np.median(eval_hat_spans))
            p90_span = float(np.percentile(eval_hat_spans, 90))
            _, global_beta_hat, _, _ = global_affine_normalize(scores_hat)

            row = {
                "beta_target": float(beta_target),
                "wd": float(wd),
                "bayes_acc": round(bayes_acc, 4),
                "bayes_nll": round(bayes_nll, 4),
                "train_acc": round(float(hist.accuracies[-1]), 4),
                "val_acc": round(val_acc, 4),
                "val_nll": round(val_nll, 4),
                "beta_hat_slope": round(beta_hat_slope, 3),
                "corr_du_val": round(corr_du, 4),
                "med_query_span_M": round(med_span, 3),
                "p90_query_span_M": round(p90_span, 3),
                "global_beta_hat": round(float(global_beta_hat), 3),
                "myth_med_span": round(myth_med_span, 3),
            }
            rows.append(row)
            print(
                f"  beta={beta_target:4.1f} wd={wd:5.3f} | "
                f"Bayes(acc={bayes_acc:.3f}, nll={bayes_nll:.3f}) vs Val(acc={val_acc:.3f}, nll={val_nll:.3f}) | "
                f"beta_slope={beta_hat_slope:5.2f} (corr={corr_du:.3f}), med_M(x)={med_span:5.2f}, global={global_beta_hat:6.2f}, myth_M={myth_med_span:4.2f}",
                flush=True,
            )

    out_dir = ROOT / "results" / "beta_recovery_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"beta_recovery_{args.mode}_rank{args.refit_rank}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved beta_BT recovery table to {csv_path} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
