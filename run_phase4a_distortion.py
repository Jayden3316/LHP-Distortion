"""Phase 4A: Compute empirical KL-constrained distortion vs. realized KL radius tau(x)
using Phase 3 user-wise BT utilities u_tilde and shared mythical-user BT policies pi_lam.

Supports:
  - Evaluating real observed preferences (at implied beta_BT_nat) AND resampled preferences
    across higher beta_BT panels (Theorem 6).
  - Reference policy pi_ref from completion log-probs (SmolLM2-135M locally, Qwen/Qwen3.5-0.8B
    on cloud GPU), with automatic caching, entropy/collapse diagnostics, and optional uniform
    and length-normalized ablations.

Usage:
  # Local CPU dry-run (reuses results/phase3_local_test/ + SmolLM2-135M):
  python run_phase4a_distortion.py --mode local-test --also-uniform-ref

  # Full-scale cloud GPU run (reuses results/phase3/ + Qwen/Qwen3.5-0.8B):
  python run_phase4a_distortion.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B --also-uniform-ref
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import Turn1Dataset, load_turn1_dataset
from src.distortion import (
    compute_distortion,
    solve_kl_ball_max_util,
    theorem6_distortion_curve,
    theorem6_peak_distortion,
)
from src.mythical_bt import fit_mythical_bt, gibbs_policy, resample_preferences
from src.ref_policy import (
    compute_completion_logprobs,
    load_causal_lm,
    logprobs_to_probs,
    ref_policy_diagnostics,
)
from src.user_bt import (
    LowRankUserBT,
    global_affine_normalize,
    per_query_normalize,
    unit_range_normalize_per_query,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


def compute_or_load_ref_logprobs(
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
    llm_name: str,
    cache_path: Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute (or load cached) raw completion log-probs and token lengths for eval prompts."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cached_prompts = data["prompt_indices"].tolist()
        if cached_prompts == eval_prompt_indices:
            raw_lps = {p: data[f"lp_{p}"] for p in eval_prompt_indices}
            lengths = {p: data[f"len_{p}"] for p in eval_prompt_indices}
            print(f"Loaded cached pi_ref log-probs for {len(eval_prompt_indices)} prompts from {cache_path}")
            return raw_lps, lengths

    print(f"Computing completion log-probs with {llm_name} on {len(eval_prompt_indices)} prompts (device={device})...")
    t0 = time.perf_counter()
    model, tokenizer = load_causal_lm(llm_name, device=device)

    raw_lps: dict[int, np.ndarray] = {}
    lengths: dict[int, np.ndarray] = {}
    save_dict: dict[str, np.ndarray] = {"prompt_indices": np.array(eval_prompt_indices, dtype=np.int64)}

    for idx_i, p_idx in enumerate(eval_prompt_indices, start=1):
        lps, lens = compute_completion_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt=ds.prompts[p_idx],
            completions=ds.prompt_responses[p_idx],
            batch_size=batch_size,
            device=device,
            max_length=max_length,
            length_normalize=False,
        )
        raw_lps[p_idx] = lps
        lengths[p_idx] = lens
        save_dict[f"lp_{p_idx}"] = lps
        save_dict[f"len_{p_idx}"] = lens
        if idx_i % 50 == 0 or idx_i == len(eval_prompt_indices):
            print(f"  Scored {idx_i}/{len(eval_prompt_indices)} prompts ({time.perf_counter() - t0:.1f}s)...", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **save_dict)

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return raw_lps, lengths


def build_ref_distributions(
    eval_prompt_indices: list[int],
    raw_lps: dict[int, np.ndarray],
    lengths: dict[int, np.ndarray],
    ref_mode: str,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Convert raw log-probs into discrete pi_ref(.|x) distributions + diagnostics table."""
    pi_refs: dict[int, np.ndarray] = {}
    diag_rows = []
    for p_idx in eval_prompt_indices:
        lp = raw_lps[p_idx]
        lens = lengths[p_idx]
        if ref_mode == "llm":
            probs = logprobs_to_probs(lp)
        elif ref_mode == "llm_len_norm":
            norm_lp = lp / np.maximum(lens.astype(np.float64), 1.0)
            probs = logprobs_to_probs(norm_lp)
        elif ref_mode == "uniform":
            probs = np.full(len(lp), 1.0 / len(lp), dtype=np.float64)
        else:
            raise ValueError(f"Unknown ref_mode={ref_mode!r}")

        pi_refs[p_idx] = probs
        d = ref_policy_diagnostics(probs)
        d["prompt_idx"] = p_idx
        d["ref_mode"] = ref_mode
        d["median_comp_tokens"] = float(np.median(lens))
        diag_rows.append(d)

    return pi_refs, pd.DataFrame(diag_rows)


def plot_phase4a_scatter(
    df: pd.DataFrame,
    save_path: Path,
    title: str,
) -> Path:
    """Plot multi-panel empirical distortion vs. tau(x) overlaid with exact Theorem 6 curve.

    Each blue dot is one (query x, KL-regularization lambda) evaluation:
      x = tau(x) = D_KL(pi_lam(.|x) || pi_ref(.|x))
      y = empirical distortion = max_{pi* in B_tau(pi_ref)} AvgUtil(pi*) / AvgUtil(pi_lam)
    The red curve is the exact Theorem 6 theoretical distortion curve dist_Thm6(tau; beta_BT),
    rising from 1.0 at tau=0 to sinh(beta_BT)/beta_BT at tau=ln(2).
    """
    panel_order = (
        df[["panel_order", "panel_label", "beta_bt"]]
        .drop_duplicates()
        .sort_values("panel_order")
        .to_dict("records")
    )
    n_panels = len(panel_order)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 4.4), dpi=160, squeeze=False)
    axes_flat = axes[0]

    for ax, info in zip(axes_flat, panel_order):
        label = info["panel_label"]
        beta_val = float(info["beta_bt"])
        sub = df[df["panel_label"] == label]
        finite = sub[np.isfinite(sub["distortion"])]

        ax.scatter(
            finite["tau"],
            finite["distortion"],
            s=18,
            alpha=0.55,
            color="#2563eb",
            edgecolors="none",
            label="Empirical RLHF (per query)",
        )

        # Compute exact Theorem 6 theoretical curve over this panel's tau range
        max_tau = max(1.0, float(finite["tau"].max()) if len(finite) > 0 else 1.0)
        ln2 = float(np.log(2.0))
        tau_grid = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, min(ln2, max_tau), 120),
                    np.linspace(min(ln2, max_tau), max_tau, 80),
                ]
            )
        )
        peak = theorem6_peak_distortion(beta_val)

        # Evaluate theoretical curve at each empirical point's tau (exclude trivial tau=0 where both equal 1.0)
        tau_arr = finite["tau"].to_numpy(dtype=np.float64)
        dist_arr = finite["distortion"].to_numpy(dtype=np.float64)
        emp_th = theorem6_distortion_curve(tau_arr, beta_val)
        n_above = int(np.sum((tau_arr > 1e-6) & (dist_arr > emp_th + 1e-6)))

        # Only plot theoretical curve directly when peak is finite & plottable (< 1e6)
        if np.isfinite(peak) and peak < 1e6:
            th_vals = theorem6_distortion_curve(tau_grid, beta_val)
            ax.plot(
                tau_grid,
                th_vals,
                color="#dc2626",
                linewidth=2.0,
                label=f"Theorem 6 (peak={peak:.2f})",
            )
            if peak > 6.0:
                ax.set_yscale("log")
        else:
            ax.plot([], [], color="#dc2626", linewidth=2.0, label=f"Theorem 6 (peak={peak:.1e})")

        ax.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.1)
        max_d = float(finite["distortion"].max()) if len(finite) > 0 else 1.0
        ax.set_title(
            f"{label}\n(emp max={max_d:.2f} | >=Thm6: {n_above}/{len(finite)})",
            fontsize=10.0,
            fontweight="bold",
        )
        ax.set_xlabel(r"Realized KL radius $\tau(x) = D_{\mathrm{KL}}(\pi_\lambda \| \pi_{\mathrm{ref}})$")
        ax.set_ylabel("Distortion")
        ax.legend(fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.04)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_distortion_vs_beta(
    fixed_tau_df: pd.DataFrame,
    lam_sweep_df: pd.DataFrame,
    save_path: Path,
    title: str,
) -> Path:
    """Two-panel Distortion vs. beta_BT figure comparing empirical scaling to Theorem 6.

    Left panel:  Exact fixed KL-budget curves (tau in tau_targets), showing worst-case (max)
                 and median distortion across queries as a function of beta_BT vs. Theorem 6.
    Right panel: Per-query scatter from lambda-regularized policies (where tau varies across
                 queries and beta_BT, mirroring DPO), colored continuously by realized tau(x).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=160)

    betas_sorted = sorted(fixed_tau_df["beta_bt"].unique())
    beta_grid = np.linspace(max(0.1, min(betas_sorted)), max(betas_sorted), 200)
    thm6_peak = np.array([theorem6_peak_distortion(float(b)) for b in beta_grid])

    # --- Panel 1: Exact fixed-tau curves vs. beta_BT ---
    ax = axes[0]
    tau_targets = sorted(fixed_tau_df["tau_target"].unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.12, 0.90, len(tau_targets))]

    for color, t_val in zip(colors, tau_targets):
        sub = fixed_tau_df[fixed_tau_df["tau_target"] == t_val]
        agg = (
            sub.groupby("beta_bt")["distortion"]
            .agg(
                max_d=lambda s: float(np.max(s[np.isfinite(s)])),
                med_d=lambda s: float(np.median(s[np.isfinite(s)])),
            )
            .reset_index()
            .sort_values("beta_bt")
        )
        ax.plot(
            agg["beta_bt"],
            agg["max_d"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=rf"Empirical max ($\tau={t_val:g}$)",
        )
        ax.plot(
            agg["beta_bt"],
            agg["med_d"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            color=color,
        )

    ax.plot(
        beta_grid,
        thm6_peak,
        color="#dc2626",
        linewidth=2.3,
        label=r"Theorem 6 worst-case: $\sinh(\beta)/\beta$",
    )
    ax.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel(r"Bradley–Terry inverse temperature $\beta_{\mathrm{BT}}$", fontsize=10.5)
    ax.set_ylabel("Distortion (log scale)", fontsize=10.5)
    ax.set_title("Fixed KL Budget: Worst-Case (solid) & Median (dashed)", fontweight="bold")
    ax.legend(fontsize=8, frameon=True, loc="upper left")

    # --- Panel 2: Per-query scatter colored by realized tau(x) (DPO-style view) ---
    ax = axes[1]
    resamp = lam_sweep_df[lam_sweep_df["is_resampled"]].copy()
    finite = resamp[np.isfinite(resamp["distortion"]) & (resamp["tau"] > 1e-6)]
    rng = np.random.default_rng(0)
    x_jitter = finite["beta_bt"].to_numpy(dtype=np.float64) + rng.uniform(-0.14, 0.14, size=len(finite))

    sc = ax.scatter(
        x_jitter,
        finite["distortion"],
        c=finite["tau"],
        cmap="viridis",
        s=18,
        alpha=0.70,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"Realized KL radius $\tau(x) = D_{\mathrm{KL}}(\pi_\lambda \| \pi_{\mathrm{ref}})$", fontsize=9.5)

    ax.plot(
        beta_grid,
        thm6_peak,
        color="#dc2626",
        linewidth=2.3,
        label=r"Theorem 6 worst-case: $\sinh(\beta)/\beta$",
    )
    ax.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel(r"Bradley–Terry inverse temperature $\beta_{\mathrm{BT}}$ (jittered)", fontsize=10.5)
    ax.set_ylabel("Distortion (log scale)", fontsize=10.5)
    ax.set_title(r"$\lambda$-Regularized Policies: Per-Query Points Colored by Realized $\tau(x)$", fontweight="bold")
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def evaluate_distortion_grid(
    ds: Turn1Dataset,
    feats: torch.Tensor,
    u_norm: np.ndarray,
    beta_nat: float,
    eval_prompt_indices: list[int],
    pi_refs: dict[int, np.ndarray],
    betas_resample: list[float],
    tau_targets: list[float],
    lambdas: np.ndarray,
    myth_wd: float,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate empirical KL-constrained distortion under BOTH exact fixed-tau budgets and lambda sweeps."""
    avg_utils: dict[int, np.ndarray] = {
        p_idx: np.mean(u_norm[:, ds.prompt_global_xy_indices[p_idx]], axis=0)
        for p_idx in eval_prompt_indices
    }

    # Report utility ceiling diagnostic: max possible distortion = max(AvgUtil) / min(AvgUtil) per prompt
    ceilings = [float(np.max(u) / max(np.min(u), 1e-12)) for u in avg_utils.values()]
    print(
        f"  AvgUtil range ceiling (max_y AvgUtil / min_y AvgUtil across eval prompts): "
        f"median={np.median(ceilings):.3f}, max={np.max(ceilings):.3f}"
    )

    settings: list[tuple[int, str, float, bool, list[tuple[int, int, int, int]]]] = [
        (0, f"Observed Data (nat beta={beta_nat:.1f})", beta_nat, False, ds.pairs)
    ]
    for idx_b, beta_bt in enumerate(betas_resample, start=1):
        pairs_b = resample_preferences(u_norm, ds.pairs, beta_bt=beta_bt, seed=seed + idx_b)
        settings.append((idx_b, f"Resampled beta_BT={beta_bt:g}", float(beta_bt), True, pairs_b))

    lam_records = []
    tau_records = []
    for panel_order, panel_label, beta_val, is_resampled, pairs_b in settings:
        myth_model, _ = fit_mythical_bt(
            feats,
            pairs_b,
            lr=lr,
            weight_decay=myth_wd,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        with torch.no_grad():
            r_all = myth_model.score(feats.to(device)).cpu().numpy().astype(np.float64)

        for p_idx in eval_prompt_indices:
            g_idxs = ds.prompt_global_xy_indices[p_idx]
            avg_u = avg_utils[p_idx]
            r_p = r_all[g_idxs]
            p0 = pi_refs[p_idx]

            # 1. Fixed-tau KL-constrained RLHF policies (same tau across all beta_BT!)
            if is_resampled:
                for t_val in tau_targets:
                    pi_tau, _ = solve_kl_ball_max_util(r_p, p0, tau=float(t_val))
                    res = compute_distortion(avg_u, pi_tau, p0)
                    tau_records.append(
                        {
                            "beta_bt": beta_val,
                            "tau_target": float(t_val),
                            "prompt_idx": p_idx,
                            "num_candidates": len(g_idxs),
                            **res,
                        }
                    )

            # 2. Lambda-regularized policies (realized tau varies with prompt & beta_BT)
            for lam in lambdas:
                pi_lam = gibbs_policy(r_p, p0, lam=float(lam))
                res = compute_distortion(avg_u, pi_lam, p0)
                lam_records.append(
                    {
                        "panel_order": panel_order,
                        "panel_label": panel_label,
                        "beta_bt": beta_val,
                        "is_resampled": is_resampled,
                        "prompt_idx": p_idx,
                        "num_candidates": len(g_idxs),
                        "lam": float(lam),
                        **res,
                    }
                )
    return pd.DataFrame(tau_records), pd.DataFrame(lam_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4A KL-constrained distortion vs. tau analysis.")
    parser.add_argument("--mode", choices=["local-test", "full"], default="local-test")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--llm", type=str, default=None, help="HF causal LM for pi_ref.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rank", type=int, default=None, help="UserBT rank to load (default: best from Phase 3).")
    parser.add_argument("--norm-mode", choices=["per-query", "unit-range", "global"], default="per-query")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    parser.add_argument("--tau-targets", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--num-lambdas", type=int, default=15)
    parser.add_argument("--lam-min", type=float, default=0.02)
    parser.add_argument("--lam-max", type=float, default=20.0)
    parser.add_argument("--min-user-prefs-gt", type=int, default=20)
    parser.add_argument("--eval-min-users", type=int, default=20)
    parser.add_argument("--ref-batch-size", type=int, default=None)
    parser.add_argument("--ref-max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--also-uniform-ref", action="store_true", help="Also run uniform pi_ref ablation.")
    parser.add_argument("--also-len-norm-ref", action="store_true", help="Also run length-normalized LLM pi_ref ablation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    phase3_dir = args.phase3_dir or (ROOT / "results" / ("phase3_local_test" if args.mode == "local-test" else "phase3"))
    out_dir = args.out_dir or (ROOT / "results" / ("phase4a_local_test" if args.mode == "local-test" else "phase4a"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_p3 = json.loads((phase3_dir / "phase3_summary.json").read_text(encoding="utf-8"))
    myth_wd = float(summary_p3.get("best_mythical_weight_decay", 1e-2))

    # 1. Load exact same Turn-1 dataset and cached ModernBERT CLS features from Phase 3
    if args.mode == "local-test":
        fixture_csv = phase3_dir / "local_fixture_en.csv"
        ds = load_turn1_dataset(fixture_csv, min_user_prefs_gt=1)
        eval_prompt_indices = ds.prompts_with_min_users(3)
        llm_name = args.llm or "HuggingFaceTB/SmolLM2-135M"
        ref_bs = args.ref_batch_size or 8
        ref_max_len = args.ref_max_length or 256
        epochs = args.epochs or 80
        train_bs = min(args.batch_size, 128)
    else:
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=args.min_user_prefs_gt)
        eval_prompt_indices = ds.prompts_with_min_users(args.eval_min_users)
        llm_name = args.llm or "Qwen/Qwen3.5-0.8B"
        ref_bs = args.ref_batch_size or (16 if device.type == "cuda" else 4)
        ref_max_len = args.ref_max_length or 1024
        epochs = args.epochs or 120
        train_bs = args.batch_size

    feats = torch.load(phase3_dir / "modernbert_cls_features.pt", map_location="cpu", weights_only=True)
    assert feats.shape[0] == ds.num_unique_xy, f"Feature mismatch: {feats.shape[0]} vs {ds.num_unique_xy}"

    # 2. Load trained UserBT checkpoint and compute global affine normalized utilities u_norm in [0, 1]
    ckpt_path = phase3_dir / (f"user_bt_rank{args.rank}_best.pt" if args.rank is not None else "user_bt_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    user_bt = LowRankUserBT(
        num_users=int(ckpt["num_users"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        rank=int(ckpt["rank"]),
    )
    user_bt.load_state_dict(ckpt["state_dict"])
    user_bt.eval()

    with torch.no_grad():
        raw_scores = user_bt.score_all_users(feats).numpy()
    _, beta_nat, _, _ = global_affine_normalize(raw_scores)
    if args.norm_mode == "per-query":
        u_norm, query_spans = per_query_normalize(raw_scores, ds.prompt_global_xy_indices)
        eval_spans = query_spans[eval_prompt_indices]
        print(
            f"Per-query shared spans M(x) on eval prompts: "
            f"min={eval_spans.min():.2f}, median={np.median(eval_spans):.2f}, max={eval_spans.max():.2f}"
        )
    elif args.norm_mode == "unit-range":
        u_norm = unit_range_normalize_per_query(raw_scores, ds.prompt_global_xy_indices)
    else:
        u_norm, _, _, _ = global_affine_normalize(raw_scores)

    print(f"=== Phase 4A ({args.mode}, norm={args.norm_mode}) on device={device} ===")
    print(f"Loaded {ckpt_path.name}: rank={ckpt['rank']}, wd={ckpt['weight_decay']}, beta_BT_nat={beta_nat:.2f}")
    print(f"Eval prompts={len(eval_prompt_indices)}, Users in AvgUtil={ds.num_users}, LLM={llm_name}")

    # 3. Compute or load cached completion log-probs for pi_ref
    safe_llm = llm_name.replace("/", "__")
    ref_cache = out_dir / f"pi_ref_logprobs_{safe_llm}.npz"
    raw_lps, lengths = compute_or_load_ref_logprobs(
        ds=ds,
        eval_prompt_indices=eval_prompt_indices,
        llm_name=llm_name,
        cache_path=ref_cache,
        batch_size=ref_bs,
        max_length=ref_max_len,
        device=device,
    )

    ref_modes = ["llm"]
    if args.also_len_norm_ref:
        ref_modes.append("llm_len_norm")
    if args.also_uniform_ref:
        ref_modes.append("uniform")

    lambdas = np.geomspace(args.lam_min, args.lam_max, args.num_lambdas)
    all_diag_dfs = []

    for ref_mode in ref_modes:
        pi_refs, diag_df = build_ref_distributions(eval_prompt_indices, raw_lps, lengths, ref_mode)
        all_diag_dfs.append(diag_df)

        n_collapsed = int(diag_df["is_collapsed"].sum())
        print(
            f"\n--- ref_mode={ref_mode}: median_norm_entropy={diag_df['normalized_entropy'].median():.3f}, "
            f"median_max_prob={diag_df['max_prob'].median():.3f}, collapsed={n_collapsed}/{len(diag_df)} ---"
        )

        tau_df, lam_df = evaluate_distortion_grid(
            ds=ds,
            feats=feats,
            u_norm=u_norm,
            beta_nat=beta_nat,
            eval_prompt_indices=eval_prompt_indices,
            pi_refs=pi_refs,
            betas_resample=args.betas,
            tau_targets=args.tau_targets,
            lambdas=lambdas,
            myth_wd=myth_wd,
            lr=args.lr,
            epochs=epochs,
            batch_size=train_bs,
            seed=args.seed,
            device=device,
        )
        suffix = f"{ref_mode}_{args.norm_mode.replace('-', '_')}"
        tau_df["ref_mode"] = ref_mode
        tau_df["norm_mode"] = args.norm_mode
        lam_df["ref_mode"] = ref_mode
        lam_df["norm_mode"] = args.norm_mode
        tau_df.to_csv(out_dir / f"phase4a_fixed_tau_{suffix}.csv", index=False)
        lam_df.to_csv(out_dir / f"phase4a_distortion_{suffix}.csv", index=False)

        # Print per-beta summary
        panel_sum = (
            tau_df.groupby("beta_bt")
            .agg(
                median_dist=("distortion", "median"),
                max_dist=("distortion", lambda s: float(np.max(s[np.isfinite(s)]))),
            )
            .reset_index()
        )
        panel_sum["thm6_peak"] = [theorem6_peak_distortion(float(b)) for b in panel_sum["beta_bt"]]
        print(panel_sum.round(4).to_string(index=False))

        ref_title = f"LLM={llm_name}" if ref_mode != "uniform" else "Uniform pi_ref"
        if ref_mode == "llm_len_norm":
            ref_title += " (length-normalized)"
        beta_plot_path = plot_distortion_vs_beta(
            fixed_tau_df=tau_df,
            lam_sweep_df=lam_df,
            save_path=out_dir / f"phase4a_distortion_vs_beta_{suffix}.png",
            title=f"Phase 4A: Empirical Distortion vs. beta_BT ({ref_title}, {args.norm_mode}, rank={ckpt['rank']})",
        )
        print(f"Saved distortion-vs-beta plot to: {beta_plot_path}")

    pd.concat(all_diag_dfs, ignore_index=True).to_csv(out_dir / "pi_ref_diagnostics.csv", index=False)
    print(f"\nPhase 4A completed in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
