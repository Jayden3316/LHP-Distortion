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
from src.mythical_bt import (
    exact_population_query_bt_mle,
    fit_mythical_bt,
    fit_mythical_bt_soft,
    gibbs_policy,
    resample_preferences,
)
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

    print(f"Computing completion log-probs with {llm_name} on {len(eval_prompt_indices)} prompts (device={device}, batch_size={batch_size})...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
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
    """Two-panel Distortion vs. beta_BT figure with tau-dependent Theorem 6 curves.

    Left panel:  Exact fixed KL-budget curves (one color per tau). Solid + markers = empirical
                 worst-case (max across queries; queries need not match across beta_BT).
                 Dashed (same color) = exact Theorem 6 curve dist_Thm6(tau; beta_BT) for that tau.
    Right panel: Per-query scatter from lambda-regularized policies (DPO-style view), colored
                 continuously by realized tau(x), overlaid with tau-specific Theorem 6 curves.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), dpi=160)

    betas_sorted = sorted(fixed_tau_df["beta_bt"].unique())
    beta_grid = np.linspace(max(0.1, min(betas_sorted)), max(betas_sorted), 200)

    # --- Panel 1: Exact fixed-tau empirical max + tau-matched Theorem 6 curves ---
    ax = axes[0]
    tau_targets = sorted(fixed_tau_df["tau_target"].unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.10, 0.90, len(tau_targets))]

    plotted_saturated_thm6 = False
    ln2 = float(np.log(2.0))
    for color, t_val in zip(colors, tau_targets):
        sub = fixed_tau_df[fixed_tau_df["tau_target"] == t_val]
        agg = (
            sub.groupby("beta_bt")["distortion"]
            .agg(max_d=lambda s: float(np.max(s[np.isfinite(s)])))
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

        # Exact Theorem 6 curve at THIS tau: for tau < ln 2 it is strictly below sinh(beta)/beta!
        th_vals = np.array([theorem6_distortion_curve(np.array([t_val]), float(b))[0] for b in beta_grid])
        if t_val < ln2 - 1e-6:
            ax.plot(
                beta_grid,
                th_vals,
                linestyle="--",
                linewidth=1.8,
                color=color,
                label=rf"Theorem 6 ($\tau={t_val:g}$)",
            )
        elif not plotted_saturated_thm6:
            ax.plot(
                beta_grid,
                th_vals,
                linestyle="--",
                linewidth=2.2,
                color="#dc2626",
                label=r"Theorem 6 ($\tau \geq \ln 2$: $\sinh(\beta)/\beta$)",
            )
            plotted_saturated_thm6 = True

    ax.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel(r"Bradley–Terry inverse temperature $\beta_{\mathrm{BT}}$", fontsize=10.5)
    ax.set_ylabel("Distortion (log scale)", fontsize=10.5)
    ax.set_title(r"Fixed KL Budget: Empirical Max (solid) vs. Theorem 6 at Same $\tau$ (dashed)", fontweight="bold")
    ax.legend(fontsize=7.8, frameon=True, loc="upper left", ncol=2)

    # --- Panel 2: Per-query scatter colored by realized tau(x) + tau-matched Theorem 6 curves ---
    ax = axes[1]
    resamp = lam_sweep_df[lam_sweep_df["is_resampled"]].copy()
    finite = resamp[np.isfinite(resamp["distortion"]) & (resamp["tau"] > 1e-6)]
    rng = np.random.default_rng(0)
    x_jitter = finite["beta_bt"].to_numpy(dtype=np.float64) + rng.uniform(-0.14, 0.14, size=len(finite))

    tau_max_cbar = max(float(finite["tau"].max()) if len(finite) > 0 else 1.0, 1.0)
    norm = plt.Normalize(vmin=0.0, vmax=tau_max_cbar)
    sc = ax.scatter(
        x_jitter,
        finite["distortion"],
        c=finite["tau"],
        cmap="viridis",
        norm=norm,
        s=18,
        alpha=0.70,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"Realized KL radius $\tau(x) = D_{\mathrm{KL}}(\pi_\lambda \| \pi_{\mathrm{ref}})$", fontsize=9.5)

    # Overlay Theorem 6 curves at representative tau values using matching colorbar colors
    for t_ref, lbl, col in [
        (0.2, r"Theorem 6 ($\tau=0.2$)", cmap(norm(0.2))),
        (0.5, r"Theorem 6 ($\tau=0.5$)", cmap(norm(0.5))),
        (ln2, r"Theorem 6 ($\tau \geq \ln 2$: $\sinh(\beta)/\beta$)", "#dc2626"),
    ]:
        th_vals = np.array([theorem6_distortion_curve(np.array([t_ref]), float(b))[0] for b in beta_grid])
        ax.plot(beta_grid, th_vals, linestyle="--", linewidth=2.0, color=col, label=lbl)

    ax.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel(r"Bradley–Terry inverse temperature $\beta_{\mathrm{BT}}$ (jittered)", fontsize=10.5)
    ax.set_ylabel("Distortion (log scale)", fontsize=10.5)
    ax.set_title(r"$\lambda$-Regularized Policies: Per-Query Points Colored by Realized $\tau(x)$", fontweight="bold")
    ax.legend(fontsize=8.2, frameon=True, loc="upper left")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def inspect_top_distortion_queries(
    ds: Turn1Dataset,
    u_norm: np.ndarray,
    pi_refs: dict[int, np.ndarray],
    lengths: dict[int, np.ndarray],
    r_all_by_beta: dict[float, np.ndarray],
    lam_df: pd.DataFrame,
    out_txt_path: Path,
    top_k: int = 3,
    target_beta: float | None = None,
) -> str:
    """Inspect top-K highest-distortion queries qualitatively: prompt, user utility stats, pi_ref, pi_RLHF, pi*."""
    sub = lam_df[lam_df["is_resampled"] & np.isfinite(lam_df["distortion"]) & (lam_df["tau"] > 1e-6)].copy()
    if target_beta is not None and (sub["beta_bt"] == target_beta).any():
        sub = sub[sub["beta_bt"] == target_beta]

    # Pick top-K distinct prompts by their peak empirical distortion
    best_per_prompt = (
        sub.sort_values("distortion", ascending=False)
        .drop_duplicates(subset=["prompt_idx"])
        .head(top_k)
    )

    lines: list[str] = []
    header_beta = f"beta_BT={target_beta:g}" if target_beta is not None else "all resampled beta_BT"
    lines.append(f"=== Qualitative Inspection: Top-{len(best_per_prompt)} Highest-Distortion Queries ({header_beta}) ===")

    for rank_i, (_, row) in enumerate(best_per_prompt.iterrows(), start=1):
        p_idx = int(row["prompt_idx"])
        beta_val = float(row["beta_bt"])
        lam_val = float(row["lam"])
        tau_val = float(row["tau"])
        dist_val = float(row["distortion"])

        g_idxs = ds.prompt_global_xy_indices[p_idx]
        U_p = u_norm[:, g_idxs]  # [N_users, |A(x)|]
        avg_u = np.mean(U_p, axis=0)
        std_u = np.std(U_p, axis=0)
        top1_users = np.argmax(U_p, axis=1)
        top1_share = np.bincount(top1_users, minlength=len(g_idxs)) / float(U_p.shape[0])

        p0 = pi_refs[p_idx]
        comp_lens = lengths[p_idx]
        r_p = r_all_by_beta[beta_val][g_idxs]
        pi_lam = gibbs_policy(r_p, p0, lam=lam_val)
        pi_star, opt_util = solve_kl_ball_max_util(avg_u, p0, tau=tau_val)
        pol_util = float(np.dot(pi_lam, avg_u))

        diag = ref_policy_diagnostics(p0)
        lines.append("")
        lines.append("-" * 96)
        lines.append(
            f"[Case #{rank_i}] prompt_idx={p_idx} | beta_BT={beta_val:g}, lam={lam_val:.4g}, "
            f"tau={tau_val:.3f} nats | distortion={dist_val:.3f} (opt_util={opt_util:.3f} vs policy_util={pol_util:.3f})"
        )
        lines.append(f"Prompt ({len(ds.prompt_annotators[p_idx])} annotators, |A(x)|={len(g_idxs)} candidates):")
        lines.append(f"  {ds.prompts[p_idx]}")
        lines.append(
            f"User Utility Stats across {U_p.shape[0]} users: "
            f"AvgUtil range=[{avg_u.min():.3f}, {avg_u.max():.3f}] (ceiling={avg_u.max()/max(avg_u.min(),1e-12):.2f}), "
            f"mean cross-user std={std_u.mean():.3f}, distinct top-1 responses={int(np.sum(top1_share > 0))}/{len(g_idxs)}"
        )
        lines.append(
            f"pi_ref Stats: norm_entropy={diag['normalized_entropy']:.3f}, "
            f"max_prob={diag['max_prob']:.4f}, min_prob={diag['min_prob']:.4f}, "
            f"effective_support={diag['effective_support']:.1f}/{len(g_idxs)}"
        )

        # Highlight key candidates: top by pi* (social-welfare opt), top by pi_RLHF (mythical choice), top by pi_ref
        idx_star = int(np.argmax(pi_star))
        idx_rlhf = int(np.argmax(pi_lam))
        idx_ref = int(np.argmax(p0))
        idx_avg = int(np.argmax(avg_u))

        highlight_order: list[tuple[int, list[str]]] = []
        seen: dict[int, list[str]] = {}
        for idx_c, tag in [
            (idx_star, "TOP pi* (KL-ball opt)"),
            (idx_avg, "MAX AvgUtil"),
            (idx_rlhf, "TOP pi_RLHF (mythical)"),
            (idx_ref, "TOP pi_ref"),
        ]:
            if idx_c not in seen:
                seen[idx_c] = [tag]
                highlight_order.append((idx_c, seen[idx_c]))
            else:
                seen[idx_c].append(tag)

        lines.append("Key Candidate Responses:")
        for c_idx, tags in highlight_order:
            text_clean = ds.prompt_responses[p_idx][c_idx].replace("\n", " ").strip()
            if len(text_clean) > 260:
                text_clean = text_clean[:260] + "..."
            lines.append(
                f"  * cand#{c_idx:<3} [{', '.join(tags)}]\n"
                f"      AvgUtil={avg_u[c_idx]:.3f} (user_std={std_u[c_idx]:.3f}, user_top1={100*top1_share[c_idx]:.1f}%) | "
                f"r_MLE={r_p[c_idx]:+.3f} | len={int(comp_lens[c_idx])} tok\n"
                f"      pi_ref={p0[c_idx]:.4f}  ->  pi_RLHF={pi_lam[c_idx]:.4f}  vs  pi*={pi_star[c_idx]:.4f}\n"
                f"      Text: {text_clean}"
            )

    report_text = "\n".join(lines) + "\n"
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text(report_text, encoding="utf-8")
    return report_text


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, np.ndarray]]:
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
    r_all_by_beta: dict[float, np.ndarray] = {}
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
        if is_resampled:
            r_all_by_beta[beta_val] = r_all

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
    return pd.DataFrame(tau_records), pd.DataFrame(lam_records), r_all_by_beta


def evaluate_three_way_decomposition(
    ds: Turn1Dataset,
    feats: torch.Tensor,
    u_norm: np.ndarray,
    eval_prompt_indices: list[int],
    pi_refs: dict[int, np.ndarray],
    betas_resample: list[float],
    lambdas: np.ndarray,
    myth_wd: float,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Compare ExactTab vs. SoftLin vs. HardLin distortion across beta_BT on the same eval prompts.

    Cleanly separates:
      - `ExactTab`: exact tabular per-query population BT-MLE (pure social-choice distortion, Thm 6)
      - `SoftLin`:  768-dim linear MythicalUserBT trained on exact soft probabilities p_{iab} (adds feature misspecification)
      - `HardLin`:  768-dim linear MythicalUserBT trained on 1-shot hard Bernoulli flips (adds finite-sample label noise)
    """
    avg_utils: dict[int, np.ndarray] = {
        p_idx: np.mean(u_norm[:, ds.prompt_global_xy_indices[p_idx]], axis=0)
        for p_idx in eval_prompt_indices
    }

    rows: list[dict] = []
    for idx_b, beta_bt in enumerate(betas_resample, start=1):
        # 1. SoftLin: exact expected soft-label cross-entropy
        soft_model, _ = fit_mythical_bt_soft(
            feats,
            ds.pairs,
            u_norm=u_norm,
            beta_bt=float(beta_bt),
            lr=lr,
            weight_decay=myth_wd,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        with torch.no_grad():
            r_soft_all = soft_model.score(feats.to(device)).cpu().numpy().astype(np.float64)

        # 2. HardLin: 1-shot hard Bernoulli resampled labels
        pairs_b = resample_preferences(u_norm, ds.pairs, beta_bt=float(beta_bt), seed=seed + idx_b)
        hard_model, _ = fit_mythical_bt(
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
            r_hard_all = hard_model.score(feats.to(device)).cpu().numpy().astype(np.float64)

        # 3. Evaluate max-over-lambda distortion per prompt for ExactTab, SoftLin, and HardLin
        for p_idx in eval_prompt_indices:
            g_idxs = ds.prompt_global_xy_indices[p_idx]
            au = avg_utils[p_idx]
            p0 = pi_refs[p_idx]
            r_tab = exact_population_query_bt_mle(u_norm[:, g_idxs], beta_bt=float(beta_bt))
            for est_name, r_vec in [
                ("ExactTab", r_tab),
                ("SoftLin", r_soft_all[g_idxs]),
                ("HardLin", r_hard_all[g_idxs]),
            ]:
                best_d = 1.0
                best_tau = 0.0
                for lam in lambdas:
                    res = compute_distortion(au, gibbs_policy(r_vec, p0, lam=float(lam)), p0)
                    if np.isfinite(res["distortion"]) and res["distortion"] > best_d:
                        best_d = float(res["distortion"])
                        best_tau = float(res["tau"])
                rows.append(
                    {
                        "beta_bt": float(beta_bt),
                        "estimator": est_name,
                        "prompt_idx": int(p_idx),
                        "max_distortion": best_d,
                        "tau_at_max": best_tau,
                    }
                )
    return pd.DataFrame(rows)


def plot_three_way_decomposition(
    decomp_df: pd.DataFrame,
    save_path: Path,
    title: str,
) -> Path:
    """Two-panel plot separating ExactTab (social choice) vs. SoftLin (misspecification) vs. HardLin (label noise)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), dpi=160)

    betas_sorted = sorted(decomp_df["beta_bt"].unique())
    beta_grid = np.linspace(max(0.1, min(betas_sorted)), max(betas_sorted), 200)
    thm6_vals = [theorem6_peak_distortion(float(b)) for b in beta_grid]

    styles = {
        "ExactTab": ("#059669", "-o", "ExactTab (Exact pop. BT-MLE: pure social choice)"),
        "SoftLin": ("#2563eb", "-s", "SoftLin (Soft-label linear BT: + model misspec.)"),
        "HardLin": ("#dc2626", "-^", "HardLin (1-shot resampled BT: + coin-flip noise)"),
    }

    # Panel 1: Median distortion across prompts
    ax = axes[0]
    for est, (col, fmt, lbl) in styles.items():
        sub = decomp_df[decomp_df["estimator"] == est]
        meds = [float(sub[sub["beta_bt"] == b]["max_distortion"].median()) for b in betas_sorted]
        ax.plot(betas_sorted, meds, fmt, color=col, lw=2.2, ms=6.0, label=lbl)
    ax.axhline(1.0, color="#94a3b8", ls=":", lw=1.0)
    ax.set_xlabel(r"Bradley-Terry Inverse Temperature $\beta_{\mathrm{BT}}$", fontsize=10.5)
    ax.set_ylabel("Median Per-Query Distortion", fontsize=10.5)
    ax.set_title("(A) Median Distortion Across Eval Prompts", fontweight="bold")
    ax.legend(fontsize=8.4, frameon=True, loc="best")

    # Panel 2: Worst-case (max) & 90th-pct distortion vs. Theorem 6 peak (log scale)
    ax = axes[1]
    for est, (col, fmt, lbl) in styles.items():
        sub = decomp_df[decomp_df["estimator"] == est]
        maxs = [float(sub[sub["beta_bt"] == b]["max_distortion"].max()) for b in betas_sorted]
        p90s = [float(sub[sub["beta_bt"] == b]["max_distortion"].quantile(0.90)) for b in betas_sorted]
        ax.plot(betas_sorted, maxs, fmt, color=col, lw=2.2, ms=6.0, label=f"{est} max")
        ax.plot(betas_sorted, p90s, "--", color=col, lw=1.3, alpha=0.7, label=f"{est} 90th pct")
    ax.plot(beta_grid, thm6_vals, "-.", color="#0f172a", lw=2.0, label=r"Theorem 6 peak $\sinh(\beta)/\beta$")
    ax.set_yscale("log")
    ax.axhline(1.0, color="#94a3b8", ls=":", lw=1.0)
    ax.set_xlabel(r"Bradley-Terry Inverse Temperature $\beta_{\mathrm{BT}}$", fontsize=10.5)
    ax.set_ylabel("Max / 90th-Pct Distortion (log scale)", fontsize=10.5)
    ax.set_title("(B) Tail Distortion vs. Theorem 6 Worst-Case Bound", fontweight="bold")
    ax.legend(fontsize=8.0, frameon=True, loc="best", ncol=2)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


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
    parser.add_argument("--tau-plot-betas", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--tau-targets", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--top-k-inspect", type=int, default=3)
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
    parser.add_argument("--only-uniform-ref", action="store_true", help="Skip LLM pi_ref and evaluate uniform pi_ref only.")
    parser.add_argument("--three-way-decomp", action="store_true", help="Also run ExactTab vs. SoftLin vs. HardLin 3-way decomposition.")
    parser.add_argument("--only-three-way-decomp", action="store_true", help="Run ONLY the 3-way decomposition (skips standard grid).")
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
        if args.ref_batch_size is not None:
            ref_bs = args.ref_batch_size
        elif device.type == "cuda":
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            ref_bs = 4 if total_gb <= 18.0 else 8
        else:
            ref_bs = 4
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

    # 3. Compute or load cached completion log-probs for pi_ref (skipped if --only-uniform-ref)
    if args.only_uniform_ref:
        raw_lps = {p: np.zeros(len(ds.prompt_global_xy_indices[p]), dtype=np.float64) for p in eval_prompt_indices}
        lengths = {p: np.ones(len(ds.prompt_global_xy_indices[p]), dtype=np.int64) for p in eval_prompt_indices}
        ref_modes = ["uniform"]
    else:
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
        suffix = f"{ref_mode}_{args.norm_mode.replace('-', '_')}"
        ref_title = f"LLM={llm_name}" if ref_mode != "uniform" else "Uniform pi_ref"
        if ref_mode == "llm_len_norm":
            ref_title += " (length-normalized)"

        n_collapsed = int(diag_df["is_collapsed"].sum())
        print(
            f"\n--- ref_mode={ref_mode}: median_norm_entropy={diag_df['normalized_entropy'].median():.3f}, "
            f"median_max_prob={diag_df['max_prob'].median():.3f}, collapsed={n_collapsed}/{len(diag_df)} ---"
        )

        if not args.only_three_way_decomp:
            tau_df, lam_df, r_all_by_beta = evaluate_distortion_grid(
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

            # Plot 1: Distortion vs. beta_BT (with tau-matched Theorem 6 curves)
            beta_plot_path = plot_distortion_vs_beta(
                fixed_tau_df=tau_df,
                lam_sweep_df=lam_df,
                save_path=out_dir / f"phase4a_distortion_vs_beta_{suffix}.png",
                title=f"Phase 4A: Empirical Distortion vs. beta_BT ({ref_title}, {args.norm_mode}, rank={ckpt['rank']})",
            )
            print(f"Saved distortion-vs-beta plot to: {beta_plot_path}")

            # Plot 2: Faceted Distortion vs. tau across representative beta_BT panels
            tau_plot_mask = (~lam_df["is_resampled"]) | (lam_df["beta_bt"].isin(args.tau_plot_betas))
            tau_plot_path = plot_phase4a_scatter(
                df=lam_df[tau_plot_mask],
                save_path=out_dir / f"phase4a_distortion_vs_tau_{suffix}.png",
                title=f"Phase 4A: Empirical Distortion vs. tau ({ref_title}, {args.norm_mode}, rank={ckpt['rank']})",
            )
            print(f"Saved distortion-vs-tau plot to : {tau_plot_path}")

            # Qualitative inspection of top-K highest-distortion queries
            top_txt_path = out_dir / f"top_distortion_cases_{suffix}.txt"
            report_text = inspect_top_distortion_queries(
                ds=ds,
                u_norm=u_norm,
                pi_refs=pi_refs,
                lengths=lengths,
                r_all_by_beta=r_all_by_beta,
                lam_df=lam_df,
                out_txt_path=top_txt_path,
                top_k=args.top_k_inspect,
            )
            print(f"Saved qualitative cases to      : {top_txt_path}")
            if ref_mode in ("llm_len_norm", "llm") and args.top_k_inspect > 0:
                print(report_text)

        if args.three_way_decomp or args.only_three_way_decomp:
            print(f"\n  Running 3-way decomposition (ExactTab vs. SoftLin vs. HardLin) for ref_mode={ref_mode}...", flush=True)
            decomp_df = evaluate_three_way_decomposition(
                ds=ds,
                feats=feats,
                u_norm=u_norm,
                eval_prompt_indices=eval_prompt_indices,
                pi_refs=pi_refs,
                betas_resample=args.betas,
                lambdas=lambdas,
                myth_wd=myth_wd,
                lr=args.lr,
                epochs=epochs,
                batch_size=train_bs,
                seed=args.seed,
                device=device,
            )
            decomp_df["ref_mode"] = ref_mode
            decomp_df["norm_mode"] = args.norm_mode
            decomp_csv = out_dir / f"phase4a_3way_decomposition_{suffix}.csv"
            decomp_df.to_csv(decomp_csv, index=False)

            piv = decomp_df.groupby(["beta_bt", "estimator"])["max_distortion"].agg(["median", "max"]).unstack("estimator")
            print(piv.round(4).to_string())

            decomp_png = plot_three_way_decomposition(
                decomp_df=decomp_df,
                save_path=out_dir / f"phase4a_3way_decomposition_{suffix}.png",
                title=f"Phase 4A 3-Way Decomposition: Social Choice vs. Misspecification vs. Label Noise ({ref_title}, rank={ckpt['rank']})",
            )
            print(f"Saved 3-way decomposition CSV : {decomp_csv}")
            print(f"Saved 3-way decomposition plot: {decomp_png}")

    pd.concat(all_diag_dfs, ignore_index=True).to_csv(out_dir / "pi_ref_diagnostics.csv", index=False)
    print(f"\nPhase 4A completed in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
