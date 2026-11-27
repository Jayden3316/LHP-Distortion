"""Qualitative Evaluation: 2D PCA in User-Preference Space phi(x, y) = W psi(x, y).

Workflow:
  1. Select the top-N highest-distortion and top-N lowest-distortion queries from Phase 4A results.
  2. Map all dataset candidate responses for those queries through the trained Phase 3 linear head W
     into user space phi(x, y) in R^d and fit a 2D PCA.
  3. Generate k completions per selected query from the DPO-trained LM (and optionally the base/ref LM
     for before-vs-after comparison), encode them with frozen ModernBERT-base, project through W, and
     transform onto the exact same 2D PCA principal components.
  4. Plot the 2D PCA embedding distribution (colored by query distortion) and quantify distributional
     shifts between dataset/reference responses and DPO generations.

Usage:
  # Local CPU smoke run (uses local-test Phase 3/4A outputs + SmolLM2-135M as stand-in LM):
  python run_qualitative_pca.py --mode local-test --k 2 --top-n 3

  # Cloud GPU run (top-10 highest & lowest distortion queries, DPO checkpoint vs. base model):
  python run_qualitative_pca.py --mode full --device cuda --top-n 10 --k 8 \
      --dpo-model /path/to/dpo_checkpoint --ref-model Qwen/Qwen3.5-0.8B
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
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA

from src.data import Turn1Dataset, load_turn1_dataset
from src.distortion import compute_distortion
from src.encoder import FrozenModernBERTEncoder
from src.mythical_bt import MythicalUserBT, gibbs_policy
from src.ref_policy import load_causal_lm
from src.user_bt import LowRankUserBT, per_query_normalize


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


def select_extreme_distortion_queries(
    distortion_df: pd.DataFrame,
    top_n: int = 10,
    target_beta: float | None = None,
    agg_fn: str = "max",
) -> pd.DataFrame:
    """Select disjoint top-N highest-distortion and top-N lowest-distortion queries.

    Returns a DataFrame with columns: `prompt_idx`, `query_distortion`, `group`
    (`high_distortion` or `low_distortion`).
    """
    sub = distortion_df[np.isfinite(distortion_df["distortion"])].copy()
    if "is_resampled" in sub.columns:
        resamp = sub[sub["is_resampled"]]
        if len(resamp) > 0:
            sub = resamp
    if "tau" in sub.columns:
        pos_tau = sub[sub["tau"] > 1e-6]
        if len(pos_tau) > 0:
            sub = pos_tau
    if target_beta is not None and (sub["beta_bt"] == target_beta).any():
        sub = sub[sub["beta_bt"] == target_beta]

    if agg_fn == "max":
        per_q = sub.groupby("prompt_idx", as_index=False)["distortion"].max()
    elif agg_fn == "mean":
        per_q = sub.groupby("prompt_idx", as_index=False)["distortion"].mean()
    elif agg_fn == "median":
        per_q = sub.groupby("prompt_idx", as_index=False)["distortion"].median()
    else:
        raise ValueError(f"Unsupported agg_fn={agg_fn!r}")

    per_q = per_q.rename(columns={"distortion": "query_distortion"}).sort_values(
        ["query_distortion", "prompt_idx"], ascending=[False, True]
    ).reset_index(drop=True)

    n_avail = len(per_q)
    if n_avail == 0:
        raise ValueError("No valid queries found in distortion_df.")

    n_each = min(top_n, n_avail // 2) if n_avail >= 2 else 1
    high = per_q.head(n_each).copy()
    high["group"] = "high_distortion"

    remaining = per_q.iloc[n_each:]
    low = remaining.tail(min(top_n, len(remaining))).copy()
    low["group"] = "low_distortion"

    return pd.concat([high, low], ignore_index=True)


@torch.no_grad()
def generate_k_completions(
    model_name_or_path: str,
    prompts: list[str],
    k: int,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
) -> list[list[str]]:
    """Generate k stochastic completions per prompt from a causal LM."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_causal_lm(model_name_or_path, device=device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_gens: list[list[str]] = []
    for idx_i, prompt in enumerate(prompts, start=1):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = int(enc["input_ids"].shape[1])

        gen_ids = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
        comp_tokens = gen_ids[:, prompt_len:]
        texts = [t.strip() for t in tokenizer.batch_decode(comp_tokens, skip_special_tokens=True)]
        all_gens.append(texts)
        if idx_i % 5 == 0 or idx_i == len(prompts):
            print(f"  Generated {k}x completions for {idx_i}/{len(prompts)} prompts from {model_name_or_path}...", flush=True)

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return all_gens


def fit_and_project_user_space_pca(
    dataset_phi: np.ndarray,
    extra_phis: dict[str, np.ndarray],
) -> tuple[PCA, np.ndarray, dict[str, np.ndarray]]:
    """Fit 2D PCA on dataset user-space activations phi(x, y) and project extra model generations."""
    n_comp = min(2, dataset_phi.shape[0], dataset_phi.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    ds_coords = pca.fit_transform(dataset_phi)
    if n_comp == 1:
        ds_coords = np.hstack([ds_coords, np.zeros((ds_coords.shape[0], 1))])

    extra_coords: dict[str, np.ndarray] = {}
    for name, arr in extra_phis.items():
        if arr.shape[0] == 0:
            extra_coords[name] = np.empty((0, 2), dtype=np.float64)
            continue
        c = pca.transform(arr)
        if n_comp == 1:
            c = np.hstack([c, np.zeros((c.shape[0], 1))])
        extra_coords[name] = c
    return pca, ds_coords, extra_coords


def _add_kde_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    levels: int = 5,
    fill_alpha: float = 0.14,
    line_alpha: float = 0.85,
    linewidths: float = 1.4,
    label: str | None = None,
) -> None:
    """Overlay smooth 2D Gaussian KDE density contours if enough non-degenerate points exist."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 15 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return

    # Trim grid to 1st-99th percentile (+ small margin) so outliers don't flatten contours
    x_lo, x_hi = np.percentile(x, [1, 99])
    y_lo, y_hi = np.percentile(y, [1, 99])
    pad_x = 0.12 * max(x_hi - x_lo, 1e-4)
    pad_y = 0.12 * max(y_hi - y_lo, 1e-4)
    gx = np.linspace(x_lo - pad_x, x_hi + pad_x, 80)
    gy = np.linspace(y_lo - pad_y, y_hi + pad_y, 80)
    xx, yy = np.meshgrid(gx, gy)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    z_max = float(zz.max())
    if z_max <= 0 or not np.isfinite(z_max):
        return
    # Iso-density levels spanning 20% to 90% of peak density
    lvl_vals = np.linspace(0.20 * z_max, 0.90 * z_max, levels)
    ax.contourf(xx, yy, zz, levels=np.append(lvl_vals, z_max * 1.01), colors=[color], alpha=fill_alpha)
    ax.contour(xx, yy, zz, levels=lvl_vals, colors=[color], linewidths=linewidths, alpha=line_alpha)
    if label is not None:
        ax.plot([], [], color=color, lw=2.0, label=label)


def _resolve_user_bt_checkpoint(phase3_dir: Path, rank: int | None) -> Path:
    """Resolve UserBT checkpoint: explicit rank -> user_bt_best.pt -> highest available rank."""
    if rank is not None:
        p = phase3_dir / f"user_bt_rank{rank}_best.pt"
        if p.exists():
            return p
        raise FileNotFoundError(f"Requested checkpoint not found: {p}")
    best = phase3_dir / "user_bt_best.pt"
    if best.exists():
        return best
    candidates = sorted(
        phase3_dir.glob("user_bt_rank*_best.pt"),
        key=lambda path: int(path.stem.split("rank")[1].split("_")[0]),
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No user_bt_*_best.pt checkpoints found in {phase3_dir}")


def compute_prompt_polarization_and_distortion(
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
    u_norm: np.ndarray,
    phi_all: np.ndarray,
    r_myth: np.ndarray | None = None,
    phase4a_df: pd.DataFrame | None = None,
    target_beta: float | None = None,
) -> pd.DataFrame:
    """Compute per-prompt distortion and user-polarization diagnostics across all eval prompts.

    If `phase4a_df` is supplied, uses its per-query distortion values; otherwise computes exact
    real-data KL-constrained distortion under uniform pi_ref using `r_myth`.
    """
    avg_u = u_norm.mean(axis=0)
    std_u = u_norm.std(axis=0)
    num_users = u_norm.shape[0]

    dist_map: dict[int, float] = {}
    if phase4a_df is not None:
        sub = phase4a_df[np.isfinite(phase4a_df["distortion"])].copy()
        if "is_resampled" in sub.columns and sub["is_resampled"].any():
            sub = sub[sub["is_resampled"]]
        if "tau" in sub.columns and (sub["tau"] > 1e-6).any():
            sub = sub[sub["tau"] > 1e-6]
        if target_beta is not None and (sub["beta_bt"] == target_beta).any():
            sub = sub[sub["beta_bt"] == target_beta]
        dist_map = sub.groupby("prompt_idx")["distortion"].max().to_dict()

    lams = np.geomspace(0.05, 5.0, 20)
    rows = []
    for p_idx in eval_prompt_indices:
        idxs = ds.prompt_global_xy_indices[p_idx]
        au = avg_u[idxs]
        if p_idx in dist_map:
            q_dist = float(dist_map[p_idx])
        elif r_myth is not None:
            rm = r_myth[idxs]
            pi_ref = np.full(len(idxs), 1.0 / len(idxs), dtype=np.float64)
            q_dist = 1.0
            for lam in lams:
                pi_lam = gibbs_policy(rm, pi_ref, float(lam))
                res = compute_distortion(au, pi_lam, pi_ref)
                if np.isfinite(res["distortion"]) and res["distortion"] > q_dist:
                    q_dist = float(res["distortion"])
        else:
            continue

        top1 = np.argmax(u_norm[:, idxs], axis=1)
        _, cnts = np.unique(top1, return_counts=True)
        probs = cnts / float(num_users)
        maj = float(probs.max())
        norm_ent = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(max(len(idxs), 2)))
        ph = phi_all[idxs]
        phi_spread = float(np.linalg.norm(ph - ph.mean(axis=0), axis=1).mean())
        corr_rm_au = np.nan
        if r_myth is not None:
            rm = r_myth[idxs]
            if np.std(rm) > 1e-8 and np.std(au) > 1e-8:
                corr_rm_au = float(np.corrcoef(rm, au)[0, 1])

        rows.append(
            {
                "prompt_idx": int(p_idx),
                "query_distortion": q_dist,
                "user_std": float(std_u[idxs].mean()),
                "top1_majority": maj,
                "user_fragmentation": 1.0 - maj,
                "top1_norm_entropy": norm_ent,
                "phi_spread": phi_spread,
                "corr_myth_avgutil": corr_rm_au,
                "num_candidates": len(idxs),
            }
        )
    return pd.DataFrame(rows)


def plot_qualitative_pca(
    points_df: pd.DataFrame,
    explained_var: np.ndarray,
    save_path: Path,
    title: str,
    centered_ev: np.ndarray | None = None,
    prompt_stats_df: pd.DataFrame | None = None,
) -> Path:
    """Multi-panel 2D PCA & polarization contour visualization in trained user-preference space.

    When centered PCA (`pc1_centered`, `pc2_centered`), user-space consensus/polarization axes
    (`consensus_axis`, `polarization_axis`), and `prompt_stats_df` are present, renders a 2x2
    contour + polarization figure:
      (A) Raw 2D PCA of phi(x, y) colored by query distortion + High/Low KDE contours (+ DPO gens)
      (B) Within-prompt centered 2D PCA of dphi(x, y) = phi(x, y) - mean_y phi(x, y) + KDE contours
      (C) User-Space Consensus vs. Top User-Disagreement (Polarization) Axes + KDE contours
      (D) Prompt-Level User Fragmentation / Polarization vs. Query Distortion (with Spearman rho)

    Otherwise renders the compact 1x2 PCA + centroid-shift figure (used by lightweight unit tests).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    has_rich = (
        prompt_stats_df is not None
        and len(prompt_stats_df) >= 10
        and {"pc1_centered", "pc2_centered", "consensus_axis", "polarization_axis"}.issubset(points_df.columns)
    )

    ev1 = 100.0 * float(explained_var[0]) if len(explained_var) >= 1 else 0.0
    ev2 = 100.0 * float(explained_var[1]) if len(explained_var) >= 2 else 0.0
    cmap = plt.get_cmap("coolwarm")
    d_min = float(points_df["query_distortion"].min())
    d_max = float(points_df["query_distortion"].max())
    norm = plt.Normalize(vmin=d_min, vmax=max(d_max, d_min + 1e-6))

    ds_sub = points_df[points_df["source"] == "dataset"]
    ref_sub = points_df[points_df["source"] == "ref_lm"]
    dpo_sub = points_df[points_df["source"] == "dpo_lm"]
    ds_hi = ds_sub[ds_sub["group"] == "high_distortion"]
    ds_lo = ds_sub[ds_sub["group"] == "low_distortion"]

    if has_rich:
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.8), dpi=160)
        ax_raw, ax_cent, ax_pol, ax_stat = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)
        ax_raw, ax_cent = axes[0], axes[1]
        ax_pol = ax_stat = None

    # --- Panel A: Raw 2D PCA + KDE Contours ---
    pt_size = 12 if len(ds_sub) > 800 else 28
    pt_alpha = 0.28 if len(ds_sub) > 800 else 0.55
    sc = ax_raw.scatter(
        ds_sub["pc1"],
        ds_sub["pc2"],
        c=ds_sub["query_distortion"],
        cmap=cmap,
        norm=norm,
        s=pt_size,
        alpha=pt_alpha,
        marker="o",
        edgecolors="none",
        label=f"Dataset responses (n={len(ds_sub):,})",
    )
    _add_kde_contours(ax_raw, ds_lo["pc1"], ds_lo["pc2"], color="#1d4ed8", label=f"Low-distortion KDE (n={len(ds_lo):,})")
    _add_kde_contours(ax_raw, ds_hi["pc1"], ds_hi["pc2"], color="#dc2626", label=f"High-distortion KDE (n={len(ds_hi):,})")

    if len(ref_sub) > 0:
        ax_raw.scatter(
            ref_sub["pc1"], ref_sub["pc2"], c=ref_sub["query_distortion"], cmap=cmap, norm=norm,
            s=65, alpha=0.88, marker="^", edgecolors="#1e293b", linewidths=0.7,
            label=f"Base/Ref LM gens (n={len(ref_sub)})",
        )
    if len(dpo_sub) > 0:
        ax_raw.scatter(
            dpo_sub["pc1"], dpo_sub["pc2"], c=dpo_sub["query_distortion"], cmap=cmap, norm=norm,
            s=115, alpha=0.95, marker="*", edgecolors="black", linewidths=0.7,
            label=f"DPO LM gens (n={len(dpo_sub)})",
        )

    cbar = fig.colorbar(sc, ax=ax_raw, pad=0.02)
    cbar.set_label("Query Distortion", fontsize=9.5)
    ax_raw.set_xlabel(f"PC1 ({ev1:.1f}% variance)", fontsize=10)
    ax_raw.set_ylabel(f"PC2 ({ev2:.1f}% variance)", fontsize=10)
    ax_raw.set_title("(A) User-Space PCA $\\phi(x, y)$ + High/Low Distortion Contours", fontweight="bold")
    ax_raw.legend(fontsize=8.2, frameon=True, loc="best")

    if has_rich:
        # --- Panel B: Within-Prompt Centered PCA dphi(x, y) + KDE Contours ---
        cev1 = 100.0 * float(centered_ev[0]) if centered_ev is not None and len(centered_ev) >= 1 else 0.0
        cev2 = 100.0 * float(centered_ev[1]) if centered_ev is not None and len(centered_ev) >= 2 else 0.0
        ax_cent.scatter(ds_lo["pc1_centered"], ds_lo["pc2_centered"], color="#2563eb", s=pt_size, alpha=0.18, edgecolors="none")
        ax_cent.scatter(ds_hi["pc1_centered"], ds_hi["pc2_centered"], color="#dc2626", s=pt_size, alpha=0.22, edgecolors="none")
        _add_kde_contours(ax_cent, ds_lo["pc1_centered"], ds_lo["pc2_centered"], color="#1d4ed8", label="Low-distortion within-prompt KDE")
        _add_kde_contours(ax_cent, ds_hi["pc1_centered"], ds_hi["pc2_centered"], color="#dc2626", label="High-distortion within-prompt KDE")
        if len(dpo_sub) > 0:
            ax_cent.scatter(
                dpo_sub["pc1_centered"], dpo_sub["pc2_centered"], c=dpo_sub["query_distortion"], cmap=cmap, norm=norm,
                s=105, alpha=0.95, marker="*", edgecolors="black", linewidths=0.7, label=f"DPO gens (n={len(dpo_sub)})",
            )
        ax_cent.axvline(0.0, color="#64748b", ls=":", lw=1.0)
        ax_cent.axhline(0.0, color="#64748b", ls=":", lw=1.0)
        # Zoom to 1st-99th percentile so contour structure is crisp
        ax_cent.set_xlim(np.percentile(ds_sub["pc1_centered"], [1, 99]) * 1.15)
        ax_cent.set_ylim(np.percentile(ds_sub["pc2_centered"], [1, 99]) * 1.15)
        ax_cent.set_xlabel(f"Centered PC1 ({cev1:.1f}% variance)", fontsize=10)
        ax_cent.set_ylabel(f"Centered PC2 ({cev2:.1f}% variance)", fontsize=10)
        ax_cent.set_title("(B) Within-Prompt Response Spread $\\Delta\\phi(x, y) = \\phi(x, y) - \\bar{\\phi}(x)$", fontweight="bold")
        ax_cent.legend(fontsize=8.2, frameon=True, loc="best")

        # --- Panel C: 1D Histogram + KDE along Consensus Direction (Low vs. High Distortion) ---
        lo_vals = ds_lo["consensus_axis"].dropna().to_numpy(dtype=np.float64)
        hi_vals = ds_hi["consensus_axis"].dropna().to_numpy(dtype=np.float64)
        all_cons = np.concatenate([lo_vals, hi_vals])
        x_lo, x_hi = np.percentile(all_cons, [0.5, 99.5])
        bins_1d = np.linspace(x_lo, x_hi, 42)
        grid_1d = np.linspace(x_lo, x_hi, 300)

        ax_pol.hist(
            lo_vals, bins=bins_1d, density=True, color="#2563eb", alpha=0.28,
            edgecolor="#1d4ed8", linewidth=0.6, label=f"Low-distortion hist (n={len(lo_vals):,})",
        )
        ax_pol.hist(
            hi_vals, bins=bins_1d, density=True, color="#dc2626", alpha=0.28,
            edgecolor="#b91c1c", linewidth=0.6, label=f"High-distortion hist (n={len(hi_vals):,})",
        )
        if lo_vals.size >= 10 and np.std(lo_vals) > 1e-8:
            kde_lo = gaussian_kde(lo_vals)
            ax_pol.plot(grid_1d, kde_lo(grid_1d), color="#1d4ed8", lw=2.3, label="Low-distortion KDE")
        if hi_vals.size >= 10 and np.std(hi_vals) > 1e-8:
            kde_hi = gaussian_kde(hi_vals)
            ax_pol.plot(grid_1d, kde_hi(grid_1d), color="#dc2626", lw=2.3, label="High-distortion KDE")

        if len(dpo_sub) > 0:
            dpo_vals = dpo_sub["consensus_axis"].dropna().to_numpy(dtype=np.float64)
            if dpo_vals.size >= 5 and np.std(dpo_vals) > 1e-8:
                kde_dpo = gaussian_kde(dpo_vals)
                ax_pol.plot(grid_1d, kde_dpo(grid_1d), color="#0f172a", ls="--", lw=2.0, label=f"DPO gens KDE (n={len(dpo_vals)})")
            elif dpo_vals.size > 0:
                ax_pol.axvline(float(np.mean(dpo_vals)), color="#0f172a", ls="--", lw=1.8, label=f"DPO gens mean (n={len(dpo_vals)})")

        ax_pol.axvline(0.0, color="#64748b", ls=":", lw=1.0)
        ax_pol.set_xlim(x_lo, x_hi)
        ax_pol.set_xlabel("Consensus Direction $\\langle \\hat{\\bar{w}}, \\Delta\\phi(x, y) \\rangle$ (Utilitarian Agreement)", fontsize=9.8)
        ax_pol.set_ylabel("Density", fontsize=10)
        ax_pol.set_title("(C) Consensus Score Distribution: Low vs. High Distortion", fontweight="bold")
        ax_pol.legend(fontsize=8.2, frameon=True, loc="best")

        # --- Panel D: Prompt-Level Polarization vs. Distortion across all eval prompts ---
        ps = prompt_stats_df.dropna(subset=["user_fragmentation", "query_distortion"])
        sc_p = ax_stat.scatter(
            ps["user_fragmentation"],
            ps["query_distortion"],
            c=ps["user_std"],
            cmap="magma",
            s=30,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
        )
        cbar_p = fig.colorbar(sc_p, ax=ax_stat, pad=0.02)
        cbar_p.set_label("Cross-User Utility Std", fontsize=9.5)

        # Binned median trend curve across fragmentation deciles
        bins = np.quantile(ps["user_fragmentation"], np.linspace(0.0, 1.0, 9))
        bins = np.unique(bins)
        if len(bins) >= 3:
            bx, by, b90 = [], [], []
            for lo_b, hi_b in zip(bins[:-1], bins[1:]):
                m = (ps["user_fragmentation"] >= lo_b) & (ps["user_fragmentation"] <= hi_b)
                if m.sum() >= 3:
                    bx.append(float(ps.loc[m, "user_fragmentation"].median()))
                    by.append(float(ps.loc[m, "query_distortion"].median()))
                    b90.append(float(ps.loc[m, "query_distortion"].quantile(0.90)))
            ax_stat.plot(bx, by, "-o", color="#0f172a", lw=2.2, ms=5, label="Binned median distortion")
            ax_stat.plot(bx, b90, "--s", color="#dc2626", lw=1.7, ms=4, label="Binned 90th-pct distortion")

        rho_frag, p_frag = spearmanr(ps["user_fragmentation"], ps["query_distortion"])
        rho_std, p_std = spearmanr(ps["user_std"], ps["query_distortion"])
        ax_stat.text(
            0.03,
            0.96,
            f"All {len(ps)} eval prompts:\n"
            f"Spearman(Frag, Dist) = {rho_frag:+.3f} (p={p_frag:.1e})\n"
            f"Spearman(UserStd, Dist) = {rho_std:+.3f} (p={p_std:.1e})",
            transform=ax_stat.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
        )
        ax_stat.set_xlabel("User Polarization / Fragmentation ($1 - \\max_y \\mathrm{Top1Share}(y|x)$)", fontsize=9.8)
        ax_stat.set_ylabel("Query Distortion (max over $\\lambda$)", fontsize=10)
        ax_stat.set_title("(D) Higher User Polarization Predicts Higher Distortion", fontweight="bold")
        ax_stat.legend(fontsize=8.2, frameon=True, loc="lower right")
    else:
        # Compact Panel 2 for lightweight / DPO-only mode: High- vs. Low-distortion centroid shift
        group_styles = {
            "high_distortion": ("#dc2626", "High-distortion queries"),
            "low_distortion": ("#2563eb", "Low-distortion queries"),
        }
        for grp_key, (col, grp_label) in group_styles.items():
            g_ds = ds_sub[ds_sub["group"] == grp_key]
            g_dpo = dpo_sub[dpo_sub["group"] == grp_key]
            ax_cent.scatter(g_ds["pc1"], g_ds["pc2"], color=col, alpha=0.22, s=26, marker="o", edgecolors="none")
            _add_kde_contours(ax_cent, g_ds["pc1"], g_ds["pc2"], color=col)
            if len(g_ds) > 0:
                c_ds = (float(g_ds["pc1"].mean()), float(g_ds["pc2"].mean()))
                ax_cent.scatter([c_ds[0]], [c_ds[1]], color=col, s=130, marker="o", edgecolors="black", linewidths=1.2, label=f"{grp_label}: dataset mean")
            if len(g_dpo) > 0:
                ax_cent.scatter(g_dpo["pc1"], g_dpo["pc2"], color=col, alpha=0.85, s=110, marker="*", edgecolors="black", linewidths=0.7)
                c_dpo = (float(g_dpo["pc1"].mean()), float(g_dpo["pc2"].mean()))
                ax_cent.scatter([c_dpo[0]], [c_dpo[1]], color=col, s=220, marker="*", edgecolors="black", linewidths=1.3, label=f"{grp_label}: DPO mean")
                if len(g_ds) > 0:
                    ax_cent.annotate(
                        "",
                        xy=c_dpo,
                        xytext=c_ds,
                        arrowprops={"arrowstyle": "->", "color": col, "lw": 2.2, "shrinkA": 6, "shrinkB": 8},
                    )
        ax_cent.set_xlabel(f"PC1 ({ev1:.1f}% variance)", fontsize=10)
        ax_cent.set_ylabel(f"PC2 ({ev2:.1f}% variance)", fontsize=10)
        ax_cent.set_title("Distributional Shift in User Space: Dataset vs. DPO Generations", fontweight="bold")
        ax_cent.legend(fontsize=8.0, frameon=True, loc="best")

    fig.suptitle(title, fontsize=13.0, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative 2D PCA & polarization contour evaluation in user space phi(x, y).")
    parser.add_argument("--mode", choices=["local-test", "full"], default="local-test")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--phase4a-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpo-model", type=str, default=None, help="Path or HF ID of DPO-trained LM (optional; if omitted in full mode, runs dataset-only contour/polarization analysis).")
    parser.add_argument("--ref-model", type=str, default=None, help="Optional base/ref LM before DPO for comparison.")
    parser.add_argument("--skip-gen", action="store_true", help="Skip LLM generation and plot dataset user-space contours + polarization only.")
    parser.add_argument("--rank", type=int, default=None, help="UserBT rank checkpoint (default: user_bt_best.pt or highest available rank).")
    parser.add_argument("--top-n", type=int, default=None, help="Number of highest & lowest distortion queries each (default: 10 in local-test, 100 in full).")
    parser.add_argument("--eval-min-users", type=int, default=20, help="Min annotators per prompt for full eval set.")
    parser.add_argument("--k", type=int, default=5, help="Number of generations per query from DPO/ref LM.")
    parser.add_argument("--target-beta", type=float, default=None, help="Optional specific beta_BT slice for ranking queries.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-user-prefs-gt", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    phase3_dir = args.phase3_dir or (ROOT / "results" / ("phase3_local_test" if args.mode == "local-test" else "phase3"))
    phase4a_dir = ROOT / "results" / ("phase4a_local_test" if args.mode == "local-test" else "phase4a")
    out_dir = args.out_dir or (ROOT / "results" / ("qualitative_pca_local_test" if args.mode == "local-test" else "qualitative_pca"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "local-test":
        ds = load_turn1_dataset(phase3_dir / "local_fixture_en.csv", min_user_prefs_gt=1)
        eval_prompt_indices = ds.prompts_with_min_users(3)
        phase4a_csv = args.phase4a_csv or (phase4a_dir / "phase4a_distortion_llm_len_norm_per_query.csv")
        dpo_model = None if args.skip_gen else (args.dpo_model or "HuggingFaceTB/SmolLM2-135M")
        top_n = args.top_n if args.top_n is not None else 10
        max_len = min(args.max_length, 256)
        max_new = min(args.max_new_tokens, 64)
        enc_bs = min(args.encode_batch_size, 16)
    else:
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=args.min_user_prefs_gt)
        eval_prompt_indices = ds.prompts_with_min_users(args.eval_min_users)
        phase4a_csv = args.phase4a_csv or (phase4a_dir / "phase4a_distortion_llm_len_norm_per_query.csv")
        dpo_model = None if args.skip_gen else args.dpo_model
        top_n = args.top_n if args.top_n is not None else 100
        max_len = args.max_length
        max_new = args.max_new_tokens
        enc_bs = args.encode_batch_size

    # 1. Load trained Phase 3 UserBT linear head W, cached ModernBERT CLS features, and MythicalUserBT
    feats_all = torch.load(phase3_dir / "modernbert_cls_features.pt", map_location="cpu", weights_only=True)
    ckpt_path = _resolve_user_bt_checkpoint(phase3_dir, args.rank)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    user_bt = LowRankUserBT(
        num_users=int(ckpt["num_users"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        rank=int(ckpt["rank"]),
    )
    user_bt.load_state_dict(ckpt["state_dict"])
    user_bt.eval()

    r_myth: np.ndarray | None = None
    myth_path = phase3_dir / "mythical_bt_best.pt"
    if myth_path.exists():
        m_ckpt = torch.load(myth_path, map_location="cpu", weights_only=True)
        myth_bt = MythicalUserBT(hidden_dim=int(m_ckpt["hidden_dim"]))
        myth_bt.load_state_dict(m_ckpt["state_dict"])
        myth_bt.eval()
        with torch.no_grad():
            r_myth = myth_bt.score(feats_all.float()).numpy().astype(np.float64)

    with torch.no_grad():
        raw_scores = user_bt.score_all_users(feats_all.float()).numpy()
        phi_all = user_bt.project(feats_all.float()).numpy().astype(np.float64)
        user_weights = user_bt.user_emb.weight.detach().cpu().numpy().astype(np.float64)  # [N, d]

    u_norm, _ = per_query_normalize(raw_scores, ds.prompt_global_xy_indices)

    # 2. Compute per-prompt distortion & user polarization across ALL eval prompts
    phase4a_df = pd.read_csv(phase4a_csv) if phase4a_csv.exists() else None
    dist_source = phase4a_csv.name if phase4a_df is not None else f"{ckpt_path.name} + mythical_bt (uniform pi_ref)"
    prompt_stats_df = compute_prompt_polarization_and_distortion(
        ds=ds,
        eval_prompt_indices=eval_prompt_indices,
        u_norm=u_norm,
        phi_all=phi_all,
        r_myth=r_myth,
        phase4a_df=phase4a_df,
        target_beta=args.target_beta,
    )
    prompt_stats_df.to_csv(out_dir / "qualitative_prompt_polarization.csv", index=False)

    # Select disjoint top-N highest & top-N lowest distortion queries
    if phase4a_df is not None:
        selected = select_extreme_distortion_queries(phase4a_df, top_n=top_n, target_beta=args.target_beta)
    else:
        rank_df = prompt_stats_df[["prompt_idx", "query_distortion"]].rename(columns={"query_distortion": "distortion"})
        selected = select_extreme_distortion_queries(rank_df, top_n=top_n)

    n_high = int((selected["group"] == "high_distortion").sum())
    n_low = int((selected["group"] == "low_distortion").sum())
    print(f"=== Qualitative PCA & Polarization ({args.mode}, ckpt={ckpt_path.name}) on device={device} ===")
    print(f"Eval prompts={len(prompt_stats_df)}, Selected {n_high} high + {n_low} low distortion queries from {dist_source}")

    # User-space consensus unit vector and top disagreement (polarization) unit vector
    w_bar = user_weights.mean(axis=0)
    w_bar_hat = w_bar / max(float(np.linalg.norm(w_bar)), 1e-12)
    w_centered = user_weights - w_bar[None, :]
    pol_axis_hat = PCA(n_components=1, random_state=42).fit(w_centered).components_[0]
    pol_axis_hat = pol_axis_hat / max(float(np.linalg.norm(pol_axis_hat)), 1e-12)

    # Collect dataset responses for selected queries (+ per-prompt mean phi(x) for within-prompt centering)
    ds_rows = []
    ds_feat_indices = []
    prompt_mean_phi: dict[int, np.ndarray] = {}
    for _, qrow in selected.iterrows():
        p_idx = int(qrow["prompt_idx"])
        q_dist = float(qrow["query_distortion"])
        grp = str(qrow["group"])
        g_idxs = ds.prompt_global_xy_indices[p_idx]
        prompt_mean_phi[p_idx] = phi_all[g_idxs].mean(axis=0)
        for c_idx, g_idx in enumerate(g_idxs):
            ds_feat_indices.append(g_idx)
            ds_rows.append(
                {
                    "source": "dataset",
                    "prompt_idx": p_idx,
                    "group": grp,
                    "query_distortion": q_dist,
                    "candidate_idx": c_idx,
                    "text": ds.prompt_responses[p_idx][c_idx],
                }
            )

    ds_phi = phi_all[ds_feat_indices]
    ds_dphi = np.vstack([ds_phi[i] - prompt_mean_phi[row["prompt_idx"]] for i, row in enumerate(ds_rows)])

    # 3. Optionally generate k completions from DPO LM (and Ref LM) & encode through ModernBERT + W
    extra_phis: dict[str, np.ndarray] = {}
    extra_dphis: dict[str, np.ndarray] = {}
    extra_rows: dict[str, list[dict]] = {}

    if dpo_model is not None:
        sel_prompts = [ds.prompts[int(p)] for p in selected["prompt_idx"]]
        dpo_gens = generate_k_completions(
            dpo_model, sel_prompts, k=args.k, device=device, max_new_tokens=max_new, temperature=args.temperature, seed=args.seed
        )
        ref_gens = None
        if args.ref_model:
            ref_gens = generate_k_completions(
                args.ref_model, sel_prompts, k=args.k, device=device, max_new_tokens=max_new, temperature=args.temperature, seed=args.seed + 100
            )

        encoder = FrozenModernBERTEncoder(
            model_name="answerdotai/ModernBERT-base",
            max_length=max_len,
            pooling="cls",
            device=device,
        )

        def _encode_and_project(gens_per_q: list[list[str]], source_name: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
            flat_p, flat_r, meta = [], [], []
            for (_, qrow), gen_list in zip(selected.iterrows(), gens_per_q):
                p_idx = int(qrow["prompt_idx"])
                for g_i, text in enumerate(gen_list):
                    flat_p.append(ds.prompts[p_idx])
                    flat_r.append(text)
                    meta.append(
                        {
                            "source": source_name,
                            "prompt_idx": p_idx,
                            "group": str(qrow["group"]),
                            "query_distortion": float(qrow["query_distortion"]),
                            "candidate_idx": g_i,
                            "text": text,
                        }
                    )
            raw_cls = encoder.encode_pairs(flat_p, flat_r, batch_size=enc_bs, output_device="cpu")
            with torch.no_grad():
                phi = user_bt.project(raw_cls.float()).numpy().astype(np.float64)
            dphi = np.vstack([phi[i] - prompt_mean_phi[row["prompt_idx"]] for i, row in enumerate(meta)])
            return phi, dphi, meta

        dpo_phi, dpo_dphi, dpo_rows = _encode_and_project(dpo_gens, "dpo_lm")
        extra_phis["dpo_lm"] = dpo_phi
        extra_dphis["dpo_lm"] = dpo_dphi
        extra_rows["dpo_lm"] = dpo_rows

        if ref_gens is not None:
            ref_phi, ref_dphi, ref_rows = _encode_and_project(ref_gens, "ref_lm")
            extra_phis["ref_lm"] = ref_phi
            extra_dphis["ref_lm"] = ref_dphi
            extra_rows["ref_lm"] = ref_rows

        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Fit raw 2D PCA on dataset phi(x, y) AND centered 2D PCA on dphi(x, y), projecting all sources
    pca, ds_coords, extra_coords = fit_and_project_user_space_pca(ds_phi, extra_phis)
    pca_cent, ds_cent_coords, extra_cent_coords = fit_and_project_user_space_pca(ds_dphi, extra_dphis)

    all_records = []
    for i, row in enumerate(ds_rows):
        all_records.append(
            {
                **row,
                "pc1": float(ds_coords[i, 0]),
                "pc2": float(ds_coords[i, 1]),
                "pc1_centered": float(ds_cent_coords[i, 0]),
                "pc2_centered": float(ds_cent_coords[i, 1]),
                "consensus_axis": float(np.dot(ds_dphi[i], w_bar_hat)),
                "polarization_axis": float(np.dot(ds_dphi[i], pol_axis_hat)),
            }
        )
    for src_name, rows_list in extra_rows.items():
        c_raw = extra_coords[src_name]
        c_cent = extra_cent_coords[src_name]
        dphi_src = extra_dphis[src_name]
        for i, row in enumerate(rows_list):
            all_records.append(
                {
                    **row,
                    "pc1": float(c_raw[i, 0]),
                    "pc2": float(c_raw[i, 1]),
                    "pc1_centered": float(c_cent[i, 0]),
                    "pc2_centered": float(c_cent[i, 1]),
                    "consensus_axis": float(np.dot(dphi_src[i], w_bar_hat)),
                    "polarization_axis": float(np.dot(dphi_src[i], pol_axis_hat)),
                }
            )

    points_df = pd.DataFrame(all_records)
    csv_out = out_dir / "qualitative_pca_embeddings.csv"
    points_df.to_csv(csv_out, index=False)

    # Quantify L2 centroid shift in full user space phi(x, y) if DPO generations exist
    shift_stats = {}
    if "dpo_lm" in extra_phis:
        dpo_phi = extra_phis["dpo_lm"]
        dpo_rows = extra_rows["dpo_lm"]
        for grp_key in ("high_distortion", "low_distortion"):
            ds_mask = np.array([r["group"] == grp_key for r in ds_rows], dtype=bool)
            dpo_mask = np.array([r["group"] == grp_key for r in dpo_rows], dtype=bool)
            if np.any(ds_mask) and np.any(dpo_mask):
                l2_phi_shift = float(np.linalg.norm(dpo_phi[dpo_mask].mean(axis=0) - ds_phi[ds_mask].mean(axis=0)))
                l2_pca_shift = float(np.linalg.norm(extra_coords["dpo_lm"][dpo_mask].mean(axis=0) - ds_coords[ds_mask].mean(axis=0)))
                shift_stats[grp_key] = {
                    "l2_centroid_shift_user_space": round(l2_phi_shift, 4),
                    "l2_centroid_shift_2d_pca": round(l2_pca_shift, 4),
                }

    ev = pca.explained_variance_ratio_
    ev_cent = pca_cent.explained_variance_ratio_
    dpo_tag = f", DPO={Path(dpo_model).name}, k={args.k}" if dpo_model is not None else " (Dataset User-Space Contours)"
    plot_path = plot_qualitative_pca(
        points_df=points_df,
        explained_var=ev,
        save_path=out_dir / "qualitative_pca_user_space.png",
        title=(
            f"User-Space Geometry & Polarization Contours (rank={ckpt['rank']}, N={len(ds_rows):,} responses): "
            f"Top-{n_high} High vs. Top-{n_low} Low Distortion Queries{dpo_tag}"
        ),
        centered_ev=ev_cent,
        prompt_stats_df=prompt_stats_df,
    )

    rho_frag, p_frag = spearmanr(prompt_stats_df["user_fragmentation"], prompt_stats_df["query_distortion"])
    rho_std, p_std = spearmanr(prompt_stats_df["user_std"], prompt_stats_df["query_distortion"])
    summary = {
        "mode": args.mode,
        "checkpoint": ckpt_path.name,
        "distortion_source": dist_source,
        "dpo_model": dpo_model,
        "ref_model": args.ref_model,
        "user_bt_rank": int(ckpt["rank"]),
        "num_eval_prompts": int(len(prompt_stats_df)),
        "n_high_queries": n_high,
        "n_low_queries": n_low,
        "num_dataset_responses_plotted": int(len(ds_rows)),
        "k_generations": args.k if dpo_model is not None else 0,
        "pca_explained_variance_ratio": [round(float(v), 4) for v in ev],
        "centered_pca_explained_variance_ratio": [round(float(v), 4) for v in ev_cent],
        "spearman_fragmentation_vs_distortion": {"rho": round(float(rho_frag), 4), "p": float(p_frag)},
        "spearman_user_std_vs_distortion": {"rho": round(float(rho_std), 4), "p": float(p_std)},
        "centroid_shifts": shift_stats,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "qualitative_pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nRaw PCA explained var     : {[round(float(v), 4) for v in ev]}")
    print(f"Centered PCA explained var: {[round(float(v), 4) for v in ev_cent]}")
    print(f"Spearman(Frag, Dist)      : rho={rho_frag:+.4f} (p={p_frag:.2e})")
    print(f"Spearman(UserStd, Dist)   : rho={rho_std:+.4f} (p={p_std:.2e})")
    if shift_stats:
        print(f"Centroid shifts (DS->DPO) : {json.dumps(shift_stats, indent=2)}")
    print(f"Saved embeddings CSV to   : {csv_out}")
    print(f"Saved 2D PCA plot to      : {plot_path}")


if __name__ == "__main__":
    main()
