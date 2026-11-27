"""Phase 4B: Real DPO Finetuning (TRL + LoRA) & Empirical KL-Constrained Distortion.

Trains actual language models via `trl.DPOTrainer` on either:
  1. Real observed human preferences from `community_alignment_en.csv`, AND/OR
  2. Resampled heterogeneous preferences from Phase 3 user utilities u_tilde at
     controlled Bradley-Terry inverse temperatures beta_BT (Theorem 6 scaling).

Sweeps DPO KL-regularization strength (`--dpo-betas`) and random seeds (`--seeds`) to
trace empirical DPO distortion vs. realized KL radius tau(x) = D_KL(pi_DPO || pi_ref),
comparing directly against Phase 4A closed-form Gibbs policies and Theorem 6.

Usage:
  # Local CPU dry-run (SmolLM2-135M, tiny fixture, LoRA r=8, 4 steps):
  python run_phase4b_dpo.py --mode local-test --save-adapters

  # Cloud GPU real-data DPO sweep (Qwen/Qwen3.5-0.8B, 3 seeds, 4 dpo_betas):
  python run_phase4b_dpo.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B \
      --only-observed --dpo-betas 0.05 0.1 0.25 0.5 --seeds 42 123 456 --save-adapters

  # Cloud GPU full Theorem 6 beta_BT sweep:
  python run_phase4b_dpo.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B \
      --betas 1.0 2.0 4.0 8.0 --dpo-betas 0.05 0.1 0.25 0.5 --seeds 42 123
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_phase4a_distortion import compute_or_load_ref_logprobs
from src.data import Turn1Dataset, load_turn1_dataset
from src.distortion import theorem6_distortion_curve
from src.dpo import (
    build_dpo_preference_dataset,
    evaluate_dpo_policy_distortion,
    score_policy_on_eval_prompts,
    train_dpo_model,
)
from src.mythical_bt import resample_preferences
from src.user_bt import (
    LowRankUserBT,
    global_affine_normalize,
    per_query_normalize,
    unit_range_normalize_per_query,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


def _resolve_user_bt_checkpoint(phase3_dir: Path, rank: int | None) -> Path:
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


def plot_dpo_distortion_vs_tau(
    df: pd.DataFrame,
    save_path: Path,
    title: str,
    phase4a_df: pd.DataFrame | None = None,
) -> Path:
    """Faceted Distortion vs. realized KL radius tau(x) for Phase 4B DPO runs.

    One panel per beta_BT panel (observed + resampled), colored by DPO KL penalty beta_DPO,
    with Theorem 6 curve and optional Phase 4A closed-form Gibbs median overlay.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    panels = list(df["panel_label"].drop_duplicates())
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), dpi=160, squeeze=False)
    axes_flat = axes.ravel()

    dpo_betas = sorted(df["dpo_beta"].unique())
    cmap = plt.get_cmap("viridis")
    beta_colors = {
        b: cmap(i / max(len(dpo_betas) - 1, 1)) for i, b in enumerate(dpo_betas)
    }

    for idx, panel_lbl in enumerate(panels):
        ax = axes_flat[idx]
        sub = df[(df["panel_label"] == panel_lbl) & np.isfinite(df["distortion"]) & np.isfinite(df["tau"])]
        if len(sub) == 0:
            ax.set_title(panel_lbl)
            continue

        beta_bt_val = float(sub["beta_bt"].iloc[0])
        is_resamp = bool(sub["is_resampled"].iloc[0])

        for db in dpo_betas:
            s_db = sub[sub["dpo_beta"] == db]
            if len(s_db) == 0:
                continue
            ax.scatter(
                s_db["tau"],
                s_db["distortion"],
                color=beta_colors[db],
                alpha=0.35 if len(s_db) > 100 else 0.72,
                s=18 if len(s_db) > 100 else 36,
                edgecolors="none",
                label=f"DPO $\\beta_{{\\mathrm{{DPO}}}}={db:g}$",
            )
            # Mark per-(dpo_beta, seed) median point clearly
            for seed_val, s_seed in s_db.groupby("seed"):
                ax.scatter(
                    [float(s_seed["tau"].median())],
                    [float(s_seed["distortion"].median())],
                    color=beta_colors[db],
                    s=80,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=5,
                )

        # Binned median & 90th percentile across all DPO points in this panel
        pos = sub[sub["tau"] > 1e-8]
        if len(pos) >= 6:
            q_edges = np.unique(np.quantile(pos["tau"], np.linspace(0.0, 1.0, min(8, max(3, len(pos) // 10)))))
            if len(q_edges) >= 3:
                bx, bmed, b90 = [], [], []
                for lo, hi in zip(q_edges[:-1], q_edges[1:]):
                    m = (pos["tau"] >= lo) & (pos["tau"] <= hi)
                    if m.sum() >= 2:
                        bx.append(float(pos.loc[m, "tau"].median()))
                        bmed.append(float(pos.loc[m, "distortion"].median()))
                        b90.append(float(pos.loc[m, "distortion"].quantile(0.90)))
                if len(bx) >= 2:
                    ax.plot(bx, bmed, "-o", color="#0f172a", lw=2.0, ms=4.5, label="DPO binned median", zorder=6)
                    ax.plot(bx, b90, "--s", color="#dc2626", lw=1.6, ms=4.0, label="DPO binned 90th pct", zorder=6)

        # Optional Phase 4A closed-form Gibbs comparison overlay
        if phase4a_df is not None and len(phase4a_df) > 0:
            p4a_sub = phase4a_df[
                (phase4a_df["is_resampled"] == is_resamp)
                & (np.isclose(phase4a_df["beta_bt"], beta_bt_val, rtol=1e-2) if is_resamp else True)
                & np.isfinite(phase4a_df["distortion"])
                & (phase4a_df["tau"] > 1e-8)
            ]
            if len(p4a_sub) >= 6:
                q4a = np.unique(np.quantile(p4a_sub["tau"], np.linspace(0.0, 1.0, 8)))
                if len(q4a) >= 3:
                    x4a, y4a = [], []
                    for lo, hi in zip(q4a[:-1], q4a[1:]):
                        m = (p4a_sub["tau"] >= lo) & (p4a_sub["tau"] <= hi)
                        if m.sum() >= 2:
                            x4a.append(float(p4a_sub.loc[m, "tau"].median()))
                            y4a.append(float(p4a_sub.loc[m, "distortion"].median()))
                    if len(x4a) >= 2:
                        ax.plot(x4a, y4a, ":^", color="#7c3aed", lw=1.9, ms=4.5, label="Phase 4A Gibbs median", zorder=5)

        # Theorem 6 curve for resampled panels (where beta_BT is controlled cleanly)
        tau_max_plot = max(float(sub["tau"].quantile(0.98)) if len(sub) > 2 else 1.0, 0.75)
        if is_resamp and beta_bt_val <= 12.0:
            tau_grid = np.linspace(0.0, tau_max_plot, 120)
            thm6 = theorem6_distortion_curve(tau_grid, beta_bt_val)
            ax.plot(tau_grid, thm6, color="#ea580c", ls="-.", lw=1.8, label=f"Thm 6 ($\\sinh(\\beta)/\\beta$)", zorder=4)

        ax.axhline(1.0, color="#94a3b8", ls=":", lw=1.0)
        ax.set_xlabel("Realized KL Radius $\\tau(x) = D_{\\mathrm{KL}}(\\pi_{\\mathrm{DPO}} \\| \\pi_{\\mathrm{ref}})$ (nats)", fontsize=9.2)
        ax.set_ylabel("Empirical Distortion", fontsize=9.5)
        ax.set_title(panel_lbl, fontweight="bold", fontsize=10.5)
        if idx == 0 or idx == n_panels - 1:
            ax.legend(fontsize=7.6, frameon=True, loc="best")

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4B: TRL DPO Finetuning & Empirical KL-Distortion.")
    parser.add_argument("--mode", choices=["local-test", "full"], default="local-test")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--phase4a-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--llm", type=str, default=None, help="Base causal LM (default: SmolLM2-135M locally, Qwen/Qwen3.5-0.8B full).")
    parser.add_argument("--rank", type=int, default=None, help="UserBT rank checkpoint (default: user_bt_best.pt or highest available).")
    parser.add_argument("--norm-mode", choices=["per-query", "unit-range", "global"], default="per-query")
    parser.add_argument("--min-user-prefs-gt", type=int, default=20)
    parser.add_argument("--eval-min-users", type=int, default=20)

    # Preference-generation beta_BT panels
    parser.add_argument("--betas", type=float, nargs="+", default=None, help="Resampled beta_BT panels (e.g. 1.0 2.0 4.0 8.0).")
    parser.add_argument("--only-observed", action="store_true", help="Train DPO ONLY on real observed preferences (skip resampled beta_BT).")
    parser.add_argument("--skip-observed", action="store_true", help="Skip real observed preferences and run only --betas resampled panels.")

    # DPO training hyperparameters (trl.DPOTrainer)
    parser.add_argument("--dpo-betas", type=float, nargs="+", default=None, help="DPO KL penalty values (trl DPOConfig.beta).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds for multi-seed variance estimation.")
    parser.add_argument("--max-train-pairs", type=int, default=None, help="Optional cap on DPO training pairs per run.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epoch training with fixed step count (-1 = full epochs).")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--train-max-length", type=int, default=None)
    parser.add_argument("--ref-batch-size", type=int, default=None)
    parser.add_argument("--ref-max-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and perform full-parameter DPO finetuning.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing during DPO.")
    parser.add_argument("--save-adapters", action="store_true", help="Save trained LoRA/DPO checkpoints for qualitative PCA eval.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    phase3_dir = args.phase3_dir or (ROOT / "results" / ("phase3_local_test" if args.mode == "local-test" else "phase3"))
    phase4a_dir = args.phase4a_dir or (ROOT / "results" / ("phase4a_local_test" if args.mode == "local-test" else "phase4a"))
    out_dir = args.out_dir or (ROOT / "results" / ("phase4b_local_test" if args.mode == "local-test" else "phase4b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mode-specific sensible defaults
    if args.mode == "local-test":
        fixture_csv = phase3_dir / "local_fixture_en.csv"
        ds = load_turn1_dataset(fixture_csv, min_user_prefs_gt=1)
        eval_prompt_indices = ds.prompts_with_min_users(3)
        llm_name = args.llm or "HuggingFaceTB/SmolLM2-135M"
        betas_resample = [] if args.only_observed else (args.betas if args.betas is not None else [2.0])
        dpo_betas = args.dpo_betas if args.dpo_betas is not None else [0.1, 0.5]
        seeds = args.seeds if args.seeds is not None else [42]
        max_train_pairs = args.max_train_pairs if args.max_train_pairs is not None else 32
        max_steps = args.max_steps if args.max_steps > 0 else 4
        train_bs = args.train_batch_size or 2
        grad_accum = args.grad_accum or 2
        train_max_len = args.train_max_length or 256
        ref_bs = args.ref_batch_size or 8
        ref_max_len = args.ref_max_length or 256
        lora_r = args.lora_r or 8
        lora_alpha = args.lora_alpha or 16
        use_grad_ckpt = args.gradient_checkpointing
    else:
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=args.min_user_prefs_gt)
        eval_prompt_indices = ds.prompts_with_min_users(args.eval_min_users)
        llm_name = args.llm or "Qwen/Qwen3.5-0.8B"
        betas_resample = [] if args.only_observed else (args.betas if args.betas is not None else [1.0, 2.0, 4.0, 8.0])
        dpo_betas = args.dpo_betas if args.dpo_betas is not None else [0.05, 0.1, 0.25, 0.5, 1.0]
        seeds = args.seeds if args.seeds is not None else [42, 123]
        max_train_pairs = args.max_train_pairs
        max_steps = args.max_steps
        total_gb = (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )
        is_small_gpu = device.type == "cuda" and total_gb <= 18.0
        train_bs = args.train_batch_size or (1 if is_small_gpu else 4)
        grad_accum = args.grad_accum or (16 if is_small_gpu else 4)
        train_max_len = args.train_max_length or 512
        ref_bs = args.ref_batch_size or (4 if is_small_gpu else 8)
        ref_max_len = args.ref_max_length or 1024
        lora_r = args.lora_r or 16
        lora_alpha = args.lora_alpha or 32
        use_grad_ckpt = args.gradient_checkpointing or is_small_gpu

    # 2. Load Phase 3 user utilities u_norm and compute AvgUtil(y | x)
    feats = torch.load(phase3_dir / "modernbert_cls_features.pt", map_location="cpu", weights_only=True)
    ckpt_path = _resolve_user_bt_checkpoint(phase3_dir, args.rank)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    user_bt = LowRankUserBT(
        num_users=int(ckpt["num_users"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        rank=int(ckpt["rank"]),
    )
    user_bt.load_state_dict(ckpt["state_dict"])
    user_bt.eval()

    with torch.no_grad():
        raw_scores = user_bt.score_all_users(feats.float()).numpy()
    _, beta_nat, _, _ = global_affine_normalize(raw_scores)
    if args.norm_mode == "per-query":
        u_norm, _ = per_query_normalize(raw_scores, ds.prompt_global_xy_indices)
    elif args.norm_mode == "unit-range":
        u_norm = unit_range_normalize_per_query(raw_scores, ds.prompt_global_xy_indices)
    else:
        u_norm, _, _, _ = global_affine_normalize(raw_scores)
    avg_util = u_norm.mean(axis=0)

    print(f"=== Phase 4B DPO ({args.mode}, norm={args.norm_mode}) on device={device} ===")
    print(f"UserBT: {ckpt_path.name} (rank={ckpt['rank']}, beta_BT_nat={beta_nat:.2f}), Eval prompts={len(eval_prompt_indices)}")
    print(f"LLM: {llm_name} | LoRA={not args.no_lora} (r={lora_r}) | dpo_betas={dpo_betas} | seeds={seeds}")

    # 3. Load or compute base reference pi_ref log-probs (reuses Phase 4A cache if available)
    safe_llm = llm_name.replace("/", "__")
    p4a_cache = phase4a_dir / f"pi_ref_logprobs_{safe_llm}.npz"
    ref_cache = p4a_cache if p4a_cache.exists() else (out_dir / f"pi_ref_logprobs_{safe_llm}.npz")
    ref_lps, ref_lens = compute_or_load_ref_logprobs(
        ds=ds,
        eval_prompt_indices=eval_prompt_indices,
        llm_name=llm_name,
        cache_path=ref_cache,
        batch_size=ref_bs,
        max_length=ref_max_len,
        device=device,
    )

    # 4. Build preference panel schedule: (panel_label, beta_bt, is_resampled, pairs_list_fn)
    panel_specs: list[tuple[str, float, bool]] = []
    if not args.skip_observed:
        panel_specs.append((f"Observed (beta_nat={beta_nat:.1f})", float(beta_nat), False))
    for b_bt in betas_resample:
        panel_specs.append((f"Resampled beta_BT={b_bt:g}", float(b_bt), True))

    all_dist_dfs: dict[str, list[pd.DataFrame]] = {"llm": [], "llm_len_norm": []}
    all_run_summaries: list[dict] = []

    for panel_lbl, beta_bt_val, is_resamp in panel_specs:
        for seed in seeds:
            if is_resamp:
                pairs_for_run = resample_preferences(
                    pairs=ds.pairs,
                    u_norm=u_norm,
                    beta_bt=beta_bt_val,
                    seed=seed,
                )
            else:
                pairs_for_run = ds.pairs

            train_ds = build_dpo_preference_dataset(
                ds=ds,
                pairs=pairs_for_run,
                max_pairs=max_train_pairs,
                seed=seed,
            )

            for dpo_beta in dpo_betas:
                tag = f"{'resamp_b' + str(beta_bt_val) if is_resamp else 'obs'}_dpob{dpo_beta}_s{seed}".replace(".", "p")
                run_ckpt_dir = out_dir / "checkpoints" / tag
                print(f"\n--- Training DPO [{panel_lbl}] dpo_beta={dpo_beta} seed={seed} (pairs={len(train_ds)}) ---", flush=True)

                dpo_model, dpo_tok, train_stats = train_dpo_model(
                    model_name=llm_name,
                    train_dataset=train_ds,
                    output_dir=run_ckpt_dir,
                    dpo_beta=dpo_beta,
                    learning_rate=args.lr,
                    num_train_epochs=args.epochs,
                    max_steps=max_steps,
                    per_device_train_batch_size=train_bs,
                    gradient_accumulation_steps=grad_accum,
                    max_length=train_max_len,
                    use_lora=not args.no_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    gradient_checkpointing=use_grad_ckpt,
                    seed=seed,
                    device=device,
                    save_adapter=args.save_adapters,
                    logging_steps=max(1, min(20, max_steps if max_steps > 0 else 20)),
                )

                # Score trained DPO policy on all eval prompt candidates
                dpo_lps, dpo_lens = score_policy_on_eval_prompts(
                    model=dpo_model,
                    tokenizer=dpo_tok,
                    ds=ds,
                    eval_prompt_indices=eval_prompt_indices,
                    batch_size=ref_bs,
                    max_length=ref_max_len,
                    device=device,
                )

                # Free GPU memory before next DPO run
                del dpo_model, dpo_tok
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                run_eval_summary: dict[str, dict] = {}
                for ref_mode in ("llm", "llm_len_norm"):
                    dist_df, _diag_df = evaluate_dpo_policy_distortion(
                        ds=ds,
                        eval_prompt_indices=eval_prompt_indices,
                        avg_util=avg_util,
                        ref_lps=ref_lps,
                        ref_lens=ref_lens,
                        dpo_lps=dpo_lps,
                        dpo_lens=dpo_lens,
                        ref_mode=ref_mode,
                    )
                    dist_df["panel_label"] = panel_lbl
                    dist_df["beta_bt"] = float(beta_bt_val)
                    dist_df["is_resampled"] = bool(is_resamp)
                    dist_df["dpo_beta"] = float(dpo_beta)
                    dist_df["seed"] = int(seed)
                    all_dist_dfs[ref_mode].append(dist_df)

                    fin = dist_df[np.isfinite(dist_df["distortion"])]
                    run_eval_summary[ref_mode] = {
                        "median_tau": round(float(fin["tau"].median()), 5),
                        "p90_tau": round(float(fin["tau"].quantile(0.90)), 5),
                        "median_distortion": round(float(fin["distortion"].median()), 5),
                        "p90_distortion": round(float(fin["distortion"].quantile(0.90)), 5),
                        "max_distortion": round(float(fin["distortion"].max()), 5),
                        "top1_changed_frac": round(float(fin["top1_changed_vs_ref"].mean()), 4),
                    }
                    print(
                        f"    [{ref_mode:12s}] med_tau={run_eval_summary[ref_mode]['median_tau']:.4f}, "
                        f"med_dist={run_eval_summary[ref_mode]['median_distortion']:.4f}, "
                        f"max_dist={run_eval_summary[ref_mode]['max_distortion']:.4f}, "
                        f"top1_shift={100*run_eval_summary[ref_mode]['top1_changed_frac']:.1f}%"
                    )

                all_run_summaries.append(
                    {
                        "tag": tag,
                        "panel_label": panel_lbl,
                        "beta_bt": float(beta_bt_val),
                        "is_resampled": bool(is_resamp),
                        "train": train_stats,
                        "eval": run_eval_summary,
                    }
                )

    # 5. Save CSVs & faceted plots (overlaying Phase 4A closed-form curve if present)
    norm_tag = args.norm_mode.replace("-", "_")
    for ref_mode in ("llm", "llm_len_norm"):
        full_df = pd.concat(all_dist_dfs[ref_mode], ignore_index=True)
        csv_path = out_dir / f"phase4b_dpo_distortion_{ref_mode}_{norm_tag}.csv"
        full_df.to_csv(csv_path, index=False)

        p4a_csv = phase4a_dir / f"phase4a_distortion_{ref_mode}_{norm_tag}.csv"
        p4a_df = pd.read_csv(p4a_csv) if p4a_csv.exists() else None

        plot_path = out_dir / f"phase4b_dpo_distortion_vs_tau_{ref_mode}_{norm_tag}.png"
        plot_dpo_distortion_vs_tau(
            df=full_df,
            save_path=plot_path,
            title=(
                f"Phase 4B DPO Empirical Distortion vs. $\\tau(x)$ "
                f"({Path(llm_name).name}, ref={ref_mode}, norm={args.norm_mode}, rank={ckpt['rank']})"
            ),
            phase4a_df=p4a_df,
        )
        print(f"Saved {ref_mode} CSV  : {csv_path}")
        print(f"Saved {ref_mode} plot : {plot_path}")

    summary = {
        "mode": args.mode,
        "device": str(device),
        "llm": llm_name,
        "checkpoint": ckpt_path.name,
        "user_bt_rank": int(ckpt["rank"]),
        "norm_mode": args.norm_mode,
        "num_eval_prompts": len(eval_prompt_indices),
        "dpo_betas": dpo_betas,
        "seeds": seeds,
        "runs": all_run_summaries,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    summary_path = out_dir / "phase4b_dpo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 4B summary to: {summary_path} ({summary['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
