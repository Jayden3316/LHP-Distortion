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
from sklearn.decomposition import PCA

from src.data import Turn1Dataset, load_turn1_dataset
from src.encoder import FrozenModernBERTEncoder
from src.ref_policy import load_causal_lm
from src.user_bt import LowRankUserBT


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


def plot_qualitative_pca(
    points_df: pd.DataFrame,
    explained_var: np.ndarray,
    save_path: Path,
    title: str,
) -> Path:
    """Two-panel 2D PCA visualization in trained user-preference space phi(x, y).

    Left panel:  All dataset responses (circles) + DPO generations (stars) + optional Ref LM
                 generations (triangles), colored continuously by query distortion.
    Right panel: Distributional shift comparison — High-distortion vs. Low-distortion query
                 centroids before (dataset / ref LM) vs. after DPO in the same PC1-PC2 space.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)

    ev1 = 100.0 * float(explained_var[0]) if len(explained_var) >= 1 else 0.0
    ev2 = 100.0 * float(explained_var[1]) if len(explained_var) >= 2 else 0.0
    cmap = plt.get_cmap("coolwarm")
    d_min = float(points_df["query_distortion"].min())
    d_max = float(points_df["query_distortion"].max())
    norm = plt.Normalize(vmin=d_min, vmax=max(d_max, d_min + 1e-6))

    # --- Panel 1: All responses + DPO generations colored by query distortion ---
    ax = axes[0]
    ds_sub = points_df[points_df["source"] == "dataset"]
    sc = ax.scatter(
        ds_sub["pc1"],
        ds_sub["pc2"],
        c=ds_sub["query_distortion"],
        cmap=cmap,
        norm=norm,
        s=32,
        alpha=0.55,
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        label=f"Dataset responses (n={len(ds_sub)})",
    )

    ref_sub = points_df[points_df["source"] == "ref_lm"]
    if len(ref_sub) > 0:
        ax.scatter(
            ref_sub["pc1"],
            ref_sub["pc2"],
            c=ref_sub["query_distortion"],
            cmap=cmap,
            norm=norm,
            s=75,
            alpha=0.88,
            marker="^",
            edgecolors="#1e293b",
            linewidths=0.8,
            label=f"Base/Ref LM gens (n={len(ref_sub)})",
        )

    dpo_sub = points_df[points_df["source"] == "dpo_lm"]
    if len(dpo_sub) > 0:
        ax.scatter(
            dpo_sub["pc1"],
            dpo_sub["pc2"],
            c=dpo_sub["query_distortion"],
            cmap=cmap,
            norm=norm,
            s=140,
            alpha=0.95,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label=f"DPO LM gens (n={len(dpo_sub)})",
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Associated Query Distortion", fontsize=9.5)
    ax.set_xlabel(f"PC1 ({ev1:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev2:.1f}% variance)", fontsize=10)
    ax.set_title("User-Space Activations $\\phi(x, y)$ Colored by Query Distortion", fontweight="bold")
    ax.legend(fontsize=8.2, frameon=True, loc="best")

    # --- Panel 2: High- vs. Low-distortion distribution & centroid shift ---
    ax = axes[1]
    group_styles = {
        "high_distortion": ("#dc2626", "High-distortion queries"),
        "low_distortion": ("#2563eb", "Low-distortion queries"),
    }
    for grp_key, (col, grp_label) in group_styles.items():
        g_ds = ds_sub[ds_sub["group"] == grp_key]
        g_dpo = dpo_sub[dpo_sub["group"] == grp_key]
        ax.scatter(g_ds["pc1"], g_ds["pc2"], color=col, alpha=0.22, s=26, marker="o", edgecolors="none")
        if len(g_ds) > 0:
            c_ds = (float(g_ds["pc1"].mean()), float(g_ds["pc2"].mean()))
            ax.scatter([c_ds[0]], [c_ds[1]], color=col, s=130, marker="o", edgecolors="black", linewidths=1.2, label=f"{grp_label}: dataset mean")
        if len(g_dpo) > 0:
            ax.scatter(g_dpo["pc1"], g_dpo["pc2"], color=col, alpha=0.85, s=110, marker="*", edgecolors="black", linewidths=0.7)
            c_dpo = (float(g_dpo["pc1"].mean()), float(g_dpo["pc2"].mean()))
            ax.scatter([c_dpo[0]], [c_dpo[1]], color=col, s=220, marker="*", edgecolors="black", linewidths=1.3, label=f"{grp_label}: DPO mean")
            if len(g_ds) > 0:
                ax.annotate(
                    "",
                    xy=c_dpo,
                    xytext=c_ds,
                    arrowprops={"arrowstyle": "->", "color": col, "lw": 2.2, "shrinkA": 6, "shrinkB": 8},
                )

    ax.set_xlabel(f"PC1 ({ev1:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev2:.1f}% variance)", fontsize=10)
    ax.set_title("Distributional Shift in User Space: Dataset vs. DPO Generations", fontweight="bold")
    ax.legend(fontsize=8.0, frameon=True, loc="best")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative 2D PCA evaluation in user space phi(x, y).")
    parser.add_argument("--mode", choices=["local-test", "full"], default="local-test")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--phase4a-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpo-model", type=str, default=None, help="Path or HF ID of DPO-trained LM.")
    parser.add_argument("--ref-model", type=str, default=None, help="Optional base/ref LM before DPO for comparison.")
    parser.add_argument("--rank", type=int, default=None, help="UserBT rank checkpoint (default: user_bt_best.pt).")
    parser.add_argument("--top-n", type=int, default=10, help="Number of highest & lowest distortion queries each.")
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
        phase4a_csv = args.phase4a_csv or (phase4a_dir / "phase4a_distortion_llm_len_norm_per_query.csv")
        dpo_model = args.dpo_model or "HuggingFaceTB/SmolLM2-135M"
        max_len = min(args.max_length, 256)
        max_new = min(args.max_new_tokens, 64)
        enc_bs = min(args.encode_batch_size, 16)
    else:
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=args.min_user_prefs_gt)
        phase4a_csv = args.phase4a_csv or (phase4a_dir / "phase4a_distortion_llm_len_norm_per_query.csv")
        if args.dpo_model is None:
            raise ValueError("--dpo-model is required in --mode full (pass path or HF ID of DPO-trained LM).")
        dpo_model = args.dpo_model
        max_len = args.max_length
        max_new = args.max_new_tokens
        enc_bs = args.encode_batch_size

    # 1. Select top-N highest & lowest distortion queries
    dist_df = pd.read_csv(phase4a_csv)
    selected = select_extreme_distortion_queries(dist_df, top_n=args.top_n, target_beta=args.target_beta)
    n_high = int((selected["group"] == "high_distortion").sum())
    n_low = int((selected["group"] == "low_distortion").sum())
    print(f"=== Qualitative PCA ({args.mode}) on device={device} ===")
    print(f"Selected {n_high} highest-distortion + {n_low} lowest-distortion queries from {phase4a_csv.name}")
    print(selected.to_string(index=False))

    # 2. Load trained Phase 3 UserBT linear head W and cached ModernBERT CLS features
    feats_all = torch.load(phase3_dir / "modernbert_cls_features.pt", map_location="cpu", weights_only=True)
    ckpt_path = phase3_dir / (f"user_bt_rank{args.rank}_best.pt" if args.rank is not None else "user_bt_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    user_bt = LowRankUserBT(
        num_users=int(ckpt["num_users"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        rank=int(ckpt["rank"]),
    )
    user_bt.load_state_dict(ckpt["state_dict"])
    user_bt.eval()

    # Collect dataset responses for selected queries
    ds_rows = []
    ds_feat_indices = []
    for _, qrow in selected.iterrows():
        p_idx = int(qrow["prompt_idx"])
        q_dist = float(qrow["query_distortion"])
        grp = str(qrow["group"])
        for c_idx, g_idx in enumerate(ds.prompt_global_xy_indices[p_idx]):
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

    with torch.no_grad():
        ds_phi = user_bt.project(feats_all[ds_feat_indices].float()).numpy().astype(np.float64)

    # 3. Generate k completions from DPO LM (and optional Ref LM) & encode through ModernBERT + W
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

    def _encode_and_project(gens_per_q: list[list[str]], source_name: str) -> tuple[np.ndarray, list[dict]]:
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
        return phi, meta

    dpo_phi, dpo_rows = _encode_and_project(dpo_gens, "dpo_lm")
    extra_phis = {"dpo_lm": dpo_phi}
    extra_rows = {"dpo_lm": dpo_rows}

    if ref_gens is not None:
        ref_phi, ref_rows = _encode_and_project(ref_gens, "ref_lm")
        extra_phis["ref_lm"] = ref_phi
        extra_rows["ref_lm"] = ref_rows

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Fit 2D PCA on dataset phi(x, y) and project DPO (and optional ref) generations onto same PCs
    pca, ds_coords, extra_coords = fit_and_project_user_space_pca(ds_phi, extra_phis)

    all_records = []
    for row, (pc1, pc2) in zip(ds_rows, ds_coords):
        all_records.append({**row, "pc1": float(pc1), "pc2": float(pc2)})
    for src_name, rows_list in extra_rows.items():
        for row, (pc1, pc2) in zip(rows_list, extra_coords[src_name]):
            all_records.append({**row, "pc1": float(pc1), "pc2": float(pc2)})

    points_df = pd.DataFrame(all_records)
    csv_out = out_dir / "qualitative_pca_embeddings.csv"
    points_df.to_csv(csv_out, index=False)

    # Quantify L2 centroid shift in full user space phi(x, y) for high vs. low distortion groups
    shift_stats = {}
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
    plot_path = plot_qualitative_pca(
        points_df=points_df,
        explained_var=ev,
        save_path=out_dir / "qualitative_pca_user_space.png",
        title=(
            f"2D PCA in Trained User Space (rank={ckpt['rank']}): "
            f"Top-{n_high} High vs. Top-{n_low} Low Distortion Queries (DPO={Path(dpo_model).name}, k={args.k})"
        ),
    )

    summary = {
        "mode": args.mode,
        "dpo_model": dpo_model,
        "ref_model": args.ref_model,
        "user_bt_rank": int(ckpt["rank"]),
        "n_high_queries": n_high,
        "n_low_queries": n_low,
        "k_generations": args.k,
        "pca_explained_variance_ratio": [round(float(v), 4) for v in ev],
        "centroid_shifts": shift_stats,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "qualitative_pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nPCA explained variance ratio: {[round(float(v), 4) for v in ev]}")
    print(f"Centroid shifts (Dataset -> DPO): {json.dumps(shift_stats, indent=2)}")
    print(f"Saved embeddings CSV to: {csv_out}")
    print(f"Saved 2D PCA plot to   : {plot_path}")


if __name__ == "__main__":
    main()
