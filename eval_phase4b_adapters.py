"""Post-hoc Evaluation of Saved Phase 4B DPO LoRA Adapters:
Standard Preference-Fit Metrics + Empirical KL-Constrained Distortion + Cross-Metric Qualitative Cases.

Evaluates one or more saved PEFT/LoRA DPO checkpoints (without retraining!) against:
  1. Standard preference-fit metrics (Section 5.2):
     - Pairwise preference accuracy (DPO implicit reward r_DPO, raw logp, len-norm logp)
     - DPO pairwise cross-entropy (NLL) and mean reward margin
     - Top-1 agreement with empirical dataset winner (highest win-rate) and AvgUtil optimum
     - Per-prompt Spearman rank correlation vs. empirical win-rates and AvgUtil
     - Realized KL tau(x), entropy, top-1 shift vs. pi_ref, and top-1 length shift (tokens)
  2. Empirical KL-constrained distortion under three reference views:
     - `llm`: raw completion log-probs sum_t log p(y_t | x, y_{<t})
     - `llm_len_norm`: length-normalized completion log-probs (1/T) sum_t log p(...)
     - `dpo_implicit_uniform`: uniform pi_ref tilted by exact log-ratio (l_DPO - l_ref),
       canceling base-model length bias without dividing away the DPO shift by T
  3. Cross-metric qualitative report (Section 5.3):
     - Top-K highest-distortion DPO prompts with full response text & stats
     - Prompts flagged by high distortion despite strong pairwise accuracy (and vice versa)

Usage:
  # Local CPU smoke test on saved local-test adapter:
  python eval_phase4b_adapters.py --mode local-test \
      --adapters results/phase4b_local_test/checkpoints/obs_dpob0p1_s42

  # Full cloud GPU evaluation across all saved adapters in results/phase4b/checkpoints/:
  python eval_phase4b_adapters.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B \
      --checkpoints-dir results/phase4b/checkpoints
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from run_phase4a_distortion import compute_or_load_ref_logprobs
from src.data import Turn1Dataset, load_turn1_dataset
from src.distortion import compute_distortion, solve_kl_ball_max_util
from src.dpo import score_policy_on_eval_prompts
from src.ref_policy import load_causal_lm, logprobs_to_probs, ref_policy_diagnostics
from src.user_bt import (
    LowRankUserBT,
    global_affine_normalize,
    per_query_normalize,
    unit_range_normalize_per_query,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


def _infer_dpo_beta_from_name(adapter_path: Path, default_beta: float = 0.1) -> float:
    m = re.search(r"dpob([0-9]+p[0-9]+|[0-9]+)", adapter_path.name)
    if not m:
        return default_beta
    return float(m.group(1).replace("p", "."))


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


def compute_empirical_candidate_winrates(
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, list[tuple[int, int]]]]:
    """Compute per-candidate empirical pairwise win-rates and local index pairs `(win_c, lose_c)`."""
    eval_set = set(eval_prompt_indices)
    global_to_local: dict[int, dict[int, int]] = {
        p_idx: {g_idx: c_idx for c_idx, g_idx in enumerate(ds.prompt_global_xy_indices[p_idx])}
        for p_idx in eval_prompt_indices
    }
    wins: dict[int, np.ndarray] = {p: np.zeros(len(ds.prompt_global_xy_indices[p]), dtype=np.float64) for p in eval_prompt_indices}
    totals: dict[int, np.ndarray] = {p: np.zeros(len(ds.prompt_global_xy_indices[p]), dtype=np.float64) for p in eval_prompt_indices}
    local_pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for _u, p_idx, win_xy, lose_xy in ds.pairs:
        if p_idx not in eval_set:
            continue
        c_w = global_to_local[p_idx][win_xy]
        c_l = global_to_local[p_idx][lose_xy]
        wins[p_idx][c_w] += 1.0
        totals[p_idx][c_w] += 1.0
        totals[p_idx][c_l] += 1.0
        local_pairs[p_idx].append((c_w, c_l))

    winrates: dict[int, np.ndarray] = {}
    for p_idx in eval_prompt_indices:
        # Laplace-smoothed win-rate so unseen/rare candidates stay well-behaved
        winrates[p_idx] = (wins[p_idx] + 0.5) / (totals[p_idx] + 1.0)
    return winrates, local_pairs


def evaluate_single_adapter(
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
    u_norm: np.ndarray,
    ref_lps: dict[int, np.ndarray],
    ref_lens: dict[int, np.ndarray],
    dpo_lps: dict[int, np.ndarray],
    dpo_lens: dict[int, np.ndarray],
    dpo_beta: float,
    emp_winrates: dict[int, np.ndarray],
    local_pairs: dict[int, list[tuple[int, int]]],
    adapter_tag: str,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-prompt preference-fit & distortion metrics across all 3 reference views."""
    avg_util = u_norm.mean(axis=0)
    all_rows: list[dict] = []

    # Pair-level DPO fit aggregates across all eval-prompt comparisons
    all_r_dpo_margins: list[float] = []
    all_raw_lp_acc: list[float] = []
    all_len_norm_acc: list[float] = []

    prompt_pair_stats: dict[int, dict[str, float]] = {}
    for p_idx in eval_prompt_indices:
        r_lp = np.asarray(ref_lps[p_idx], dtype=np.float64)
        d_lp = np.asarray(dpo_lps[p_idx], dtype=np.float64)
        d_len = np.maximum(np.asarray(dpo_lens[p_idx], dtype=np.float64), 1.0)
        r_dpo = dpo_beta * (d_lp - r_lp)
        d_lp_norm = d_lp / d_len

        pairs_p = local_pairs.get(p_idx, [])
        if pairs_p:
            cw = np.array([x[0] for x in pairs_p], dtype=np.int64)
            cl = np.array([x[1] for x in pairs_p], dtype=np.int64)
            m_imp = r_dpo[cw] - r_dpo[cl]
            nll_imp = float(np.mean(np.logaddexp(0.0, -m_imp)))
            acc_imp = float(np.mean(m_imp > 0))
            acc_raw = float(np.mean((d_lp[cw] - d_lp[cl]) > 0))
            acc_ln = float(np.mean((d_lp_norm[cw] - d_lp_norm[cl]) > 0))
            all_r_dpo_margins.extend(m_imp.tolist())
            all_raw_lp_acc.extend(((d_lp[cw] - d_lp[cl]) > 0).astype(np.float64).tolist())
            all_len_norm_acc.extend(((d_lp_norm[cw] - d_lp_norm[cl]) > 0).astype(np.float64).tolist())
            prompt_pair_stats[p_idx] = {
                "pair_acc_implicit_reward": acc_imp,
                "pair_nll_implicit_reward": nll_imp,
                "pair_margin_implicit_reward": float(np.mean(m_imp)),
                "pair_acc_raw_logp": acc_raw,
                "pair_acc_len_norm_logp": acc_ln,
                "num_eval_pairs": len(pairs_p),
            }
        else:
            prompt_pair_stats[p_idx] = {
                "pair_acc_implicit_reward": float("nan"),
                "pair_nll_implicit_reward": float("nan"),
                "pair_margin_implicit_reward": float("nan"),
                "pair_acc_raw_logp": float("nan"),
                "pair_acc_len_norm_logp": float("nan"),
                "num_eval_pairs": 0,
            }

    ref_modes = ("llm", "llm_len_norm", "dpo_implicit_uniform")
    for ref_mode in ref_modes:
        for p_idx in eval_prompt_indices:
            g_idxs = ds.prompt_global_xy_indices[p_idx]
            au = avg_util[g_idxs]
            ew = emp_winrates[p_idx]
            r_lp = np.asarray(ref_lps[p_idx], dtype=np.float64)
            d_lp = np.asarray(dpo_lps[p_idx], dtype=np.float64)
            r_len = np.maximum(np.asarray(ref_lens[p_idx], dtype=np.float64), 1.0)
            d_len = np.maximum(np.asarray(dpo_lens[p_idx], dtype=np.float64), 1.0)

            if ref_mode == "llm":
                pi_ref = logprobs_to_probs(r_lp)
                pi_dpo = logprobs_to_probs(d_lp)
            elif ref_mode == "llm_len_norm":
                pi_ref = logprobs_to_probs(r_lp / r_len)
                pi_dpo = logprobs_to_probs(d_lp / d_len)
            else:
                # Uniform pi_ref tilted by exact DPO log-ratio (cancels base length bias without /T scaling!)
                pi_ref = np.full(len(g_idxs), 1.0 / len(g_idxs), dtype=np.float64)
                pi_dpo = logprobs_to_probs(d_lp - r_lp)

            res = compute_distortion(avg_util=au, pi=pi_dpo, pi_ref=pi_ref)
            diag = ref_policy_diagnostics(pi_dpo)

            top1_ref = int(np.argmax(pi_ref))
            top1_dpo = int(np.argmax(pi_dpo))
            top1_emp = int(np.argmax(ew))
            top1_opt = int(np.argmax(au))

            rho_emp = float(spearmanr(pi_dpo, ew).statistic) if len(g_idxs) >= 3 and np.std(pi_dpo) > 1e-12 and np.std(ew) > 1e-12 else float("nan")
            rho_au = float(spearmanr(pi_dpo, au).statistic) if len(g_idxs) >= 3 and np.std(pi_dpo) > 1e-12 and np.std(au) > 1e-12 else float("nan")

            all_rows.append(
                {
                    "adapter": adapter_tag,
                    "dpo_beta": float(dpo_beta),
                    "ref_mode": ref_mode,
                    "prompt_idx": int(p_idx),
                    "num_candidates": len(g_idxs),
                    "tau": res["tau"],
                    "policy_util": res["policy_util"],
                    "opt_util": res["opt_util"],
                    "ref_util": float(np.dot(pi_ref, au)),
                    "distortion": res["distortion"],
                    "norm_entropy": diag["normalized_entropy"],
                    "is_collapsed": diag["is_collapsed"],
                    "top1_changed_vs_ref": bool(top1_dpo != top1_ref),
                    "top1_matches_emp_best": bool(top1_dpo == top1_emp),
                    "top1_matches_util_opt": bool(top1_dpo == top1_opt),
                    "spearman_vs_emp_winrate": rho_emp,
                    "spearman_vs_avgutil": rho_au,
                    "top1_dpo_len": int(d_len[top1_dpo]),
                    "top1_ref_len": int(r_len[top1_ref]),
                    "top1_emp_len": int(d_len[top1_emp]),
                    **prompt_pair_stats[p_idx],
                }
            )

    df = pd.DataFrame(all_rows)
    m_all = np.asarray(all_r_dpo_margins, dtype=np.float64)
    summary: dict = {
        "adapter": adapter_tag,
        "dpo_beta": float(dpo_beta),
        "num_eval_prompts": len(eval_prompt_indices),
        "num_eval_pairs": int(m_all.size),
        "overall_pair_acc_implicit_reward": round(float(np.mean(m_all > 0)), 4) if m_all.size else float("nan"),
        "overall_pair_nll_implicit_reward": round(float(np.mean(np.logaddexp(0.0, -m_all))), 4) if m_all.size else float("nan"),
        "overall_pair_margin_implicit_reward": round(float(np.mean(m_all)), 4) if m_all.size else float("nan"),
        "overall_pair_acc_raw_logp": round(float(np.mean(all_raw_lp_acc)), 4) if all_raw_lp_acc else float("nan"),
        "overall_pair_acc_len_norm_logp": round(float(np.mean(all_len_norm_acc)), 4) if all_len_norm_acc else float("nan"),
        "by_ref_mode": {},
    }
    for ref_mode in ref_modes:
        sub = df[(df["ref_mode"] == ref_mode) & np.isfinite(df["distortion"])]
        summary["by_ref_mode"][ref_mode] = {
            "median_tau": round(float(sub["tau"].median()), 5),
            "p90_tau": round(float(sub["tau"].quantile(0.90)), 5),
            "median_distortion": round(float(sub["distortion"].median()), 5),
            "p90_distortion": round(float(sub["distortion"].quantile(0.90)), 5),
            "max_distortion": round(float(sub["distortion"].max()), 5),
            "top1_changed_vs_ref_frac": round(float(sub["top1_changed_vs_ref"].mean()), 4),
            "top1_matches_emp_best_frac": round(float(sub["top1_matches_emp_best"].mean()), 4),
            "top1_matches_util_opt_frac": round(float(sub["top1_matches_util_opt"].mean()), 4),
            "mean_spearman_vs_emp": round(float(sub["spearman_vs_emp_winrate"].mean()), 4),
            "mean_spearman_vs_avgutil": round(float(sub["spearman_vs_avgutil"].mean()), 4),
            "mean_top1_dpo_len": round(float(sub["top1_dpo_len"].mean()), 1),
            "mean_top1_ref_len": round(float(sub["top1_ref_len"].mean()), 1),
            "mean_top1_emp_len": round(float(sub["top1_emp_len"].mean()), 1),
        }
    return df, summary


def write_cross_metric_qualitative_report(
    ds: Turn1Dataset,
    u_norm: np.ndarray,
    ref_lps: dict[int, np.ndarray],
    ref_lens: dict[int, np.ndarray],
    dpo_lps: dict[int, np.ndarray],
    dpo_lens: dict[int, np.ndarray],
    emp_winrates: dict[int, np.ndarray],
    df_mode: pd.DataFrame,
    ref_mode: str,
    out_path: Path,
    top_k: int = 5,
) -> str:
    """Write human-readable report of top-K highest-distortion DPO queries + cross-metric mismatches."""
    lines = [
        f"=== Phase 4B DPO Qualitative & Cross-Metric Report (ref_mode={ref_mode}) ===",
        "",
    ]

    def _format_case(header: str, row: pd.Series) -> None:
        p_idx = int(row["prompt_idx"])
        g_idxs = ds.prompt_global_xy_indices[p_idx]
        u_q = u_norm[:, g_idxs]
        au = u_q.mean(axis=0)
        std_u = u_q.std(axis=0)
        ew = emp_winrates[p_idx]
        r_lp = np.asarray(ref_lps[p_idx], dtype=np.float64)
        d_lp = np.asarray(dpo_lps[p_idx], dtype=np.float64)
        r_len = np.maximum(np.asarray(ref_lens[p_idx], dtype=np.float64), 1.0)
        d_len = np.maximum(np.asarray(dpo_lens[p_idx], dtype=np.float64), 1.0)
        if ref_mode == "llm":
            pi_ref, pi_dpo = logprobs_to_probs(r_lp), logprobs_to_probs(d_lp)
        elif ref_mode == "llm_len_norm":
            pi_ref, pi_dpo = logprobs_to_probs(r_lp / r_len), logprobs_to_probs(d_lp / d_len)
        else:
            pi_ref = np.full(len(g_idxs), 1.0 / len(g_idxs))
            pi_dpo = logprobs_to_probs(d_lp - r_lp)

        pi_star, _ = solve_kl_ball_max_util(au, pi_ref, tau=float(row["tau"]))
        lines.append("-" * 96)
        lines.append(
            f"{header} | prompt_idx={p_idx} | dist={row['distortion']:.3f}, tau={row['tau']:.4f} | "
            f"pair_acc={row['pair_acc_implicit_reward']:.3f}, pair_nll={row['pair_nll_implicit_reward']:.3f}, "
            f"spearman_emp={row['spearman_vs_emp_winrate']:+.2f}"
        )
        lines.append(f"Prompt ({len(ds.prompt_annotators[p_idx])} annotators, |A(x)|={len(g_idxs)}):")
        lines.append(f"  {ds.prompts[p_idx].replace(chr(10), ' ').strip()}")

        picks = [
            (int(np.argmax(pi_dpo)), "TOP pi_DPO"),
            (int(np.argmax(pi_star)), "TOP pi* (KL-ball opt)"),
            (int(np.argmax(au)), "MAX AvgUtil"),
            (int(np.argmax(ew)), "TOP Empirical WinRate"),
            (int(np.argmax(pi_ref)), "TOP pi_ref"),
        ]
        seen: dict[int, list[str]] = {}
        ordered: list[int] = []
        for idx_c, tag in picks:
            if idx_c not in seen:
                seen[idx_c] = [tag]
                ordered.append(idx_c)
            else:
                seen[idx_c].append(tag)
        for c_idx in ordered:
            txt = ds.prompt_responses[p_idx][c_idx].replace("\n", " ").strip()
            if len(txt) > 260:
                txt = txt[:260] + "..."
            lines.append(
                f"  * cand#{c_idx:<3} [{', '.join(seen[c_idx])}]\n"
                f"      AvgUtil={au[c_idx]:.3f} (std={std_u[c_idx]:.3f}) | EmpWinRate={ew[c_idx]:.3f} | len={int(d_len[c_idx])} tok\n"
                f"      pi_ref={pi_ref[c_idx]:.4f} -> pi_DPO={pi_dpo[c_idx]:.4f} vs pi*={pi_star[c_idx]:.4f}\n"
                f"      Text: {txt}"
            )

    sub = df_mode[np.isfinite(df_mode["distortion"])].copy()
    top_dist = sub.sort_values("distortion", ascending=False).head(top_k)
    lines.append(f"--- Part 1: Top-{len(top_dist)} Highest Distortion Prompts ---")
    for rank_i, (_, row) in enumerate(top_dist.iterrows(), start=1):
        _format_case(f"[High-Dist #{rank_i}]", row)

    # Cross-metric mismatch A: HIGH distortion despite HIGH pairwise accuracy (>= 0.75)
    high_acc_high_dist = sub[sub["pair_acc_implicit_reward"] >= 0.75].sort_values("distortion", ascending=False).head(3)
    if len(high_acc_high_dist) > 0:
        lines.append("\n--- Part 2: High Distortion DESPITE High Pairwise Accuracy (>= 75%) ---")
        for rank_i, (_, row) in enumerate(high_acc_high_dist.iterrows(), start=1):
            _format_case(f"[Missed-by-Acc #{rank_i}]", row)

    # Cross-metric mismatch B: LOW distortion despite POOR pairwise accuracy (< 0.55)
    low_acc_low_dist = sub[sub["pair_acc_implicit_reward"] < 0.55].sort_values("distortion", ascending=True).head(3)
    if len(low_acc_low_dist) > 0:
        lines.append("\n--- Part 3: Low Distortion DESPITE Poor Pairwise Accuracy (< 55%) ---")
        for rank_i, (_, row) in enumerate(low_acc_low_dist.iterrows(), start=1):
            _format_case(f"[Missed-by-Dist #{rank_i}]", row)

    report = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc evaluation of saved Phase 4B DPO LoRA adapters.")
    parser.add_argument("--mode", choices=["local-test", "full"], default="full")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--phase4a-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--llm", type=str, default=None, help="Base causal LM for pi_ref.")
    parser.add_argument("--checkpoints-dir", type=Path, default=None, help="Directory containing saved LoRA adapter subdirs.")
    parser.add_argument("--adapters", type=Path, nargs="+", default=None, help="Explicit list of saved LoRA adapter dirs.")
    parser.add_argument("--rank", type=int, default=None, help="UserBT rank checkpoint.")
    parser.add_argument("--norm-mode", choices=["per-query", "unit-range", "global"], default="per-query")
    parser.add_argument("--ref-batch-size", type=int, default=None)
    parser.add_argument("--ref-max-length", type=int, default=None)
    parser.add_argument("--top-k-inspect", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    phase3_dir = args.phase3_dir or (ROOT / "results" / ("phase3_local_test" if args.mode == "local-test" else "phase3"))
    phase4a_dir = args.phase4a_dir or (ROOT / "results" / ("phase4a_local_test" if args.mode == "local-test" else "phase4a"))
    out_dir = args.out_dir or (ROOT / "results" / ("phase4b_eval_local_test" if args.mode == "local-test" else "phase4b_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "local-test":
        ds = load_turn1_dataset(phase3_dir / "local_fixture_en.csv", min_user_prefs_gt=1)
        eval_prompt_indices = ds.prompts_with_min_users(3)
        llm_name = args.llm or "HuggingFaceTB/SmolLM2-135M"
        ref_bs = args.ref_batch_size or 8
        ref_max_len = args.ref_max_length or 256
    else:
        ds = load_turn1_dataset(args.csv, min_user_prefs_gt=20)
        eval_prompt_indices = ds.prompts_with_min_users(20)
        llm_name = args.llm or "Qwen/Qwen3.5-0.8B"
        ref_bs = args.ref_batch_size or (16 if device.type == "cuda" else 4)
        ref_max_len = args.ref_max_length or 1024

    # Discover saved adapters
    adapter_dirs: list[Path] = []
    if args.adapters:
        adapter_dirs.extend(args.adapters)
    if args.checkpoints_dir and args.checkpoints_dir.exists():
        for sub in sorted(args.checkpoints_dir.iterdir()):
            if sub.is_dir() and (sub / "adapter_config.json").is_file():
                adapter_dirs.append(sub)
    if not adapter_dirs:
        raise FileNotFoundError("No valid PEFT adapter directories (containing adapter_config.json) found.")

    # Load Phase 3 user utilities
    feats = torch.load(phase3_dir / "modernbert_cls_features.pt", map_location="cpu", weights_only=True)
    ckpt_path = _resolve_user_bt_checkpoint(phase3_dir, args.rank)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    user_bt = LowRankUserBT(int(ckpt["num_users"]), int(ckpt["hidden_dim"]), int(ckpt["rank"]))
    user_bt.load_state_dict(ckpt["state_dict"])
    user_bt.eval()
    with torch.no_grad():
        raw_scores = user_bt.score_all_users(feats.float()).numpy()
    if args.norm_mode == "per-query":
        u_norm, _ = per_query_normalize(raw_scores, ds.prompt_global_xy_indices)
    elif args.norm_mode == "unit-range":
        u_norm = unit_range_normalize_per_query(raw_scores, ds.prompt_global_xy_indices)
    else:
        u_norm, _, _, _ = global_affine_normalize(raw_scores)

    # Reuse cached pi_ref log-probs if available
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

    emp_winrates, local_pairs = compute_empirical_candidate_winrates(ds, eval_prompt_indices)

    all_dfs: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    for adir in adapter_dirs:
        tag = adir.name
        dpo_beta = _infer_dpo_beta_from_name(adir)
        print(f"\n=== Evaluating adapter: {tag} (dpo_beta={dpo_beta}) ===", flush=True)

        # Cache DPO log-probs per adapter so re-evaluating against new Phase 3 utilities is instant!
        dpo_cache = out_dir / f"dpo_logprobs_{tag}.npz"
        if dpo_cache.exists():
            data = np.load(dpo_cache, allow_pickle=False)
            dpo_lps = {p: data[f"lp_{p}"] for p in eval_prompt_indices}
            dpo_lens = {p: data[f"len_{p}"] for p in eval_prompt_indices}
            print(f"  Loaded cached DPO log-probs from {dpo_cache}")
        else:
            dpo_model, dpo_tok = load_causal_lm(str(adir), device=device)
            dpo_lps, dpo_lens = score_policy_on_eval_prompts(
                model=dpo_model,
                tokenizer=dpo_tok,
                ds=ds,
                eval_prompt_indices=eval_prompt_indices,
                batch_size=ref_bs,
                max_length=ref_max_len,
                device=device,
            )
            save_dict = {"prompt_indices": np.array(eval_prompt_indices, dtype=np.int64)}
            for p in eval_prompt_indices:
                save_dict[f"lp_{p}"] = dpo_lps[p]
                save_dict[f"len_{p}"] = dpo_lens[p]
            np.savez_compressed(dpo_cache, **save_dict)
            del dpo_model, dpo_tok
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        df_ad, sum_ad = evaluate_single_adapter(
            ds=ds,
            eval_prompt_indices=eval_prompt_indices,
            u_norm=u_norm,
            ref_lps=ref_lps,
            ref_lens=ref_lens,
            dpo_lps=dpo_lps,
            dpo_lens=dpo_lens,
            dpo_beta=dpo_beta,
            emp_winrates=emp_winrates,
            local_pairs=local_pairs,
            adapter_tag=tag,
        )
        all_dfs.append(df_ad)
        all_summaries.append(sum_ad)

        print(
            f"  Pairwise fit (n={sum_ad['num_eval_pairs']:,} pairs): "
            f"acc_imp={sum_ad['overall_pair_acc_implicit_reward']:.4f}, "
            f"nll_imp={sum_ad['overall_pair_nll_implicit_reward']:.4f}, "
            f"margin={sum_ad['overall_pair_margin_implicit_reward']:+.3f}, "
            f"acc_raw_lp={sum_ad['overall_pair_acc_raw_logp']:.4f}, "
            f"acc_ln_lp={sum_ad['overall_pair_acc_len_norm_logp']:.4f}"
        )
        for rm, st in sum_ad["by_ref_mode"].items():
            print(
                f"  [{rm:20s}] med_tau={st['median_tau']:.4f}, med_dist={st['median_distortion']:.4f}, "
                f"max_dist={st['max_distortion']:.4f} | top1_emp={100*st['top1_matches_emp_best_frac']:.1f}%, "
                f"top1_opt={100*st['top1_matches_util_opt_frac']:.1f}%, rho_emp={st['mean_spearman_vs_emp']:+.3f}, "
                f"len(dpo/ref/emp)={st['mean_top1_dpo_len']:.0f}/{st['mean_top1_ref_len']:.0f}/{st['mean_top1_emp_len']:.0f}"
            )

        # Write cross-metric qualitative cases for llm_len_norm & dpo_implicit_uniform
        for rm in ("llm", "llm_len_norm", "dpo_implicit_uniform"):
            write_cross_metric_qualitative_report(
                ds=ds,
                u_norm=u_norm,
                ref_lps=ref_lps,
                ref_lens=ref_lens,
                dpo_lps=dpo_lps,
                dpo_lens=dpo_lens,
                emp_winrates=emp_winrates,
                df_mode=df_ad[df_ad["ref_mode"] == rm],
                ref_mode=rm,
                out_path=out_dir / f"dpo_qualitative_cases_{tag}_{rm}.txt",
                top_k=args.top_k_inspect,
            )

    full_df = pd.concat(all_dfs, ignore_index=True)
    csv_out = out_dir / f"phase4b_adapter_eval_{args.norm_mode.replace('-', '_')}.csv"
    full_df.to_csv(csv_out, index=False)

    summary_out = out_dir / "phase4b_adapter_eval_summary.json"
    summary_out.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "llm": llm_name,
                "user_bt_checkpoint": ckpt_path.name,
                "norm_mode": args.norm_mode,
                "adapters": all_summaries,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved full evaluation CSV to    : {csv_out}")
    print(f"Saved evaluation summary JSON to: {summary_out} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
