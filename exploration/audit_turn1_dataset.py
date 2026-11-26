"""Phase 1 Turn-1 dataset selection audit for Community Alignment (English split)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data import dataset_selection_summary, load_turn1_dataset


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_turn1_dataset(DATA_CSV, min_user_prefs_gt=20)
    summary = dataset_selection_summary(ds)

    print("=== Turn-1 Dataset (K > 20 users) ===")
    print(f"Retained users      : {ds.num_users:,}")
    print(f"Unique prompts      : {ds.num_prompts:,} (pregenerated: {sum(ds.prompt_is_pregenerated):,})")
    print(f"Unique (x, y) pairs : {ds.num_unique_xy:,}")
    print(f"Pairwise comparisons: {ds.num_pairs:,}")
    print()

    print("=== Evaluation Prompt Thresholds (M = min unique retained users/prompt) ===")
    print(f"{'M':>4} | {'Prompts':>7} | {'Min |A(x)|':>10} | {'Med |A(x)|':>10} | {'Max |A(x)|':>10} | {'Total (x,y)':>11} | {'Total Pairs':>11}")
    print("-" * 78)
    for m, stats in summary["prompt_thresholds"].items():
        print(
            f"{m:>4} | {stats['num_prompts']:>7,} | {stats['min_resps']:>10,} | "
            f"{stats['median_resps']:>10.1f} | {stats['max_resps']:>10,} | "
            f"{stats['total_unique_xy']:>11,} | {stats['total_pairs']:>11,}"
        )

    # Response reuse & word-length stats on M >= 20 prompts
    eval_prompts = ds.prompts_with_min_users(20)
    pair_xy_counts: Counter[int] = Counter()
    for _, p_idx, pos_g, neg_g in ds.pairs:
        if p_idx in set(eval_prompts):
            pair_xy_counts[pos_g] += 1
            pair_xy_counts[neg_g] += 1

    prompt_words = [len(ds.prompts[p].split()) for p in eval_prompts]
    resp_words = [len(r.split()) for p in eval_prompts for r in ds.prompt_responses[p]]
    xy_appearances = [pair_xy_counts[g] for p in eval_prompts for g in ds.prompt_global_xy_indices[p]]

    print("\n=== Length & Reuse Stats on M >= 20 Prompts ===")
    print(
        f"Prompt word count  : mean={np.mean(prompt_words):.1f}, median={np.median(prompt_words):.0f}, "
        f"p95={np.percentile(prompt_words, 95):.0f}, max={np.max(prompt_words)}"
    )
    print(
        f"Response word count: mean={np.mean(resp_words):.1f}, median={np.median(resp_words):.0f}, "
        f"p95={np.percentile(resp_words, 95):.0f}, p99={np.percentile(resp_words, 99):.0f}, max={np.max(resp_words)}"
    )
    print(
        f"Pairwise appearances per unique (x,y): mean={np.mean(xy_appearances):.1f}, "
        f"median={np.median(xy_appearances):.0f}, p90={np.percentile(xy_appearances, 90):.0f}, max={np.max(xy_appearances)}"
    )

    out_json = RESULTS_DIR / "turn1_selection_audit.json"
    out_json.write_text(
        json.dumps(
            {
                **summary,
                "m20_prompt_words": {
                    "mean": float(np.mean(prompt_words)),
                    "median": float(np.median(prompt_words)),
                    "p95": float(np.percentile(prompt_words, 95)),
                    "max": int(np.max(prompt_words)),
                },
                "m20_response_words": {
                    "mean": float(np.mean(resp_words)),
                    "median": float(np.median(resp_words)),
                    "p95": float(np.percentile(resp_words, 95)),
                    "p99": float(np.percentile(resp_words, 99)),
                    "max": int(np.max(resp_words)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved audit JSON to: {out_json}")


if __name__ == "__main__":
    main()
