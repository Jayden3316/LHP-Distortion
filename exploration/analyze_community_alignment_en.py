"""Analyze per-user preference query counts in the English split of Community Alignment."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"
RESULTS_DIR = ROOT / "results"

TURN_PREF_COLS = [
    "first_turn_preferred_response",
    "second_turn_preferred_response",
    "third_turn_preferred_response",
    "fourth_turn_preferred_response",
]


def main() -> None:
    csv.field_size_limit(2**31 - 1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    user_convs: Counter[str] = Counter()
    user_pref_queries: Counter[str] = Counter()
    user_pregen_convs: Counter[str] = Counter()
    user_self_convs: Counter[str] = Counter()
    user_turn_breakdown: dict[int, Counter[str]] = {t: Counter() for t in range(1, 5)}

    pref_value_counts: Counter[str] = Counter()
    prompt_users: defaultdict[str, set[str]] = defaultdict(set)
    pregen_prompt_users: defaultdict[str, set[str]] = defaultdict(set)

    total_convs = 0
    total_pref_queries = 0

    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_convs += 1
            uid = row["annotator_id"]
            user_convs[uid] += 1

            is_pregen = str(row.get("is_pregenerated_first_prompt", "")).strip().lower() == "true"
            if is_pregen:
                user_pregen_convs[uid] += 1
            else:
                user_self_convs[uid] += 1

            first_prompt = (row.get("first_turn_prompt") or "").strip()
            if first_prompt:
                prompt_users[first_prompt].add(uid)
                if is_pregen:
                    pregen_prompt_users[first_prompt].add(uid)

            for turn_idx, col in enumerate(TURN_PREF_COLS, start=1):
                val = (row.get(col) or "").strip()
                if val:
                    user_pref_queries[uid] += 1
                    user_turn_breakdown[turn_idx][uid] += 1
                    pref_value_counts[val] += 1
                    total_pref_queries += 1

    users = sorted(user_convs.keys())
    df = pd.DataFrame(
        [
            {
                "annotator_id": uid,
                "num_pref_queries": user_pref_queries[uid],
                "num_conversations": user_convs[uid],
                "num_pregen_conversations": user_pregen_convs[uid],
                "num_self_conversations": user_self_convs[uid],
                "turn1_prefs": user_turn_breakdown[1][uid],
                "turn2_prefs": user_turn_breakdown[2][uid],
                "turn3_prefs": user_turn_breakdown[3][uid],
                "turn4_prefs": user_turn_breakdown[4][uid],
            }
            for uid in users
        ]
    )

    csv_out = RESULTS_DIR / "community_alignment_en_user_counts.csv"
    df.to_csv(csv_out, index=False)

    print(f"English conversations: {total_convs:,}")
    print(f"English preference queries (turns with a preference): {total_pref_queries:,}")
    print(f"Unique English annotators: {len(df):,}")
    print(f"Preferred response values: {dict(pref_value_counts.most_common())}")
    print()

    print("=== Per-User Summary Statistics (English split) ===")
    cols = ["num_pref_queries", "num_conversations", "num_pregen_conversations", "num_self_conversations"]
    print(df[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(2).to_string())
    print()

    print("=== Turns with Preference Breakdown ===")
    for t in range(1, 5):
        total_t = sum(user_turn_breakdown[t].values())
        print(f"  Turn {t}: {total_t:,} preferences")
    print()

    # Prompt overlap summary
    overlap_counts = [len(u) for u in prompt_users.values()]
    pregen_overlap = [len(u) for u in pregen_prompt_users.values()]
    print("=== First-Turn Prompt Overlap Across Users (English) ===")
    print(f"Unique first-turn prompts: {len(prompt_users):,} (pregenerated: {len(pregen_prompt_users):,})")
    print(f"Prompts rated by >=2 users : {sum(c >= 2 for c in overlap_counts):,}")
    print(f"Prompts rated by >=5 users : {sum(c >= 5 for c in overlap_counts):,}")
    print(f"Prompts rated by >=10 users: {sum(c >= 10 for c in overlap_counts):,}")
    print(f"Max users rating a single prompt: {max(overlap_counts) if overlap_counts else 0}")
    if pregen_overlap:
        s = pd.Series(pregen_overlap)
        print(
            f"Pregenerated prompts users/prompt: mean={s.mean():.2f}, median={s.median():.0f}, "
            f"p90={s.quantile(0.9):.0f}, max={s.max()}"
        )

    # Plot 3-panel histogram
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)

    # Panel 1: Preference queries per user
    ax = axes[0]
    q_vals = df["num_pref_queries"]
    ax.hist(q_vals, bins=40, color="#2563eb", edgecolor="white", alpha=0.88)
    ax.axvline(q_vals.mean(), color="#dc2626", linestyle="--", linewidth=1.6, label=f"Mean = {q_vals.mean():.1f}")
    ax.axvline(q_vals.median(), color="#0f766e", linestyle=":", linewidth=1.8, label=f"Median = {q_vals.median():.0f}")
    ax.set_title("Preference Queries per User", fontsize=12, fontweight="bold")
    ax.set_xlabel("Queries with a Preference (across turns 1–4)")
    ax.set_ylabel("Number of Users")
    ax.legend(frameon=True, fontsize=9)

    # Panel 2: Conversations per user
    ax = axes[1]
    c_vals = df["num_conversations"]
    ax.hist(c_vals, bins=40, color="#059669", edgecolor="white", alpha=0.88)
    ax.axvline(c_vals.mean(), color="#dc2626", linestyle="--", linewidth=1.6, label=f"Mean = {c_vals.mean():.1f}")
    ax.axvline(c_vals.median(), color="#0f766e", linestyle=":", linewidth=1.8, label=f"Median = {c_vals.median():.0f}")
    ax.set_title("Conversations per User", fontsize=12, fontweight="bold")
    ax.set_xlabel("Conversations")
    ax.set_ylabel("Number of Users")
    ax.legend(frameon=True, fontsize=9)

    # Panel 3: Users per pregenerated first-turn prompt
    ax = axes[2]
    p_vals = pd.Series(pregen_overlap)
    ax.hist(p_vals, bins=range(1, int(p_vals.max()) + 2), color="#7c3aed", edgecolor="white", alpha=0.88, align="left")
    ax.axvline(p_vals.mean(), color="#dc2626", linestyle="--", linewidth=1.6, label=f"Mean = {p_vals.mean():.1f}")
    ax.axvline(p_vals.median(), color="#0f766e", linestyle=":", linewidth=1.8, label=f"Median = {p_vals.median():.0f}")
    ax.set_title("Users per Pregenerated First-Turn Prompt", fontsize=12, fontweight="bold")
    ax.set_xlabel("Unique Users Rating Prompt")
    ax.set_ylabel("Number of Prompts")
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        f"Community Alignment Dataset (English Split): {len(df):,} Users, {total_pref_queries:,} Preference Queries, {total_convs:,} Conversations",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig_path = RESULTS_DIR / "community_alignment_en_user_counts_histogram.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved per-user counts to: {csv_out}")
    print(f"Saved histogram to: {fig_path}")


if __name__ == "__main__":
    main()
