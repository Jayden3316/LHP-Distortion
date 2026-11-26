"""Analyze the number of PRISM dataset data points associated with each user."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "prism-alignment"
RESULTS_DIR = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    survey = load_jsonl(DATA_DIR / "survey.jsonl")
    conversations = load_jsonl(DATA_DIR / "conversations.jsonl")
    utterances = load_jsonl(DATA_DIR / "utterances.jsonl")

    all_users = [row["user_id"] for row in survey]
    conv_counts = Counter(row["user_id"] for row in conversations)
    utt_counts = Counter(row["user_id"] for row in utterances)

    # Unique (user_id, interaction_id) pairs represent individual interaction turns
    user_interactions: dict[str, set[str]] = {}
    for row in utterances:
        user_interactions.setdefault(row["user_id"], set()).add(row["interaction_id"])

    records = []
    for uid in all_users:
        records.append(
            {
                "user_id": uid,
                "num_conversations": conv_counts.get(uid, 0),
                "num_interactions": len(user_interactions.get(uid, set())),
                "num_utterances": utt_counts.get(uid, 0),
            }
        )

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "prism_user_counts.csv"
    df.to_csv(csv_path, index=False)

    active = df[df["num_utterances"] > 0]
    survey_only = df[df["num_utterances"] == 0]

    print(f"Total surveyed users: {len(df)}")
    print(f"Active users (with >=1 conversation/utterance): {len(active)}")
    print(f"Survey-only users (0 conversations/utterances): {len(survey_only)}")
    print()

    for label, subset in [("Active users (n=1396)", active), ("All surveyed users (n=1500)", df)]:
        print(f"=== {label} ===")
        stats = subset[["num_utterances", "num_interactions", "num_conversations"]].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
        print(stats.round(2).to_string())
        print()

    print("=== Exact conversation count distribution (all 1,500 users) ===")
    print(df["num_conversations"].value_counts().sort_index().to_string())

    # Plot 3-panel histogram for active users (with survey-only count annotated)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)

    # Panel 1: Utterances per user
    ax = axes[0]
    u_vals = active["num_utterances"]
    bins_u = np.arange(0, u_vals.max() + 5, 4)
    ax.hist(u_vals, bins=bins_u, color="#2563eb", edgecolor="white", alpha=0.88)
    ax.axvline(u_vals.mean(), color="#dc2626", linestyle="--", linewidth=1.6, label=f"Mean = {u_vals.mean():.1f}")
    ax.axvline(u_vals.median(), color="#0f766e", linestyle=":", linewidth=1.8, label=f"Median = {u_vals.median():.0f}")
    ax.set_title("Scored Utterances per User", fontsize=12, fontweight="bold")
    ax.set_xlabel("Utterances (prompt-response-score rows)")
    ax.set_ylabel("Number of Users")
    ax.legend(frameon=True, fontsize=9)

    # Panel 2: Interaction turns per user
    ax = axes[1]
    i_vals = active["num_interactions"]
    bins_i = np.arange(0, i_vals.max() + 3, 2)
    ax.hist(i_vals, bins=bins_i, color="#7c3aed", edgecolor="white", alpha=0.88)
    ax.axvline(i_vals.mean(), color="#dc2626", linestyle="--", linewidth=1.6, label=f"Mean = {i_vals.mean():.1f}")
    ax.axvline(i_vals.median(), color="#0f766e", linestyle=":", linewidth=1.8, label=f"Median = {i_vals.median():.0f}")
    ax.set_title("Interaction Turns per User", fontsize=12, fontweight="bold")
    ax.set_xlabel("Interaction Turns (unique interaction_id)")
    ax.set_ylabel("Number of Users")
    ax.legend(frameon=True, fontsize=9)

    # Panel 3: Conversations per user (including 0 for survey-only users)
    ax = axes[2]
    conv_vc = df["num_conversations"].value_counts().sort_index()
    colors = ["#94a3b8" if k == 0 else "#059669" for k in conv_vc.index]
    bars = ax.bar(conv_vc.index, conv_vc.values, color=colors, edgecolor="white", width=0.75)
    for bar, count in zip(bars, conv_vc.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 18,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(conv_vc.index)
    ax.set_ylim(0, conv_vc.max() * 1.13)
    ax.set_title("Conversations per User (All 1,500 Users)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Conversations (0 = survey-only)")
    ax.set_ylabel("Number of Users")

    fig.suptitle(
        "PRISM Alignment Dataset (HannahRoseKirk/prism-alignment): Data Points per User\n"
        f"(1,396 active users in panels 1–2; {len(survey_only)} survey-only users with 0 conversations shown in gray in panel 3)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    fig_path = RESULTS_DIR / "prism_user_counts_histogram.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved per-user counts to: {csv_path}")
    print(f"Saved histogram to: {fig_path}")


if __name__ == "__main__":
    main()
