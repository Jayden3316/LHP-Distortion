"""Turn-1 dataset loading, user filtering, and pairwise expansion for Community Alignment."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESP_COLS = (
    "first_turn_response_a",
    "first_turn_response_b",
    "first_turn_response_c",
    "first_turn_response_d",
)
PREF_TO_SLOT = {
    "response_a": 0,
    "response_b": 1,
    "response_c": 2,
    "response_d": 3,
}


@dataclass(frozen=True)
class Turn1Dataset:
    """Structured Turn-1 pairwise preference dataset with deterministic IDs."""

    user_ids: list[str]
    user_to_idx: dict[str, int]
    prompts: list[str]
    prompt_to_idx: dict[str, int]
    prompt_responses: list[list[str]]
    prompt_is_pregenerated: list[bool]
    prompt_annotators: list[set[int]]
    unique_xy: list[tuple[int, int]]
    xy_to_global: dict[tuple[int, int], int]
    prompt_global_xy_indices: list[list[int]]
    # Each pair is (user_idx, prompt_idx, pos_global_xy_idx, neg_global_xy_idx)
    pairs: list[tuple[int, int, int, int]]
    user_turn1_counts: dict[str, int]

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_prompts(self) -> int:
        return len(self.prompts)

    @property
    def num_unique_xy(self) -> int:
        return len(self.unique_xy)

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)

    def get_xy_text(self, global_xy_idx: int) -> tuple[str, str]:
        p_idx, local_r_idx = self.unique_xy[global_xy_idx]
        return self.prompts[p_idx], self.prompt_responses[p_idx][local_r_idx]

    def prompts_with_min_users(self, min_users: int) -> list[int]:
        return [
            p_idx
            for p_idx, users in enumerate(self.prompt_annotators)
            if len(users) >= min_users
        ]


def load_turn1_dataset(
    csv_path: str | Path,
    min_user_prefs_gt: int = 20,
    max_rows: int | None = None,
) -> Turn1Dataset:
    """Load Turn-1 pairwise preferences from community_alignment_en.csv.

    Rules:
    - Restricts strictly to Turn-1 columns (ignores Turn-2/3/4).
    - Drops rows with missing/invalid `first_turn_preferred_response` or empty chosen text.
    - Retains only users with strictly more than `min_user_prefs_gt` valid Turn-1 choices.
    - Expands each 4-way choice into ordered pairs `(chosen, rejected)` for every distinct
      non-empty non-chosen candidate response (never emitting self-pairs).
    """
    csv.field_size_limit(2**31 - 1)
    path = Path(csv_path)

    raw_rows: list[tuple[str, str, bool, tuple[str, ...], int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            pref_label = (row.get("first_turn_preferred_response") or "").strip().lower()
            if pref_label not in PREF_TO_SLOT:
                continue
            uid = (row.get("annotator_id") or "").strip()
            prompt = (row.get("first_turn_prompt") or "").strip()
            if not uid or not prompt:
                continue
            resps = tuple((row.get(col) or "").strip() for col in RESP_COLS)
            chosen_slot = PREF_TO_SLOT[pref_label]
            if not resps[chosen_slot]:
                continue
            is_pregen = str(row.get("is_pregenerated_first_prompt", "")).strip().lower() == "true"
            raw_rows.append((uid, prompt, is_pregen, resps, chosen_slot))

    # Strict user threshold: count > min_user_prefs_gt
    user_counts = Counter(r[0] for r in raw_rows)
    kept_users_set = {u for u, c in user_counts.items() if c > min_user_prefs_gt}
    filtered_rows = [r for r in raw_rows if r[0] in kept_users_set]

    user_ids = sorted(kept_users_set)
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    prompts_set = sorted({r[1] for r in filtered_rows})
    prompt_to_idx = {p: i for i, p in enumerate(prompts_set)}

    prompt_resp_sets: list[set[str]] = [set() for _ in prompts_set]
    prompt_is_pregen = [False for _ in prompts_set]
    prompt_annotators: list[set[int]] = [set() for _ in prompts_set]

    for uid, prompt, is_pregen, resps, _ in filtered_rows:
        p_idx = prompt_to_idx[prompt]
        prompt_annotators[p_idx].add(user_to_idx[uid])
        if is_pregen:
            prompt_is_pregen[p_idx] = True
        for resp in resps:
            if resp:
                prompt_resp_sets[p_idx].add(resp)

    prompt_responses: list[list[str]] = [sorted(s) for s in prompt_resp_sets]
    prompt_resp_to_local: list[dict[str, int]] = [
        {resp: j for j, resp in enumerate(resps)} for resps in prompt_responses
    ]

    unique_xy: list[tuple[int, int]] = []
    xy_to_global: dict[tuple[int, int], int] = {}
    prompt_global_xy_indices: list[list[int]] = []
    for p_idx, resps in enumerate(prompt_responses):
        g_indices = []
        for r_idx in range(len(resps)):
            g_idx = len(unique_xy)
            unique_xy.append((p_idx, r_idx))
            xy_to_global[(p_idx, r_idx)] = g_idx
            g_indices.append(g_idx)
        prompt_global_xy_indices.append(g_indices)

    pairs: list[tuple[int, int, int, int]] = []
    for uid, prompt, _, resps, chosen_slot in filtered_rows:
        u_idx = user_to_idx[uid]
        p_idx = prompt_to_idx[prompt]
        chosen_text = resps[chosen_slot]
        pos_local = prompt_resp_to_local[p_idx][chosen_text]
        pos_global = xy_to_global[(p_idx, pos_local)]

        seen_neg_local: set[int] = set()
        for slot_idx, resp_text in enumerate(resps):
            if slot_idx == chosen_slot or not resp_text:
                continue
            neg_local = prompt_resp_to_local[p_idx][resp_text]
            if neg_local == pos_local or neg_local in seen_neg_local:
                continue
            seen_neg_local.add(neg_local)
            neg_global = xy_to_global[(p_idx, neg_local)]
            pairs.append((u_idx, p_idx, pos_global, neg_global))

    retained_user_counts = {u: user_counts[u] for u in user_ids}
    return Turn1Dataset(
        user_ids=user_ids,
        user_to_idx=user_to_idx,
        prompts=prompts_set,
        prompt_to_idx=prompt_to_idx,
        prompt_responses=prompt_responses,
        prompt_is_pregenerated=prompt_is_pregen,
        prompt_annotators=prompt_annotators,
        unique_xy=unique_xy,
        xy_to_global=xy_to_global,
        prompt_global_xy_indices=prompt_global_xy_indices,
        pairs=pairs,
        user_turn1_counts=retained_user_counts,
    )


def dataset_selection_summary(ds: Turn1Dataset) -> dict:
    """Compute selection diagnostics for a Turn1Dataset."""
    users_per_prompt = np.array([len(s) for s in ds.prompt_annotators], dtype=int)
    resps_per_prompt = np.array([len(r) for r in ds.prompt_responses], dtype=int)
    pair_counts_per_prompt = Counter(p_idx for _, p_idx, _, _ in ds.pairs)

    thresholds = {}
    for m in (10, 15, 20, 25, 30, 40, 50):
        mask = users_per_prompt >= m
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            thresholds[m] = {
                "num_prompts": int(len(idxs)),
                "median_resps": float(np.median(resps_per_prompt[idxs])),
                "min_resps": int(np.min(resps_per_prompt[idxs])),
                "max_resps": int(np.max(resps_per_prompt[idxs])),
                "total_unique_xy": int(np.sum(resps_per_prompt[idxs])),
                "total_pairs": int(sum(pair_counts_per_prompt[int(i)] for i in idxs)),
            }
        else:
            thresholds[m] = {
                "num_prompts": 0,
                "median_resps": 0.0,
                "min_resps": 0,
                "max_resps": 0,
                "total_unique_xy": 0,
                "total_pairs": 0,
            }

    return {
        "num_users": ds.num_users,
        "num_prompts": ds.num_prompts,
        "num_unique_xy": ds.num_unique_xy,
        "num_pairs": ds.num_pairs,
        "prompt_thresholds": thresholds,
    }


def extract_local_fixture_csv(
    source_csv: str | Path,
    dest_csv: str | Path,
    num_prompts: int = 5,
    max_users: int = 10,
) -> Path:
    """Extract a tiny deterministic Turn-1 CSV slice from the English dataset for local CPU tests."""
    csv.field_size_limit(2**31 - 1)
    src = Path(source_csv)
    dst = Path(dest_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # First pass: find top pregenerated prompts and their annotators
    prompt_rows: defaultdict[str, list[dict]] = defaultdict(list)
    fieldnames: list[str] | None = None
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if str(row.get("is_pregenerated_first_prompt", "")).strip().lower() != "true":
                continue
            if (row.get("first_turn_preferred_response") or "").strip().lower() not in PREF_TO_SLOT:
                continue
            p = (row.get("first_turn_prompt") or "").strip()
            if p:
                prompt_rows[p].append(row)

    top_prompts = [
        p for p, _ in sorted(prompt_rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:num_prompts]
    ]
    user_counter: Counter[str] = Counter()
    for p in top_prompts:
        for row in prompt_rows[p]:
            user_counter[row["annotator_id"]] += 1

    top_users = {
        u for u, _ in sorted(user_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:max_users]
    }

    selected_rows = []
    for p in top_prompts:
        for row in prompt_rows[p]:
            if row["annotator_id"] in top_users:
                selected_rows.append(row)

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    return dst
