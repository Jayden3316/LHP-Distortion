"""Stream facebook/community-alignment-dataset and save only the English split."""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

import requests


URL = "https://huggingface.co/datasets/facebook/community-alignment-dataset/resolve/main/community_alignment.csv"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "community-alignment"
OUT_CSV = OUT_DIR / "community_alignment_en.csv"


def main() -> None:
    csv.field_size_limit(2**31 - 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {URL} ...")
    lang_counts: Counter[str] = Counter()
    en_rows = 0
    total_rows = 0

    with requests.get(URL, stream=True, allow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        resp.raw.auto_close = False
        text_stream = io.TextIOWrapper(resp.raw, encoding="utf-8", newline="")
        reader = csv.DictReader(text_stream)
        assert reader.fieldnames is not None

        with OUT_CSV.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                lang = row.get("assigned_lang", "")
                lang_counts[lang] += 1
                if lang == "en":
                    writer.writerow(row)
                    en_rows += 1
                if total_rows % 10000 == 0:
                    print(f"  Processed {total_rows:,} rows (en: {en_rows:,})...", flush=True)

    print(f"\nDone! Total rows streamed: {total_rows:,}")
    print("Rows by assigned_lang:")
    for lang, cnt in lang_counts.most_common():
        print(f"  {lang}: {cnt:,}")
    print(f"Saved {en_rows:,} English rows to {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
