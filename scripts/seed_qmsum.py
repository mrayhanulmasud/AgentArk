"""Seed ./datasets/data/QMSum.json from a HuggingFace Hub mirror.

inference.py reads local JSON at `./datasets/data/{test_dataset_name}.json`
and expects each record to have at least `query` and `gt` fields. The
repo does not ship QMSum.json, so run this once to materialize it from
`pszemraj/qmsum-cleaned` (SCROLLS-style `{input, output}` schema) or
from any other mirror passed via --dataset_hub_path.

Usage:
    python scripts/seed_qmsum.py
    python scripts/seed_qmsum.py --split test --max_samples 50
    python scripts/seed_qmsum.py --dataset_hub_path other/mirror
"""
import argparse
import json
import os
import sys

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_hub_path", default="pszemraj/qmsum-cleaned")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--output_path", default="datasets/data/QMSum.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap record count (useful for smoke tests).")
    args = parser.parse_args()

    print(f"[seed_qmsum] loading {args.dataset_hub_path} split={args.split}")
    dataset = load_dataset(args.dataset_hub_path, split=args.split)
    columns = set(dataset.column_names)

    if "meeting_transcripts" in columns:
        # Original Yale-LILY schema — one row per meeting with query lists.
        records = []
        for entry in dataset:
            transcript = "\n".join(
                f"{t['speaker']}: {t['content']}"
                for t in entry["meeting_transcripts"]
            )
            for q in entry.get("general_query_list", []):
                records.append({
                    "query": f"Question: {q['query']}\n\n## Meeting Transcript\n{transcript}",
                    "gt": q["answer"],
                    "topic": entry.get("topic", ""),
                    "source": "QMSum",
                })
            for q in entry.get("specific_query_list", []):
                records.append({
                    "query": f"Question: {q['query']}\n\n## Meeting Transcript\n{transcript}",
                    "gt": q["answer"],
                    "topic": entry.get("topic", ""),
                    "source": "QMSum",
                })
    elif "input" in columns and "output" in columns:
        # SCROLLS-style schema — one row per query, pre-formatted prompt.
        records = [
            {
                "query": entry["input"],
                "gt": entry["output"],
                "topic": entry.get("id", ""),
                "source": "QMSum",
            }
            for entry in dataset
        ]
    else:
        print(f"[seed_qmsum] unexpected schema: {sorted(columns)}",
              file=sys.stderr)
        sys.exit(1)

    if args.max_samples is not None:
        records = records[: args.max_samples]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[seed_qmsum] wrote {len(records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
