#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the deterministic Chapman split manifest."
    )
    parser.add_argument(
        "--root",
        default="data/chapman_shaoxing",
        help="Dataset root containing WFDBRecords",
    )
    args = parser.parse_args()

    root = Path(args.root)
    candidates = sorted(root.glob("medtsllm_manifest_*.csv"))
    if not candidates:
        raise SystemExit(
            f"No medtsllm_manifest_*.csv found in {root}. "
            "Start training once; the dataset loader creates it automatically."
        )

    for manifest in candidates:
        print("=" * 100)
        print(manifest)
        df = pd.read_csv(manifest)
        print(f"Total eligible records: {len(df):,}")
        print("\nSplit counts:")
        print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())
        print("\nClass x split:")
        print(
            df.groupby(["rhythm", "split"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["train", "val", "test"], fill_value=0)
            .to_string()
        )


if __name__ == "__main__":
    main()
