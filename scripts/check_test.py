#!/usr/bin/env python3
"""Compare analyze.py output against the planted ground truth in the test data."""
from __future__ import annotations

import pandas as pd

EXPECTED_VERDICT = {
    "honest": "honest",
    "phantom": "phantom",
    "overstated": "overstated",
    "inflated": "overstated",  # a 1% real cut sold as 24% off is overstated
    "permanent": "phantom",    # price never moved, so the sale saves nothing
    "no_claim": "no_claim",
}

EXPECTED_FLAGS = {
    "inflated": "pre_inflation",
    "permanent": "permanent_sale",
}


def main() -> int:
    truth = pd.read_csv("data/test/ground_truth.csv")
    scored = pd.read_parquet("results/test/scored_variants.parquet")
    df = scored.merge(truth, on="variant_id", how="inner")

    df["expected"] = df["behaviour"].map(EXPECTED_VERDICT)
    df["correct"] = df["verdict"] == df["expected"]

    print(f"{len(df)} variants scored against ground truth\n")
    print(f"{'behaviour':<12} {'n':>5} {'correct':>8} {'accuracy':>9}   verdicts seen")
    print("-" * 74)

    failures = 0
    for behaviour, grp in df.groupby("behaviour"):
        n, ok = len(grp), int(grp["correct"].sum())
        acc = 100.0 * ok / n
        seen = ", ".join(
            f"{v}:{c}" for v, c in grp["verdict"].value_counts().items()
        )
        print(f"{behaviour:<12} {n:>5} {ok:>8} {acc:>8.1f}%   {seen}")
        if acc < 99.0:
            failures += 1

    print("\nflag checks")
    print("-" * 74)
    for behaviour, flag in EXPECTED_FLAGS.items():
        grp = df[df["behaviour"] == behaviour]
        hit = int(grp[flag].sum())
        acc = 100.0 * hit / len(grp)
        print(f"{behaviour:<12} {flag:<16} {hit}/{len(grp)}  {acc:.1f}%")
        if acc < 99.0:
            failures += 1

        # The flag must not fire on everything else, or it means nothing.
        others = df[df["behaviour"] != behaviour]
        fp = int(others[flag].sum())
        fp_rate = 100.0 * fp / len(others)
        print(f"{'':<12} {'false positives':<16} {fp}/{len(others)}  {fp_rate:.1f}%")
        if fp_rate > 1.0:
            failures += 1

    overall = 100.0 * df["correct"].sum() / len(df)
    print("\n" + "=" * 74)
    print(f"  overall verdict accuracy  {overall:.2f}%")
    print(f"  {'PASS' if failures == 0 else f'FAIL ({failures} checks below threshold)'}")
    print("=" * 74)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
