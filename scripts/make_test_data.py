#!/usr/bin/env python3
"""Generate synthetic snapshots with known planted behaviour, to test analyze.py.

Each variant is built to fall into exactly one category, so the analyzer's output
can be checked against the ground truth rather than eyeballed.

    python scripts/make_test_data.py --out data/test
    python scripts/analyze.py --data data/test --sale-date 2026-11-27 --brands ""
"""
from __future__ import annotations

import argparse
import random
from datetime import timedelta
from pathlib import Path

import pandas as pd

SALE_DATE = pd.Timestamp("2026-11-27")
START = pd.Timestamp("2026-10-07")

# (count, behaviour) -> expected verdict among advertised variants
PLAN = [
    (300, "honest"),      # genuine cut below the long-run price
    (250, "phantom"),     # badge on, price unchanged from baseline
    (150, "overstated"),  # badge claims 50%, real cut is tiny
    (120, "inflated"),    # price raised during blackout, then "cut" back to normal
    (100, "permanent"),   # compare_at always set, price never moves
    (280, "no_claim"),    # no badge at all
]


def build() -> pd.DataFrame:
    rng = random.Random(20261127)
    days = pd.date_range(START, SALE_DATE, freq="D")
    blackout_start = SALE_DATE - timedelta(days=14)

    rows = []
    vid = 1
    for count, behaviour in PLAN:
        for _ in range(count):
            vid += 1
            domain = f"store{vid % 12:02d}.example"
            base = round(rng.uniform(20, 240), 2)

            for d in days:
                price = base
                compare = None
                in_blackout = blackout_start <= d < SALE_DATE
                is_sale_day = d == SALE_DATE

                if behaviour == "honest":
                    if is_sale_day:
                        price = round(base * 0.70, 2)
                        compare = base
                elif behaviour == "phantom":
                    if is_sale_day:
                        price = base
                        compare = round(base * 1.45, 2)
                elif behaviour == "overstated":
                    if is_sale_day:
                        price = round(base * 0.97, 2)
                        compare = round(base * 1.60, 2)
                elif behaviour == "inflated":
                    if in_blackout:
                        price = round(base * 1.30, 2)
                    elif is_sale_day:
                        price = round(base * 0.99, 2)
                        compare = round(base * 1.30, 2)
                elif behaviour == "permanent":
                    compare = round(base * 1.50, 2)
                    price = base
                # no_claim leaves price flat and compare None

                # A little noise everywhere, so medians are doing real work.
                if not is_sale_day and behaviour != "permanent":
                    price = round(price * rng.uniform(0.995, 1.005), 2)

                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "domain": domain,
                        "product_id": vid,
                        "variant_id": vid,
                        "price": price,
                        "compare_at_price": compare,
                        "available": True,
                        "behaviour": behaviour,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/test")
    args = ap.parse_args()

    df = build()
    root = Path(args.out)
    truth = df[["variant_id", "behaviour"]].drop_duplicates()

    for date, chunk in df.groupby("date"):
        d = root / str(date)
        d.mkdir(parents=True, exist_ok=True)
        chunk.drop(columns=["behaviour"]).to_parquet(
            d / "prices.parquet", index=False
        )
        pd.DataFrame(
            columns=["first_seen", "domain", "product_id", "variant_id"]
        ).to_parquet(d / "new_products.parquet", index=False)

    truth.to_csv(root / "ground_truth.csv", index=False)
    print(
        f"wrote {df['date'].nunique()} days, "
        f"{df['variant_id'].nunique()} variants to {root}"
    )
    for count, behaviour in PLAN:
        print(f"  {behaviour:<12} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
