#!/usr/bin/env python3
"""Score advertised discounts against each variant's own price history.

The question is not "is there a discount badge" but "is this price actually
lower than what this exact variant normally sells for". Every variant is judged
against itself, so the result does not depend on comparing across brands.

Definitions, all per variant:

  claimed   = (compare_at_price - price) / compare_at_price   on the sale date
  baseline  = median observed price, excluding the blackout window and the sale
              date itself. The blackout matters: prices inflated in the fortnight
              before a sale would otherwise raise the baseline and launder a fake
              discount into a real-looking one.
  real      = (baseline - price) / baseline                   on the sale date

A variant that advertises a discount is then one of:

  phantom     real <= 0           the price is not below its own normal price
  overstated  0 < real < claimed/2 less than half the advertised saving is real
  honest      real >= claimed/2

Reported separately, because they are different behaviours:

  permanent_sale  compare_at_price was set on >= 90% of observed days, so the
                  "regular price" the discount is measured against never applied
  pre_inflation   the price rose above baseline during the blackout window before
                  being cut for the sale

Headline number is the phantom rate among variants advertising a discount, which
is the directly comparable figure to published big-retail studies.

    python scripts/analyze.py --sale-date 2026-11-27
    python scripts/analyze.py --sale-date 2026-11-27 --data data/snapshots
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/prices.parquet"))
    if not paths:
        raise SystemExit(f"no snapshots found under {root}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_catalog(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/new_products.parquet"))
    frames = [pd.read_parquet(p) for p in paths if p.stat().st_size > 0]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["domain", "variant_id", "product_title"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["domain", "variant_id"], keep="first"
    )


def classify(
    prices: pd.DataFrame,
    sale_date: pd.Timestamp,
    blackout_days: int,
    min_baseline_days: int,
    phantom_tolerance: float,
) -> pd.DataFrame:
    blackout_start = sale_date - timedelta(days=blackout_days)

    sale = prices[prices["date"] == sale_date].copy()
    if sale.empty:
        raise SystemExit(f"no snapshot for sale date {sale_date.date()}")

    baseline_rows = prices[prices["date"] < blackout_start]
    if baseline_rows.empty:
        raise SystemExit(
            f"no snapshots before {blackout_start.date()}; "
            f"baseline window is empty. Reduce --blackout-days."
        )

    key = ["domain", "variant_id"]

    base = (
        baseline_rows.groupby(key)["price"]
        .agg(baseline="median", baseline_days="count", baseline_min="min")
        .reset_index()
    )

    blackout_rows = prices[
        (prices["date"] >= blackout_start) & (prices["date"] < sale_date)
    ]
    blackout = (
        blackout_rows.groupby(key)["price"]
        .agg(blackout_max="max")
        .reset_index()
    )

    # How often was a "was" price displayed at all, across the whole window.
    flagged = prices.assign(on_sale=prices["compare_at_price"].notna())
    sale_freq = (
        flagged.groupby(key)["on_sale"]
        .agg(days_observed="count", days_flagged="sum")
        .reset_index()
    )

    df = (
        sale.merge(base, on=key, how="inner")
        .merge(blackout, on=key, how="left")
        .merge(sale_freq, on=key, how="left")
    )

    df = df[df["baseline_days"] >= min_baseline_days]
    df = df[df["price"].notna() & (df["price"] > 0)]
    df = df[df["baseline"].notna() & (df["baseline"] > 0)]
    if df.empty:
        raise SystemExit("no variants met the minimum baseline requirement")

    df["claimed"] = np.where(
        df["compare_at_price"].notna() & (df["compare_at_price"] > df["price"]),
        (df["compare_at_price"] - df["price"]) / df["compare_at_price"],
        0.0,
    )
    df["real"] = (df["baseline"] - df["price"]) / df["baseline"]
    df["advertised"] = df["claimed"] > 0

    df["permanent_sale"] = (df["days_flagged"] / df["days_observed"]) >= 0.90
    df["pre_inflation"] = df["blackout_max"] > df["baseline"] * 1.02

    def label(row) -> str:
        if not row["advertised"]:
            return "no_claim"
        # Tolerance band, not a bare <= 0 test. Retail prices wobble by small
        # amounts for reasons unrelated to discounting (rounding, currency, A/B
        # tests), and a hard zero boundary misfiles those wobbles as real
        # savings. Anything within the band is no saving in any meaningful sense.
        if row["real"] <= phantom_tolerance:
            return "phantom"
        if row["real"] < row["claimed"] / 2:
            return "overstated"
        return "honest"

    df["verdict"] = df.apply(label, axis=1)
    return df


def summarise(df: pd.DataFrame) -> dict:
    adv = df[df["advertised"]]
    n_adv = len(adv)

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    counts = adv["verdict"].value_counts().to_dict()
    phantom = int(counts.get("phantom", 0))
    overstated = int(counts.get("overstated", 0))
    honest = int(counts.get("honest", 0))

    return {
        "variants_scored": int(len(df)),
        "variants_advertising_discount": n_adv,
        "share_advertising_discount_pct": pct(n_adv, len(df)),
        "phantom": phantom,
        "phantom_rate_pct": pct(phantom, n_adv),
        "overstated": overstated,
        "overstated_rate_pct": pct(overstated, n_adv),
        "honest": honest,
        "honest_rate_pct": pct(honest, n_adv),
        "median_claimed_discount_pct": round(100 * adv["claimed"].median(), 2)
        if n_adv
        else 0.0,
        "median_real_discount_pct": round(100 * adv["real"].median(), 2)
        if n_adv
        else 0.0,
        "permanent_sale_variants": int(df["permanent_sale"].sum()),
        "permanent_sale_rate_pct": pct(int(df["permanent_sale"].sum()), len(df)),
        "pre_inflation_variants": int(df["pre_inflation"].sum()),
        "pre_inflation_rate_pct": pct(int(df["pre_inflation"].sum()), len(df)),
    }


def per_group(df: pd.DataFrame, col: str, min_n: int = 0) -> pd.DataFrame:
    """Aggregate verdicts by a column, suppressing groups too small to report.

    Creator and celebrity stores often carry only 20 to 40 products, so a single
    brand's percentage swings wildly on a handful of variants. Publishing such a
    number next to a 2,000-variant retailer's invites a false comparison and,
    worse, puts a named small merchant at the top of a "worst offenders" list on
    noise alone. Groups below the floor are dropped from brand-level output and
    still counted in the segment and headline figures.
    """
    adv = df[df["advertised"]]
    if adv.empty:
        return pd.DataFrame()
    if min_n > 0:
        big = adv.groupby(col)[col].transform("size") >= min_n
        adv = adv[big]
        if adv.empty:
            return pd.DataFrame()
    g = adv.groupby(col)
    out = pd.DataFrame(
        {
            "advertised_variants": g.size(),
            "phantom": g["verdict"].apply(lambda s: (s == "phantom").sum()),
            "overstated": g["verdict"].apply(lambda s: (s == "overstated").sum()),
            "honest": g["verdict"].apply(lambda s: (s == "honest").sum()),
            "median_claimed_pct": g["claimed"].median() * 100,
            "median_real_pct": g["real"].median() * 100,
        }
    ).reset_index()
    out["phantom_rate_pct"] = (
        100 * out["phantom"] / out["advertised_variants"]
    ).round(2)
    out["honesty_score"] = (100 - out["phantom_rate_pct"]).round(2)
    out["median_claimed_pct"] = out["median_claimed_pct"].round(2)
    out["median_real_pct"] = out["median_real_pct"].round(2)
    return out.sort_values("phantom_rate_pct", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/snapshots")
    ap.add_argument("--sale-date", required=True)
    ap.add_argument("--blackout-days", type=int, default=14)
    ap.add_argument("--min-baseline-days", type=int, default=10)
    ap.add_argument(
        "--phantom-tolerance",
        type=float,
        default=0.005,
        help="real saving at or below this fraction counts as no saving",
    )
    ap.add_argument("--brands", default="brands.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--min-brand-variants",
        type=int,
        default=25,
        help=(
            "brands with fewer advertised variants are left out of brand-level "
            "output; they still count in category and headline figures"
        ),
    )
    args = ap.parse_args()

    root = Path(args.data)
    sale_date = pd.Timestamp(args.sale_date)

    prices = load_prices(root)
    print(
        f"loaded {len(prices):,} price rows across "
        f"{prices['date'].nunique()} days "
        f"({prices['date'].min().date()} to {prices['date'].max().date()})"
    )

    df = classify(
        prices,
        sale_date,
        args.blackout_days,
        args.min_baseline_days,
        args.phantom_tolerance,
    )

    cats: dict[str, str] = {}
    bp = Path(args.brands)
    if bp.exists():
        with bp.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cats[r["domain"]] = r.get("category", "")
    df["category"] = df["domain"].map(cats).fillna("")

    stats = summarise(df)
    stats["sale_date"] = str(sale_date.date())
    stats["blackout_days"] = args.blackout_days
    stats["phantom_tolerance"] = args.phantom_tolerance
    stats["baseline_window_days"] = int(
        (sale_date - timedelta(days=args.blackout_days) - prices["date"].min()).days
    )
    stats["stores"] = int(df["domain"].nunique())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "headline.json").write_text(json.dumps(stats, indent=2) + "\n")

    by_brand = per_group(df, "domain", min_n=args.min_brand_variants)
    by_cat = per_group(df, "category")
    stats["min_brand_variants"] = args.min_brand_variants
    stats["brands_reportable"] = int(len(by_brand))
    if not by_brand.empty:
        by_brand.to_csv(out / "by_brand.csv", index=False)
    if not by_cat.empty:
        by_cat.to_csv(out / "by_category.csv", index=False)

    keep = [
        "domain",
        "category",
        "product_id",
        "variant_id",
        "price",
        "compare_at_price",
        "baseline",
        "baseline_days",
        "claimed",
        "real",
        "verdict",
        "permanent_sale",
        "pre_inflation",
    ]
    df[keep].to_parquet(out / "scored_variants.parquet", index=False)

    print("\n" + "=" * 58)
    print(f"  SALE DATE {stats['sale_date']}   {stats['stores']} stores")
    print("=" * 58)
    for k in [
        "variants_scored",
        "variants_advertising_discount",
        "share_advertising_discount_pct",
        "phantom_rate_pct",
        "overstated_rate_pct",
        "honest_rate_pct",
        "median_claimed_discount_pct",
        "median_real_discount_pct",
        "permanent_sale_rate_pct",
        "pre_inflation_rate_pct",
    ]:
        print(f"  {k:<34} {stats[k]}")
    print("=" * 58)
    print(
        f"\n  HEADLINE: {stats['phantom_rate_pct']}% of advertised discounts are "
        f"phantom\n  (price not below the variant's own pre-sale baseline)\n"
    )
    print(f"  wrote {out}/headline.json, by_brand.csv, by_category.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
