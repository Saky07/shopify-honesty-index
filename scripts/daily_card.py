#!/usr/bin/env python3
"""Render the day's post image from whatever snapshots exist.

A 52-day series needs something true to say on day 19, and the collector runs
unattended, so the daily news has to come out of the data rather than be invented.
The number this tracks is the share of variants displaying a struck-through "was"
price, which is measurable from day one, long before Black Friday: it is the
standing level of discount signalling across the sample, and the line it traces
over seven weeks is the story.

    python scripts/daily_card.py
    python scripts/daily_card.py --data data/snapshots --out posts/

Writes posts/day-NN.png at 1600x900, sized for a social post.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

SALE = date(2026, 11, 27)

# Validated reference palette, light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"
# matplotlib warns once per text object for every family it cannot resolve, so
# handing it a wishlist produces dozens of lines per run. Resolve once instead,
# against what is actually installed: Helvetica on macOS, DejaVu on CI.
FONT_PREFERENCE = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def pick_font() -> str:
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_PREFERENCE:
        if name in installed:
            return name
    return plt.rcParams["font.sans-serif"][0]


def load_daily(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/prices.parquet")):
        if path.stat().st_size == 0:
            continue
        try:
            df = pd.read_parquet(path, columns=["date", "compare_at_price", "price"])
        except Exception as exc:
            print(f"  warn: skipping {path}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        flagged = df["compare_at_price"].notna() & (
            df["compare_at_price"] > df["price"]
        )
        rows.append(
            {
                "date": pd.to_datetime(df["date"].iloc[0]).date(),
                "variants": len(df),
                "flagged": int(flagged.sum()),
                "pct": 100.0 * flagged.sum() / len(df),
            }
        )
    if not rows:
        raise SystemExit(
            f"no usable snapshots under {root}. Run scripts/collect.py first."
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def window(daily: pd.DataFrame) -> tuple[date, int]:
    """Day 1 is the first snapshot on disk, not a date written into the source.

    Snapshot dates are UTC, so the first one depends on when collection actually
    started rather than on when the project was planned. Deriving it means the
    counter cannot drift from the data it is counting.
    """
    start = daily["date"].iloc[0]
    return start, (SALE - start).days + 1


def store_count(brands_path: Path) -> int | None:
    """Live store count, read rather than hardcoded so the card can't drift."""
    if not brands_path.exists():
        return None
    try:
        import csv

        with brands_path.open(newline="", encoding="utf-8") as fh:
            return sum(1 for r in csv.DictReader(fh) if r.get("status") == "live")
    except Exception:
        return None


def render(daily: pd.DataFrame, out_path: Path, stores: int | None) -> dict:
    latest = daily.iloc[-1]
    start, total_days = window(daily)
    day_n = (latest["date"] - start).days + 1
    days_left = (SALE - latest["date"]).days

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    plt.rcParams["font.family"] = pick_font()

    # Header band: the counter is the series' spine, so it leads.
    fig.text(
        0.055,
        0.925,
        f"DAY {day_n} OF {total_days}",
        fontsize=19,
        color=SERIES,
        weight="bold",
    )
    fig.text(
        0.945,
        0.925,
        f"{days_left} days to Black Friday",
        fontsize=19,
        color=MUTED,
        ha="right",
    )

    # Hero figure. One number, stated plainly, with the sample under it.
    fig.text(
        0.055,
        0.745,
        f"{latest['pct']:.1f}%",
        fontsize=96,
        color=INK,
        weight="bold",
        va="center",
    )
    fig.text(
        0.055,
        0.615,
        "of products are showing a struck-through “was” price right now",
        fontsize=26,
        color=INK_2,
    )
    sample = f"{int(latest['flagged']):,} of {int(latest['variants']):,} variants"
    if stores:
        sample += f" across {stores} Shopify stores"
    fig.text(0.055, 0.553, sample, fontsize=18, color=MUTED)

    # The trend. One series, so no legend: the subtitle names it.
    ax = fig.add_axes([0.055, 0.145, 0.89, 0.335])
    ax.set_facecolor(SURFACE)

    x = [(d - start).days + 1 for d in daily["date"]]
    y = daily["pct"].tolist()

    ax.plot(x, y, color=SERIES, linewidth=2.0, solid_capstyle="round", zorder=3)
    if len(x) > 1:
        ax.fill_between(x, y, min(y) - 1, color=SERIES, alpha=0.07, zorder=2)

    # Only the current point is marked and labelled. A number on every point is
    # noise, and the reader is being told one thing.
    ax.plot(
        [x[-1]],
        [y[-1]],
        "o",
        markersize=9,
        color=SERIES,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
    )

    ax.set_xlim(0.5, total_days + 0.5)
    pad = max(1.5, (max(y) - min(y)) * 0.35) if len(y) > 1 else 2.0
    ax.set_ylim(max(0, min(y) - pad), max(y) + pad)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=14, length=0)

    ticks = [1] + list(range(7, total_days + 1, 7))
    if total_days not in ticks:
        ticks.append(total_days)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"d{t}" for t in ticks[:-1]] + ["BF"])

    # Black Friday as a reference line, not a series.
    ax.axvline(total_days, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)

    fig.text(
        0.055,
        0.055,
        "Shopify Honesty Index  ·  measuring every Black Friday discount against "
        "that product's own price history",
        fontsize=15,
        color=MUTED,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches=None)
    plt.close(fig)

    return {
        "day": int(day_n),
        "days_left": int(days_left),
        "pct": round(float(latest["pct"]), 1),
        "flagged": int(latest["flagged"]),
        "variants": int(latest["variants"]),
        "path": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/snapshots")
    ap.add_argument("--out", default="posts")
    ap.add_argument("--brands", default="brands.csv")
    args = ap.parse_args()

    daily = load_daily(Path(args.data))
    start, total_days = window(daily)
    day_n = (daily.iloc[-1]["date"] - start).days + 1
    info = render(
        daily,
        Path(args.out) / f"day-{day_n:02d}.png",
        store_count(Path(args.brands)),
    )

    print(f"wrote {info['path']}")
    print(f"\n  day {info['day']} of {total_days}, {info['days_left']} to Black Friday")
    print(f"  {info['pct']}% of variants showing a was-price")
    print(f"  {info['flagged']:,} of {info['variants']:,} variants")

    if len(daily) > 1:
        change = daily["pct"].iloc[-1] - daily["pct"].iloc[-2]
        print(f"  {change:+.2f} points vs yesterday")
    print(f"\n  snapshots available: {len(daily)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
