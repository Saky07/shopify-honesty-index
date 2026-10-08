#!/usr/bin/env python3
"""Daily price snapshot of every store in brands.csv.

Writes four files per run under data/snapshots/<date>/

    prices.parquet        one row per variant per day, the facts and nothing else
    new_products.parquet  catalogue rows for products seen for the first time
    run_log.csv           per-store outcome, so a failure is recorded not silent
    summary.json          counts, timing, and how hard we got throttled

Splitting the catalogue out keeps the daily commit small: titles, tags and
vendors are written once, on the day a product first appears, instead of being
repeated in every snapshot for the rest of the collection window.

    python scripts/collect.py
    python scripts/collect.py --limit 5 --out data/test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    LIMITER,
    MAX_PAGES,
    PAGE_SIZE,
    PRODUCTS_PATH,
    get_json,
    make_session,
    normalise_domain,
    to_float,
)

LOG_FIELDS = ["domain", "status", "pages", "products", "variants"]
CATALOG_COLUMNS = [
    "first_seen",
    "domain",
    "product_id",
    "variant_id",
    "handle",
    "product_title",
    "variant_title",
    "vendor",
    "product_type",
    "tags",
    "product_created_at",
    "product_published_at",
]


def fetch_store(domain: str, delay: float) -> tuple[list[dict], list[dict], dict]:
    """Page through one store's public catalogue.

    Every failure mode returns cleanly. One bad store must never abort the run,
    because an aborted run costs a baseline day that cannot be recovered later.
    """
    session = make_session()
    prices: list[dict] = []
    catalogue: list[dict] = []
    log = {"domain": domain, "status": "ok", "pages": 0, "products": 0, "variants": 0}
    seen: set[int] = set()

    for page in range(1, MAX_PAGES + 1):
        payload, status = get_json(
            session, f"https://{domain}{PRODUCTS_PATH}?limit={PAGE_SIZE}&page={page}"
        )
        if payload is None:
            log["status"] = status if page == 1 else f"partial:{status}"
            break

        products = payload.get("products") if isinstance(payload, dict) else None
        if not isinstance(products, list) or not products:
            break
        log["pages"] = page

        for product in products:
            if not isinstance(product, dict):
                continue
            product_id = product.get("id")
            if product_id is None or product_id in seen:
                # Some stores loop back to page 1 rather than returning an empty
                # list. Without this guard the collector spins to MAX_PAGES.
                continue
            seen.add(product_id)
            log["products"] += 1

            tags = product.get("tags")
            if isinstance(tags, list):
                tags = ",".join(str(t) for t in tags)

            for variant in product.get("variants") or []:
                if not isinstance(variant, dict):
                    continue
                variant_id = variant.get("id")
                if variant_id is None:
                    continue
                log["variants"] += 1

                # product_id is deliberately absent here: variant_id already
                # identifies the row, and new_products.parquet carries the
                # variant -> product mapping once. Repeating it in every daily
                # snapshot costs about a third of the file for nothing.
                prices.append(
                    {
                        "domain": domain,
                        "variant_id": int(variant_id),
                        "price": to_float(variant.get("price")),
                        "compare_at_price": to_float(variant.get("compare_at_price")),
                        "available": bool(variant.get("available")),
                    }
                )
                catalogue.append(
                    {
                        "domain": domain,
                        "product_id": int(product_id),
                        "variant_id": int(variant_id),
                        "handle": product.get("handle") or "",
                        "product_title": product.get("title") or "",
                        "variant_title": variant.get("title") or "",
                        "vendor": product.get("vendor") or "",
                        "product_type": product.get("product_type") or "",
                        "tags": tags or "",
                        "product_created_at": product.get("created_at") or "",
                        "product_published_at": product.get("published_at") or "",
                    }
                )

        if len(products) < PAGE_SIZE:
            break
        time.sleep(delay)

    return prices, catalogue, log


def known_variants(root: Path) -> set[tuple[str, int]]:
    """Variant keys already recorded by any previous snapshot."""
    known: set[tuple[str, int]] = set()
    for path in sorted(root.glob("*/new_products.parquet")):
        if path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_parquet(path, columns=["domain", "variant_id"])
        except Exception as exc:
            print(f"  warn: could not read {path}: {exc}", file=sys.stderr)
            continue
        known.update(zip(frame["domain"], frame["variant_id"]))
    return known


def load_domains(path: Path, limit: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # brands.csv is pre-verified and carries no status column; a verification
    # report does, so honour it when present.
    if rows and "status" in rows[0]:
        rows = [r for r in rows if r.get("status") == "live"]
    domains = [normalise_domain(r["domain"]) for r in rows if r.get("domain")]
    return domains[:limit] if limit else domains


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brands", default="brands.csv")
    parser.add_argument("--out", default="data/snapshots")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.8, help="seconds between pages")
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="minimum seconds between any two requests, across all threads",
    )
    parser.add_argument("--limit", type=int, default=0, help="only N stores, for testing")
    parser.add_argument("--date", default="", help="override snapshot date (YYYY-MM-DD)")
    args = parser.parse_args()
    LIMITER.configure(args.rate)

    brands = Path(args.brands)
    if not brands.exists():
        print(f"missing {brands}", file=sys.stderr)
        return 1

    domains = load_domains(brands, args.limit)
    if not domains:
        print(f"no stores listed in {brands}", file=sys.stderr)
        return 1

    snapshot_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = Path(args.out)
    out_dir = root / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)

    already_seen = known_variants(root)
    print(f"snapshot {snapshot_date}: {len(domains)} stores")
    print(f"{len(already_seen):,} variants already in the catalogue\n")

    started = time.time()
    all_prices: list[dict] = []
    all_catalogue: list[dict] = []
    logs: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_store, d, args.delay): d for d in domains}
        for i, future in enumerate(as_completed(futures), 1):
            domain = futures[future]
            try:
                prices, catalogue, log = future.result()
            except Exception as exc:
                prices, catalogue = [], []
                log = {
                    "domain": domain,
                    "status": f"crash:{type(exc).__name__}",
                    "pages": 0,
                    "products": 0,
                    "variants": 0,
                }
            all_prices.extend(prices)
            all_catalogue.extend(catalogue)
            logs.append(log)
            flag = "ok " if log["status"] == "ok" else "!! "
            print(
                f"{flag}[{i:>4}/{len(domains)}] {domain:<34} "
                f"{log['variants']:>6} variants  {log['status']}"
            )

    prices_frame = pd.DataFrame(all_prices)
    prices_frame.insert(0, "date", snapshot_date)
    prices_frame.to_parquet(out_dir / "prices.parquet", index=False, compression="snappy")

    catalogue_frame = pd.DataFrame(all_catalogue)
    if catalogue_frame.empty:
        catalogue_frame = pd.DataFrame(columns=CATALOG_COLUMNS)
    else:
        catalogue_frame = catalogue_frame.drop_duplicates(subset=["domain", "variant_id"])
        is_new = [
            (d, v) not in already_seen
            for d, v in zip(catalogue_frame["domain"], catalogue_frame["variant_id"])
        ]
        catalogue_frame = catalogue_frame[is_new].copy()
        catalogue_frame.insert(0, "first_seen", snapshot_date)
    catalogue_frame.to_parquet(
        out_dir / "new_products.parquet", index=False, compression="snappy"
    )

    with (out_dir / "run_log.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(logs, key=lambda r: r["domain"]))

    succeeded = sum(1 for r in logs if r["status"] == "ok")
    summary = {
        "date": snapshot_date,
        "stores_attempted": len(domains),
        "stores_ok": succeeded,
        "stores_failed": len(domains) - succeeded,
        "variants": int(len(prices_frame)),
        "new_variants": int(len(catalogue_frame)),
        "seconds": round(time.time() - started, 1),
        "throttle_events": LIMITER.throttle_events,
        "final_rate_interval": round(LIMITER.interval, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 54)
    for key, value in summary.items():
        print(f"  {key:<22} {value}")
    print("=" * 54)

    if succeeded < len(domains) * 0.7:
        print("\n  WARNING: over 30% of stores failed. Check run_log.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
