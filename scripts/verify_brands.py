#!/usr/bin/env python3
"""Check which domains serve a usable public /products.json feed.

brands.csv ships already verified, so this is a maintenance tool rather than a
required step: run it to re-check the list, or to vet candidates before adding
them. It writes a report alongside, and never edits brands.csv itself.

    python scripts/verify_brands.py
    python scripts/verify_brands.py --input candidates.csv --output report.csv
    python scripts/verify_brands.py --retry-failed --input report.csv --output report.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    LIMITER,
    PRODUCTS_PATH,
    TRANSIENT,
    get_json,
    make_session,
    normalise_domain,
)

FIELDS = ["domain", "brand", "category", "status", "products_seen", "has_compare_at"]


def check(row: dict, attempts: int, base_delay: float) -> dict:
    domain = normalise_domain(row.get("domain", ""))
    result = {
        "domain": domain,
        "brand": row.get("brand", ""),
        "category": row.get("category", ""),
        "status": "blank_domain" if not domain else "unknown",
        "products_seen": 0,
        "has_compare_at": "",
    }
    if not domain:
        return result

    # Stagger the start so a pool of workers does not fire in lockstep, which is
    # what draws the first throttle of a run.
    time.sleep(random.uniform(0, 1.2))

    payload, status = get_json(
        make_session(),
        f"https://{domain}{PRODUCTS_PATH}?limit=5",
        attempts=attempts,
        base_delay=base_delay,
    )

    if payload is None:
        result["status"] = status
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("products"), list):
        result["status"] = "no_products_key"
        return result

    products = payload["products"]
    if not products:
        result["status"] = "empty_catalog"
        return result

    variants = (products[0] or {}).get("variants") or []
    if not variants:
        result["status"] = "no_variants"
        return result

    # compare_at_price is what the whole study measures. A store that never
    # exposes it cannot be scored, so that shows up here rather than in November.
    result["has_compare_at"] = str(
        any("compare_at_price" in (v or {}) for v in variants)
    ).lower()
    result["products_seen"] = len(products)
    result["status"] = "live"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="brands.csv")
    parser.add_argument("--output", default="verification_report.csv")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--base-delay", type=float, default=4.0)
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="minimum seconds between any two requests, across all threads",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "treat --input as a previous report, keep every settled result, and "
            "re-check only the throttles and timeouts"
        ),
    )
    args = parser.parse_args()
    LIMITER.configure(args.rate)

    source = Path(args.input)
    if not source.exists():
        print(f"missing input file: {source}", file=sys.stderr)
        return 1

    with source.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("domain") or "").strip()]

    settled: list[dict] = []
    if args.retry_failed:
        if not rows or "status" not in rows[0]:
            print("--retry-failed needs a report with a status column", file=sys.stderr)
            return 1
        settled = [
            {k: r.get(k, "") for k in FIELDS}
            for r in rows
            if r.get("status") not in TRANSIENT
        ]
        rows = [r for r in rows if r.get("status") in TRANSIENT]
        if not rows:
            print("nothing to retry, every result is settled")
            return 0
        print(f"retrying {len(rows)} transient failures, keeping {len(settled)} settled")
    else:
        print(f"checking {len(rows)} domains")

    print(
        f"workers={args.workers} attempts={args.attempts} "
        f"base_delay={args.base_delay}s rate={args.rate}s/request\n"
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check, r, args.attempts, args.base_delay) for r in rows]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            flag = "ok " if result["status"] == "live" else "   "
            print(f"{flag}[{i:>4}/{len(rows)}] {result['domain']:<34} {result['status']}")

    if settled:
        recovered = sum(1 for r in results if r["status"] == "live")
        print(f"\n  recovered {recovered} store(s) the earlier pass missed")
        results.extend(settled)

    results.sort(key=lambda r: (r["status"] != "live", r["category"], r["domain"]))

    destination = Path(args.output)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    live = [r for r in results if r["status"] == "live"]

    if LIMITER.throttle_events:
        print(
            f"\n  rate limiter absorbed {LIMITER.throttle_events} throttle event(s), "
            f"interval ended at {LIMITER.interval:.2f}s"
        )

    print("\n" + "=" * 54)
    for status, count in Counter(r["status"] for r in results).most_common():
        print(f"  {status:<26} {count:>4}")
    print("=" * 54)
    print(f"  LIVE STORES                {len(live):>4}")
    print(f"  wrote {destination}")

    print("\n  live by category:")
    for category, count in Counter(r["category"] for r in live).most_common():
        print(f"    {category:<20} {count:>3}")

    transient = [r for r in results if r["status"] in TRANSIENT]
    if transient:
        print(
            f"\n  {len(transient)} failed for transient reasons. Re-check with:\n"
            f"    python scripts/verify_brands.py --retry-failed "
            f"--input {destination} --output {destination} --rate 1.2"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
