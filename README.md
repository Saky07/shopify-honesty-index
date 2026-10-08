# Shopify Honesty Index

**Are independent DTC brands more honest about Black Friday discounts than big-box retail?**

Published studies keep finding that a large share of Black Friday discounts at major
retailers are not discounts at all. WalletHub put it at 36% of online items offering no
saving against their pre-Black-Friday price. PriceRunner and PriceSpy report similar
patterns in the UK and Nordics.

Every one of those studies looks at large retailers, because that is who price comparison
sites track. Nobody watches the independent DTC brands, so nobody knows whether they
behave the same way. This project collects the evidence to find out.

**243 verified Shopify storefronts**, polled daily from **7 October 2026** to **Black
Friday, 27 November 2026**. Three segments are measured by one identical method, so the
interesting result may be the gap between them rather than any single figure:

| Segment | Stores |
|---|---|
| Established DTC (apparel, home, food and beverage, beauty, and others) | 206 |
| Creator and celebrity brands | 37 |
| Big-box retail | published figures above, for comparison |

The answer is not known in advance, and the dataset is published either way.

---

## Method

Every Shopify storefront serves its catalogue at a public `/products.json` endpoint. That
feed carries, per variant, the current `price`, the `compare_at_price` that renders as the
struck-through "was" price, and an availability flag. Polling it daily builds a price
history that no single snapshot can show.

Each variant is judged **against its own history**, never against other brands:

| Term | Definition |
|---|---|
| `claimed` | `(compare_at_price - price) / compare_at_price` on the sale date |
| `baseline` | median price over the collection window, excluding the 14 days before the sale |
| `real` | `(baseline - price) / baseline` on the sale date |

The 14-day blackout is the part that matters. Prices raised shortly before a sale would
otherwise lift the baseline and launder a fake discount into a real-looking one.

A variant advertising a discount is then classified:

- **phantom** — `real` at or below 0.5%, so the price is not below its own normal price
- **overstated** — real saving is less than half the advertised saving
- **honest** — real saving is at least half the advertised saving

Two behaviours are reported separately, because they are different things:

- **permanent sale** — `compare_at_price` was set on 90%+ of observed days, so the
  "regular price" being discounted from never actually applied
- **pre-inflation** — the price rose above baseline during the blackout window before
  being cut for the sale

The headline figure is the **phantom rate among variants advertising a discount**, chosen
because it is directly comparable to the published big-retail numbers above.

### Why 0.5% and not zero

An earlier version tested `real <= 0` and misfiled 8% of known-fake discounts as real
savings, because retail prices wobble slightly for reasons unrelated to discounting. The
tolerance band is set by `--phantom-tolerance` and reported in every result file.

---

## Results

Published after 27 November 2026.

---

## Running it

```bash
pip install -r requirements.txt

# One snapshot. The GitHub Action does this daily and commits the result.
python scripts/collect.py

# Render the day's post image from whatever snapshots exist.
python scripts/daily_card.py

# Score the sale once the history is there.
python scripts/analyze.py --sale-date 2026-11-27
```

`brands.csv` ships verified: all 243 stores served a usable feed when the list was
frozen. To re-check it, or to vet candidates before adding them, run
`scripts/verify_brands.py`, which writes a report and never edits `brands.csv`.

### Verifying the analysis without waiting for November

`make_test_data.py` generates 52 days of synthetic snapshots containing six planted
behaviours with known ground truth. `check_test.py` confirms the analyzer recovers them.

```bash
python scripts/make_test_data.py --out data/test
python scripts/analyze.py --data data/test --sale-date 2026-11-27 --out results/test
python scripts/check_test.py
```

Current state: **100% verdict accuracy across 1,200 planted variants, 0 false positives on
both behaviour flags.**

---

## Data layout

```
data/snapshots/<date>/
    prices.parquet        one row per variant per day
    new_products.parquet  catalogue rows for products first seen that day
    run_log.csv           per-store outcome, including failures
    summary.json          row counts, timing, and throttling
```

The catalogue is split out deliberately. Titles, tags and vendors are written once, on the
day a product first appears, rather than repeated in all 52 snapshots.

Failures are recorded rather than dropped. A store missing from a day's data should always
be explainable from that day's `run_log.csv`.

---

## Collection ethics

- Only the public `/products.json` endpoint is read. No authentication, no accounts, no
  customer data, no attempt to reach anything a storefront does not serve openly.
- Requests are rate limited, backed off on 429 and 5xx, and capped per store.
- The user agent identifies the project and links back to this repository.
- Results are reported in aggregate, by category and segment. Individual small merchants
  are not singled out for criticism.
- Brand-level figures are published only for stores with at least 25 advertised variants
  (`--min-brand-variants`). Creator catalogues are frequently 20 to 40 products, where one
  variant moves the percentage several points, and a noisy number at the top of a "worst
  offenders" list would be both wrong and unfair. Smaller stores still count toward the
  segment and headline figures.
- Any store operator who would rather not be included can open an issue and will be removed
  from `brands.csv`.

## Limitations

- `/products.json` exposes an availability boolean, not inventory counts, so stock-level
  questions cannot be answered from this data.
- Only published products appear in the feed. Products hidden or unpublished are invisible.
- Some stores disable the endpoint, which is why the candidate list is overselected before
  culling. Of 357 candidates, 243 serve a usable feed; the rest are not on Shopify, have the
  endpoint disabled, or sit behind bot protection.
- Shopify rate limits by client IP across its whole storefront fleet rather than per store,
  so the collector enforces one global request interval across all threads and widens it on
  any 429. Early runs without this lost whole blocks of stores, and because the block burned
  was simply whichever went first, the losses looked random rather than like a rate problem.
- Variants that appear partway through the window have shorter baselines. The
  `--min-baseline-days` floor excludes those with too little history to judge.
- A discount can be genuine against a competitor's price while being phantom against the
  brand's own history. This project measures the second thing, and says so.

## License

MIT for the code. Collected data is published under CC BY 4.0.
