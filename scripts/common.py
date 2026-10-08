"""HTTP plumbing shared by the collector and the verifier."""
from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

import requests

DEFAULT_UA = (
    "shopify-honesty-index/1.0 (research project; "
    "+https://github.com/Saky07/shopify-honesty-index)"
)

# Public, documented Shopify storefront endpoint. Read-only, no auth.
PRODUCTS_PATH = "/products.json"

TIMEOUT = (10, 25)  # connect, read
PAGE_SIZE = 250
MAX_PAGES = 40  # 40 * 250 = 10,000 products per store, a generous ceiling

# Statuses that say something about our request rate rather than about the
# store, so they are worth retrying. Everything else (404, 403, 401, non-JSON)
# is a real answer and retrying it just wastes time.
TRANSIENT = frozenset(
    {
        "http_429",
        "http_441",
        "http_500",
        "http_502",
        "http_503",
        "http_504",
        "error:ConnectionError",
        "error:ConnectTimeout",
        "error:ReadTimeout",
        "error:SSLError",
        "error:Timeout",
        "error:ChunkedEncodingError",
        "unknown",
    }
)


class RateLimiter:
    """One request rate shared by every thread in the process.

    Shopify throttles by client IP across its whole storefront fleet, not per
    store, so per-domain politeness is no protection: six threads each pausing a
    second between pages still look like six requests a second to the thing doing
    the limiting. Early runs lost whole blocks of stores to this, and since the
    block burned was simply whichever went first, the losses looked random rather
    than like a rate problem.

    The interval adapts. Every 429 doubles it, a long run of clean responses
    eases it back toward the floor. The sleep happens while holding the lock,
    which is deliberate: it serialises the *start* of every request, so the
    process as a whole never exceeds the rate.
    """

    # A 429 is a timed penalty, not a nudge. Doubling from a one-second floor
    # creeps back into the throttle again and again while the window is still
    # open, which is how a run loses sixty stores in a row; the first refusal
    # has to jump straight to a wait long enough to outlast it.
    PENALTY_FLOOR = 20.0

    def __init__(self, min_interval: float = 0.8, max_interval: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._floor = min_interval
        self._ceiling = max_interval
        self._interval = min_interval
        self._last = 0.0
        self._clean_run = 0
        self.throttle_events = 0

    def configure(self, min_interval: float, max_interval: float = 120.0) -> None:
        """Reset the rate in place.

        In place, rather than by swapping in a new object, so that modules which
        imported the singleton at import time keep pointing at the live one.
        """
        with self._lock:
            self._floor = min_interval
            self._ceiling = max_interval
            self._interval = min_interval
            self._clean_run = 0
            self.throttle_events = 0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            gap = self._last + self._interval - now
            if gap > 0:
                time.sleep(gap)
                now = time.monotonic()
            self._last = now

    def penalise(self) -> None:
        with self._lock:
            self.throttle_events += 1
            self._clean_run = 0
            self._interval = min(
                max(self._interval * 2.0, self.PENALTY_FLOOR), self._ceiling
            )

    def reward(self) -> None:
        with self._lock:
            self._clean_run += 1
            if self._clean_run >= 25 and self._interval > self._floor:
                self._interval = max(self._floor, self._interval / 1.5)
                self._clean_run = 0

    @property
    def interval(self) -> float:
        return self._interval


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return fallback


LIMITER = RateLimiter(_env_float("HONESTY_INDEX_MIN_INTERVAL", 0.8))


def user_agent() -> str:
    return os.environ.get("HONESTY_INDEX_UA", DEFAULT_UA)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent(),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def normalise_domain(raw: str) -> str:
    domain = (raw or "").strip().lower()
    for prefix in ("https://", "http://"):
        if domain.startswith(prefix):
            domain = domain[len(prefix) :]
    return domain.rstrip("/")


def get_json(
    session: requests.Session,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 1.5,
) -> tuple[Any | None, str]:
    """GET a URL and parse JSON.

    Returns (payload, status). The payload is None on any failure and the status
    always carries a short machine-readable reason, so a store that drops out of
    a run is explainable from the log rather than simply missing.
    """
    status = "unknown"

    for attempt in range(attempts):
        LIMITER.wait()
        try:
            response = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        except requests.RequestException as exc:
            status = f"error:{type(exc).__name__}"
        else:
            if response.status_code == 429:
                LIMITER.penalise()
            elif response.status_code < 400:
                LIMITER.reward()

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "").lower()
                if "json" not in content_type:
                    return None, f"not_json:{content_type[:30]}"
                try:
                    return response.json(), "ok"
                except ValueError:
                    return None, "bad_json"

            if response.status_code in (429, 500, 502, 503, 504):
                status = f"http_{response.status_code}"
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 30.0))
                        continue
                    except ValueError:
                        pass
            else:
                return None, f"http_{response.status_code}"

        if attempt < attempts - 1:
            # A 429 waits considerably longer than a flaky connection: the
            # throttle window runs to tens of seconds, and a brisk retry just
            # burns an attempt against a door that is still shut.
            factor = 4.0 if status == "http_429" else 1.0
            time.sleep(base_delay * factor * (2**attempt) + random.uniform(0, 1.5))

    return None, status


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # reject NaN
