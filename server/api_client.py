"""HTTP+SSE client for OpenTaoAPI.

The trader is a downstream consumer. We need three things from the API:

  - Subnet snapshot history at startup (one batched fetch per netuid).
  - Live snapshot updates over Server-Sent Events.
  - The pool-weighted buy-and-hold benchmark series for stats.

Everything else (current pool reserves for live pre-flight, coldkey
balance, stake info for before/after deltas) the live runner gets
straight from the chain via its own AsyncSubtensor, since latency
matters and the API would just be a thin pass-through.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OpenTaoAPIClient:
    """Async HTTP client. SSE consumption runs on a background task that
    folds new snapshots into ``snapshot_buffer`` in real time."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        # netuid -> list of raw snapshot dicts, oldest first.
        self.snapshot_buffer: dict[int, list[dict]] = {}
        self._buffer_lock = asyncio.Lock()
        self._sse_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._sse_events = 0
        self._sse_last_event_ts: Optional[datetime] = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def shutdown(self) -> None:
        self._stop.set()
        if self._sse_task is not None:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- snapshot seeding ---

    async def seed_snapshots(self, hours: int = 720) -> int:
        """Pull ``hours`` of history for every active subnet, bucketed by
        netuid into ``snapshot_buffer``. Returns the number of subnets
        seeded successfully."""
        if self._client is None:
            raise RuntimeError("Call startup() before seeding")

        try:
            r = await self._client.get(f"{self.base_url}/api/v1/subnets")
            r.raise_for_status()
            listing = r.json()
        except Exception:
            logger.exception("Failed to enumerate subnets from %s", self.base_url)
            return 0

        items = listing.get("subnets", []) if isinstance(listing, dict) else (
            listing if isinstance(listing, list) else []
        )

        seeded = 0
        async with self._buffer_lock:
            for item in items:
                try:
                    n = int(item["netuid"])
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    # Explicit limit=5000 (the API's max) — without it the
                    # API defaults to 500 rows AND returns the OLDEST 500
                    # within the window, which silently feeds the trader
                    # stale snapshots whenever hours covers > ~10 days at
                    # the default 30-min cadence.
                    r = await self._client.get(
                        f"{self.base_url}/api/v1/history/{n}/snapshots",
                        params={"hours": hours, "limit": 5000},
                    )
                    r.raise_for_status()
                    rows = r.json()
                except Exception:
                    continue
                if rows:
                    self.snapshot_buffer[n] = rows
                    seeded += 1
        logger.info("Seeded %d subnets from %s (%dh history)", seeded, self.base_url, hours)
        return seeded

    # --- SSE consumer ---

    async def start_sse(self) -> None:
        """Spawn the background task that consumes /api/v1/stream and
        folds events into snapshot_buffer."""
        if self._sse_task is not None and not self._sse_task.done():
            return
        self._stop.clear()
        self._sse_task = asyncio.create_task(self._sse_loop())

    async def _sse_loop(self) -> None:
        url = f"{self.base_url}/api/v1/stream"
        bar_seconds = 1800  # match the API's default poll cadence
        while not self._stop.is_set():
            try:
                async with self._client.stream(
                    "GET", url,
                    headers={"Accept": "text/event-stream"},
                    timeout=None,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if self._stop.is_set():
                            return
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        try:
                            evt = json.loads(payload)
                            n = int(evt["netuid"])
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            continue
                        await self._fold(n, evt, bar_seconds)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("SSE loop hiccup, reconnecting: %s", e)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=3)
                except asyncio.TimeoutError:
                    continue

    async def _fold(self, netuid: int, evt: dict, bar_seconds: int) -> None:
        async with self._buffer_lock:
            buf = self.snapshot_buffer.setdefault(netuid, [])
            if not buf:
                buf.append(evt)
                return
            try:
                last_ts = _ts(buf[-1].get("timestamp", ""))
                new_ts = _ts(evt.get("timestamp", ""))
                last_bar = int(last_ts.timestamp() // bar_seconds)
                new_bar = int(new_ts.timestamp() // bar_seconds)
            except Exception:
                return
            if new_bar > last_bar:
                buf.append(evt)
                if len(buf) > 3000:
                    del buf[: len(buf) - 3000]
            else:
                # Same bar: overwrite the latest sample.
                buf[-1] = evt
            self._sse_events += 1
            self._sse_last_event_ts = datetime.now(timezone.utc)

    # --- one-shot queries used by the live trader ---

    async def get_subnet_info(self, netuid: int, refresh: bool = False) -> dict | None:
        params = {"refresh": "true"} if refresh else {}
        try:
            r = await self._client.get(
                f"{self.base_url}/api/v1/subnet/{netuid}/info",
                params=params,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            logger.exception("get_subnet_info(%d) failed", netuid)
            return None

    async def is_stale(self) -> tuple[bool, str]:
        """Reflect /health's stale flag. If the API's snapshot poller is
        behind, our trading buffer is also stale."""
        try:
            r = await self._client.get(f"{self.base_url}/health", timeout=5)
            data = r.json()
        except Exception as e:
            return True, f"/health unreachable: {e}"
        poller = data.get("poller") or {}
        return bool(poller.get("stale")), str(poller)

    # --- benchmark series for stats ---

    async def compute_benchmark_series(
        self,
        timestamps: list[str],
        anchor_ts: str,
        initial_capital_tao: float,
        exclude_netuids: list[int] | None = None,
        min_pool_depth_tao: float = 50.0,
    ) -> tuple[list[float], list[int]]:
        """Pool-weighted buy-and-hold series, computed against the API's
        snapshot data. At ``anchor_ts`` we fetch every active subnet,
        build the basket (weight = tao_in share), then at each requested
        timestamp value the basket at the prevailing prices."""
        if not timestamps:
            return [], []

        exclude = set(exclude_netuids or [])
        try:
            r = await self._client.get(f"{self.base_url}/api/v1/subnets")
            r.raise_for_status()
            listing = r.json()
        except Exception:
            return [initial_capital_tao for _ in timestamps], []

        items = listing.get("subnets", []) if isinstance(listing, dict) else (
            listing if isinstance(listing, list) else []
        )

        # Collect (netuid, anchor_price, anchor_tao_in) by querying each
        # subnet's snapshot history near the anchor timestamp.
        eligible: list[tuple[int, float, float]] = []
        snapshot_series: dict[int, list[dict]] = {}
        for item in items:
            try:
                n = int(item["netuid"])
            except (KeyError, TypeError, ValueError):
                continue
            if n in exclude:
                continue
            try:
                # Pull enough history to cover from anchor to now. Explicit
                # limit=5000 to override the API's 500-row default which
                # would silently truncate to the oldest rows.
                r = await self._client.get(
                    f"{self.base_url}/api/v1/history/{n}/snapshots",
                    params={"hours": 1440, "limit": 5000},
                )
                r.raise_for_status()
                rows = r.json()
            except Exception:
                continue
            if not rows:
                continue
            anchor_row = _row_at_or_before(rows, anchor_ts)
            if anchor_row is None:
                continue
            tao_in = float(anchor_row.get("tao_in", 0.0) or 0.0)
            price = float(anchor_row.get("alpha_price_tao", 0.0) or 0.0)
            if tao_in < min_pool_depth_tao or price <= 0:
                continue
            eligible.append((n, tao_in, price))
            snapshot_series[n] = rows

        if not eligible:
            return [initial_capital_tao for _ in timestamps], []

        total_tao_in = sum(t for _, t, _ in eligible)
        holdings = {
            n: (initial_capital_tao * (t / total_tao_in)) / p
            for n, t, p in eligible
        }

        values: list[float] = []
        for ts in timestamps:
            total = 0.0
            for n, alpha_amt in holdings.items():
                row = _row_at_or_before(snapshot_series[n], ts)
                if row is None:
                    continue
                price = float(row.get("alpha_price_tao", 0.0) or 0.0)
                total += alpha_amt * price
            values.append(total)
        return values, list(holdings.keys())


def _ts(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


def _row_at_or_before(rows: list[dict], ts_iso: str) -> dict | None:
    """Linear search over sorted snapshot rows for the most recent row
    at or before the given timestamp. ``rows`` is assumed to be sorted
    ascending by timestamp (which is how the API returns them)."""
    target = _ts(ts_iso)
    chosen = None
    for r in rows:
        rt = _ts(r.get("timestamp", ""))
        if rt > target:
            break
        chosen = r
    return chosen or (rows[0] if rows else None)
