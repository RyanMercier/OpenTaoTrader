"""Pair Mean Reversion (PMR).

Computes pairwise rolling correlations (30d) across the active subnet
universe; for each subnet, finds its highest-correlated partner. When the
return spread between a subnet and its partner blows out by > pmr_entry_z
standard deviations, BUY the underperformer (we expect the spread to
revert toward the mean).

Unlike a true market-neutral pairs trade, we only take the long leg of
the spread (no shorts available in the AMM). Net result: it's a
mean-reversion signal that uses a peer to define "cheap" rather than just
the subnet's own history.

State management:
- Per-subnet rolling price ring buffers (30d window, ~1440 bars).
- A periodic correlation matrix recomputation (every pmr_recompute_bars
  ticks) picks each subnet's best partner; we cache (partner_id, …)
  per subnet.
- At each tick we compute the current spread (log-return-style) vs the
  rolling-mean spread, and fire when |z| > pmr_entry_z and the subnet
  is on the cheap side.
"""

from __future__ import annotations

from collections import deque
from math import log
from typing import Optional

from ..models import (
    Direction,
    Features,
    Position,
    Signal,
    Snapshot,
    StrategyName,
)
from . import register_strategy
from .base import Strategy


_30D_BARS = 30 * 48  # 1440 30-min bars

_PRICE_HIST: dict[int, deque] = {}
_LAST_TS_SEEN: dict = {"ts": None}

# Pair cache: netuid -> partner_netuid. Recomputed periodically.
_PARTNERS: dict[int, int] = {}
_LAST_RECOMPUTE: dict = {"ts": None, "bars_since": 0}


def _push_price(netuid: int, price: float, ts) -> None:
    dq = _PRICE_HIST.setdefault(netuid, deque(maxlen=_30D_BARS))
    dq.append(price)
    if ts != _LAST_TS_SEEN["ts"]:
        _LAST_TS_SEEN["ts"] = ts
        _LAST_RECOMPUTE["bars_since"] += 1


def _log_returns(prices: deque) -> list[float]:
    plist = list(prices)
    out = []
    for i in range(1, len(plist)):
        prev = plist[i - 1]
        cur = plist[i]
        if prev > 0 and cur > 0:
            out.append(log(cur / prev))
    return out


def _corr(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 30:
        return 0.0
    a = a[-n:]
    b = b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((x - mb) ** 2 for x in b)
    if sa <= 0 or sb <= 0:
        return 0.0
    sab = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return sab / (sa * sb) ** 0.5


def _recompute_partners(min_corr: float) -> None:
    """For each subnet, find its highest-correlated peer over 30d log returns."""
    nets = list(_PRICE_HIST.keys())
    returns = {n: _log_returns(_PRICE_HIST[n]) for n in nets}
    new_partners: dict[int, int] = {}
    for a in nets:
        best, best_c = None, min_corr
        for b in nets:
            if a == b:
                continue
            c = _corr(returns[a], returns[b])
            if c > best_c:
                best, best_c = b, c
        if best is not None:
            new_partners[a] = best
    _PARTNERS.clear()
    _PARTNERS.update(new_partners)


@register_strategy("pmr")
class PairMeanReversionStrategy(Strategy):
    """Long the cheap leg when its spread vs partner blows out."""

    def name(self) -> StrategyName:
        return StrategyName.PMR

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        price = snapshot.alpha_price_tao
        depth = features.pool_depth_tao
        if price <= 0:
            return None
        _push_price(netuid, price, snapshot.timestamp)

        if depth is None or depth <= c.pmr_min_pool_depth:
            return None

        if _LAST_RECOMPUTE["bars_since"] >= c.pmr_recompute_bars:
            _recompute_partners(c.pmr_min_corr)
            _LAST_RECOMPUTE["bars_since"] = 0

        partner = _PARTNERS.get(netuid)
        if partner is None or partner not in _PRICE_HIST:
            return None
        a_hist = _PRICE_HIST[netuid]
        b_hist = _PRICE_HIST[partner]
        if len(a_hist) < 100 or len(b_hist) < 100:
            return None
        # Spread series: log(a) - log(b) at each bar. Use last min-len bars.
        n = min(len(a_hist), len(b_hist))
        a_tail = list(a_hist)[-n:]
        b_tail = list(b_hist)[-n:]
        spreads = []
        for i in range(n):
            if a_tail[i] > 0 and b_tail[i] > 0:
                spreads.append(log(a_tail[i]) - log(b_tail[i]))
        if len(spreads) < 50:
            return None
        cur = spreads[-1]
        history = spreads[:-1]
        mean = sum(history) / len(history)
        var = sum((s - mean) ** 2 for s in history) / len(history)
        std = var ** 0.5
        if std <= 0:
            return None
        z = (cur - mean) / std
        # Long A only if A is CHEAP (z < -pmr_entry_z) — long the underperformer.
        if z > -c.pmr_entry_z:
            return None

        strength = min(max(abs(z) / 4.0, 0.5), 1.0)
        reason = (
            f"PMR entry SN{netuid} vs SN{partner}: spread z={z:.2f}, "
            f"depth {depth:.0f}"
        )
        return Signal(
            timestamp=snapshot.timestamp,
            netuid=netuid,
            direction=Direction.BUY,
            strategy=self.name(),
            strength=strength,
            reason=reason,
            features=features.to_dict(),
        )

    def generate_exit_signal(
        self,
        netuid: int,
        features: Features,
        snapshot: Snapshot,
        position: Position,
    ) -> Optional[Signal]:
        c = self.config
        if position.strategy != self.name():
            return None
        hours_held = position.hold_duration_hours(snapshot.timestamp)
        if hours_held >= c.pmr_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"PMR time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.pmr_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"PMR stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.pmr_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"PMR take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        # Exit when spread reverts (z crosses 0)
        partner = _PARTNERS.get(netuid)
        if partner is not None and partner in _PRICE_HIST:
            a_hist = _PRICE_HIST[netuid]
            b_hist = _PRICE_HIST[partner]
            if len(a_hist) > 50 and len(b_hist) > 50:
                n = min(len(a_hist), len(b_hist))
                a_tail = list(a_hist)[-n:]
                b_tail = list(b_hist)[-n:]
                spreads = [log(a_tail[i]) - log(b_tail[i])
                           for i in range(n) if a_tail[i] > 0 and b_tail[i] > 0]
                if len(spreads) >= 50:
                    mean = sum(spreads[:-1]) / max(len(spreads) - 1, 1)
                    if spreads[-1] >= mean:
                        return Signal(
                            timestamp=snapshot.timestamp,
                            netuid=netuid,
                            direction=Direction.SELL,
                            strategy=self.name(),
                            strength=1.0,
                            reason=f"PMR mean-revert exit SN{netuid}: spread back to mean",
                            features=features.to_dict(),
                        )
        return None
