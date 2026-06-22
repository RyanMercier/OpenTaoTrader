"""Cross-Sectional Momentum + Volatility Brake (XMVB).

Classic cross-sectional momentum: rank subnets by 7d return at each tick,
long the top-K. Add a volatility crash filter: if 7d realized volatility
sits in the top quintile of the cross section, exclude that subnet (these
are the names most likely to give the gain back overnight).

This is the bread-and-butter cross-sectional momentum factor adapted to
the Bittensor subnet universe — the kind of signal that's robust across
many asset classes provided the volatility brake is in place.
"""

from __future__ import annotations

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


_TICK: dict = {"ts": None, "returns": {}, "vols": {}}


def _push(ts, netuid: int, ret: Optional[float], vol: Optional[float]) -> None:
    if ts != _TICK["ts"]:
        _TICK["ts"] = ts
        _TICK["returns"] = {}
        _TICK["vols"] = {}
    if ret is not None:
        _TICK["returns"][netuid] = ret
    if vol is not None:
        _TICK["vols"][netuid] = vol


def _rank_returns(netuid: int) -> Optional[tuple[int, int]]:
    r = _TICK["returns"]
    if netuid not in r or not r:
        return None
    ordered = sorted(r.items(), key=lambda kv: -kv[1])
    for i, (n, _) in enumerate(ordered):
        if n == netuid:
            return i + 1, len(ordered)
    return None


def _vol_pct(netuid: int) -> Optional[float]:
    """Where this subnet's vol sits in the cross section (0.0 = lowest, 1.0 = highest)."""
    v = _TICK["vols"]
    if netuid not in v or not v:
        return None
    my_vol = v[netuid]
    n = len(v)
    rank = sum(1 for x in v.values() if x < my_vol)
    return rank / max(n - 1, 1)


@register_strategy("xmvb")
class XMVBStrategy(Strategy):
    """Top-K by 7d return, excluding high-vol names."""

    def name(self) -> StrategyName:
        return StrategyName.XMVB

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        ret = features.price_momentum_7d
        vol = features.price_volatility_7d
        depth = features.pool_depth_tao
        _push(snapshot.timestamp, netuid, ret, vol)

        if ret is None or vol is None or depth is None:
            return None
        if depth <= c.xmvb_min_pool_depth:
            return None
        if ret <= 0:
            return None

        rank_info = _rank_returns(netuid)
        if rank_info is None:
            return None
        rank, total = rank_info
        if total < c.xmvb_min_universe:
            return None
        if rank > c.xmvb_top_k:
            return None
        # Volatility brake: exclude top quintile by 7d vol
        vp = _vol_pct(netuid)
        if vp is not None and vp >= c.xmvb_max_vol_pct:
            return None

        strength = 1.0 - (rank - 1) / max(c.xmvb_top_k, 1)
        strength = max(0.5, min(1.0, strength))
        reason = (
            f"XMVB entry SN{netuid}: 7d ret {ret*100:+.1f}%, rank {rank}/{total}, "
            f"vol pct {vp:.2f}"
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
        if hours_held >= c.xmvb_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"XMVB time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.xmvb_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"XMVB stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
