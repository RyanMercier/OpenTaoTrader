"""Exit-only: count consecutive drain snapshots per netuid and fire a SELL
signal once the streak crosses the threshold.

Safety net for any active position. Wire ahead of other exit conditions.
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


@register_strategy("drain_exit")
class DrainDetector(Strategy):
    """Exit-only safety: trip on N consecutive draining snapshots."""

    def __init__(self, config):
        super().__init__(config)
        self._drain_streak: dict[int, int] = {}

    def name(self) -> StrategyName:
        return StrategyName.DRAIN_EXIT

    def update(self, netuid: int, features: Features) -> None:
        """Must be called once per netuid per timestep, in chronological order,
        so the streak counter stays consistent.
        """
        sv = features.stake_velocity_24h
        if sv is None:
            # Unknown, don't reset, but don't increment either
            return
        if sv < self.config.dd_velocity_threshold:
            self._drain_streak[netuid] = self._drain_streak.get(netuid, 0) + 1
        else:
            self._drain_streak[netuid] = 0

    def current_streak(self, netuid: int) -> int:
        return self._drain_streak.get(netuid, 0)

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        # Exit-only strategy, never generates entries.
        return None

    def generate_exit_signal(
        self, netuid: int, features: Features, snapshot: Snapshot, position: Position
    ) -> Optional[Signal]:
        streak = self.current_streak(netuid)
        if streak < self.config.dd_consecutive_epochs:
            return None
        # Short-duration positions have their own intraday stops and
        # shouldn't be force-closed on a 24h drain reading. Drain is
        # informative for multi-day holds; closing very young positions
        # on it adds drag. Let the position's own exit logic handle them.
        min_hold_hours = getattr(self.config, "dd_min_hold_hours", 6.0)
        if position.hold_duration_hours(snapshot.timestamp) < min_hold_hours:
            return None
        sv = features.stake_velocity_24h if features.stake_velocity_24h is not None else 0.0
        strength = min(abs(sv) / 0.10, 1.0)
        reason = (
            f"Liquidity drain on SN{netuid}: tao_in dropping "
            f"{sv*100:.2f}%/24h for {streak} consecutive epochs"
        )
        return Signal(
            timestamp=snapshot.timestamp,
            netuid=netuid,
            direction=Direction.SELL,
            strategy=self.name(),
            strength=strength,
            reason=reason,
            features=features.to_dict(),
        )
