"""Strategy interface. Implementations generate entry and exit Signals from
causally-computed Features plus the latest Snapshot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..config import TradingConfig
from ..models import Features, Position, Signal, Snapshot, StrategyName


class Strategy(ABC):
    def __init__(self, config: TradingConfig):
        self.config = config

    @abstractmethod
    def name(self) -> StrategyName: ...

    @abstractmethod
    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]: ...

    @abstractmethod
    def generate_exit_signal(
        self,
        netuid: int,
        features: Features,
        snapshot: Snapshot,
        position: Position,
    ) -> Optional[Signal]: ...

    def can_run_in_regime(self, regime: str) -> bool:
        return True
