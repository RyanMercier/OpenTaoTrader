"""Portfolio tracking, free TAO, open positions, trade log, hotkey cooldowns.

Every buy and sell goes through the AMM math here. Callers don't compute
slippage on their own.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from .amm import buy_alpha, sell_alpha
from .config import TradingConfig
from .models import (
    Direction,
    PortfolioState,
    Position,
    Signal,
    Snapshot,
    StrategyName,
)


class PortfolioTracker:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.free_tao: float = config.initial_capital_tao
        self.positions: dict[int, Position] = {}
        self.trades: list = []  # list[Trade]
        self.value_history: list[PortfolioState] = []
        self.peak_value: float = config.initial_capital_tao
        self.hotkey_cooldowns: dict[int, int] = {
            i: 0 for i in range(config.num_hotkeys)
        }

    # ---- hotkey management ---------------------------------------------

    def get_available_hotkey(self, current_block: int) -> Optional[int]:
        for hk_id, last_block in self.hotkey_cooldowns.items():
            if current_block >= last_block + self.config.blocks_per_cooldown:
                return hk_id
        return None

    def _mark_hotkey_used(self, hk_id: int, block: int) -> None:
        self.hotkey_cooldowns[hk_id] = block

    # ---- trade execution ------------------------------------------------

    def execute_buy(
        self,
        signal: Signal,
        tao_amount: float,
        snapshot: Snapshot,
        hotkey_id: int,
    ):
        from .models import Trade  # local import to avoid cycle

        if tao_amount <= 0 or tao_amount > self.free_tao + 1e-9:
            return None
        result = buy_alpha(tao_amount, snapshot.tao_in, snapshot.alpha_in)
        alpha_received = result["alpha_received"]
        if alpha_received <= 0:
            return None

        trade = Trade(
            id=str(uuid.uuid4()),
            timestamp=snapshot.timestamp,
            block=snapshot.block,
            netuid=signal.netuid,
            direction=Direction.BUY,
            strategy=signal.strategy,
            tao_amount=tao_amount,
            alpha_amount=alpha_received,
            spot_price=result["spot_price"],
            effective_price=result["effective_price"],
            slippage_pct=result["slippage_pct"],
            signal_strength=signal.strength,
            hotkey_id=hotkey_id,
        )

        self.free_tao -= tao_amount
        self.positions[signal.netuid] = Position(
            netuid=signal.netuid,
            entry_time=snapshot.timestamp,
            entry_block=snapshot.block,
            entry_price=result["effective_price"],
            alpha_amount=alpha_received,
            tao_invested=tao_amount,
            strategy=signal.strategy,
            hotkey_id=hotkey_id,
        )
        self.trades.append(trade)
        self._mark_hotkey_used(hotkey_id, snapshot.block)
        return trade

    def execute_sell(
        self,
        netuid: int,
        snapshot: Snapshot,
        reason: str,
        strategy: StrategyName,
    ):
        from .models import Trade

        position = self.positions.get(netuid)
        if position is None:
            return None
        result = sell_alpha(position.alpha_amount, snapshot.tao_in, snapshot.alpha_in)
        tao_received = result["tao_received"]

        pnl_tao = tao_received - position.tao_invested
        pnl_pct = pnl_tao / position.tao_invested if position.tao_invested > 0 else 0.0
        hold_hours = position.hold_duration_hours(snapshot.timestamp)

        trade = Trade(
            id=str(uuid.uuid4()),
            timestamp=snapshot.timestamp,
            block=snapshot.block,
            netuid=netuid,
            direction=Direction.SELL,
            strategy=strategy,
            tao_amount=tao_received,
            alpha_amount=position.alpha_amount,
            spot_price=result["spot_price"],
            effective_price=result["effective_price"],
            slippage_pct=result["slippage_pct"],
            signal_strength=1.0,
            hotkey_id=position.hotkey_id,
            entry_price=position.entry_price,
            pnl_tao=pnl_tao,
            pnl_pct=pnl_pct,
            hold_duration_hours=hold_hours,
            entry_strategy=position.strategy,
        )

        self.free_tao += tao_received
        del self.positions[netuid]
        self.trades.append(trade)
        self._mark_hotkey_used(position.hotkey_id, snapshot.block)
        return trade

    # ---- state snapshots ------------------------------------------------

    def get_state(
        self,
        timestamp: datetime,
        current_snapshots: dict[int, Snapshot],
    ) -> PortfolioState:
        position_value = 0.0
        for netuid, pos in self.positions.items():
            snap = current_snapshots.get(netuid)
            if snap is None or snap.tao_in <= 0 or snap.alpha_in <= 0:
                # Fall back to tao_invested if we can't price it
                position_value += pos.tao_invested
            else:
                position_value += pos.current_value_tao(snap.tao_in, snap.alpha_in)

        total_value = self.free_tao + position_value
        total_pnl_tao = total_value - self.config.initial_capital_tao
        total_pnl_pct = total_pnl_tao / self.config.initial_capital_tao if self.config.initial_capital_tao > 0 else 0.0

        if total_value > self.peak_value:
            self.peak_value = total_value
        drawdown_pct = 0.0
        if self.peak_value > 0:
            drawdown_pct = (total_value - self.peak_value) / self.peak_value

        state = PortfolioState(
            timestamp=timestamp,
            free_tao=self.free_tao,
            positions=dict(self.positions),
            total_value_tao=total_value,
            total_pnl_tao=total_pnl_tao,
            total_pnl_pct=total_pnl_pct,
            num_trades=len(self.trades),
            peak_value_tao=self.peak_value,
            drawdown_pct=drawdown_pct,
        )
        self.value_history.append(state)
        return state

    # ---- serialization --------------------------------------------------

    def to_json(self) -> dict:
        return {
            "free_tao": self.free_tao,
            "peak_value": self.peak_value,
            "hotkey_cooldowns": self.hotkey_cooldowns,
            "positions": {
                str(n): p.to_dict() for n, p in self.positions.items()
            },
            "trades": [t.to_dict() for t in self.trades],
        }

    @classmethod
    def from_json(cls, data: dict, config: TradingConfig) -> "PortfolioTracker":
        self = cls(config)
        self.free_tao = data.get("free_tao", config.initial_capital_tao)
        self.peak_value = data.get("peak_value", config.initial_capital_tao)
        self.hotkey_cooldowns = {int(k): int(v) for k, v in data.get("hotkey_cooldowns", {}).items()}
        self.positions = {}
        for netuid_str, pd in data.get("positions", {}).items():
            self.positions[int(netuid_str)] = Position(
                netuid=pd["netuid"],
                entry_time=datetime.fromisoformat(pd["entry_time"]),
                entry_block=pd["entry_block"],
                entry_price=pd["entry_price"],
                alpha_amount=pd["alpha_amount"],
                tao_invested=pd["tao_invested"],
                strategy=StrategyName(pd["strategy"]),
                hotkey_id=pd["hotkey_id"],
            )
        # Trades are re-hydrated as dicts; full round-trip isn't needed for
        # paper trading continuity but we preserve the log.
        from .models import Trade
        self.trades = []
        for td in data.get("trades", []):
            try:
                self.trades.append(
                    Trade(
                        id=td["id"],
                        timestamp=datetime.fromisoformat(td["timestamp"]),
                        block=td["block"],
                        netuid=td["netuid"],
                        direction=Direction(td["direction"]),
                        strategy=StrategyName(td["strategy"]),
                        tao_amount=td["tao_amount"],
                        alpha_amount=td["alpha_amount"],
                        spot_price=td["spot_price"],
                        effective_price=td["effective_price"],
                        slippage_pct=td["slippage_pct"],
                        signal_strength=td["signal_strength"],
                        hotkey_id=td["hotkey_id"],
                        entry_price=td.get("entry_price"),
                        pnl_tao=td.get("pnl_tao"),
                        pnl_pct=td.get("pnl_pct"),
                        hold_duration_hours=td.get("hold_duration_hours"),
                        entry_strategy=StrategyName(td["entry_strategy"]) if td.get("entry_strategy") else None,
                    )
                )
            except Exception:
                continue
        return self
