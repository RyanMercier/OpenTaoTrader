"""Pre-trade risk checks and position sizing.

Position sizing is the minimum of:
  - capital-per-slot (free_tao - reserve) / remaining_slots
  - pool-depth cap         max_position_pct_of_pool * tao_in
  - portfolio concentration max_single_position_pct * total_value_tao
  - slippage cap            max_tao_for_slippage(max_slippage_pct)
  - signal strength scaling

Whichever binds wins.
"""

from __future__ import annotations

from .amm import max_tao_for_slippage
from .config import TradingConfig
from .models import PortfolioState, Signal, Snapshot


MIN_TRADE_TAO = 0.1


class RiskManager:
    def __init__(self, config: TradingConfig):
        self.config = config

    def check_entry(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        snapshot: Snapshot,
    ) -> tuple[bool, str, float]:
        c = self.config

        # Already holding this subnet
        if signal.netuid in portfolio.positions:
            return False, f"already holding SN{signal.netuid}", 0.0

        # Max position count
        if len(portfolio.positions) >= c.max_positions:
            return False, f"at max_positions={c.max_positions}", 0.0

        # Pool liveness
        if snapshot.tao_in <= 0 or snapshot.alpha_in <= 0:
            return False, "pool empty", 0.0

        # Pool depth filter
        if snapshot.tao_in < c.min_pool_depth_tao:
            return False, f"pool depth {snapshot.tao_in:.1f} < {c.min_pool_depth_tao}", 0.0

        # Capital available after reserve and remaining slots
        reserve_tao = c.initial_capital_tao * c.reserve_pct
        free_after_reserve = max(portfolio.free_tao - reserve_tao, 0.0)
        remaining_slots = max(c.max_positions - len(portfolio.positions), 1)
        slot_cap = free_after_reserve / remaining_slots

        # Pool depth cap
        pool_cap = c.max_position_pct_of_pool * snapshot.tao_in

        # Portfolio concentration cap
        conc_cap = c.max_single_position_pct * max(portfolio.total_value_tao, 0.0)

        # Slippage cap
        slip_cap = max_tao_for_slippage(c.max_slippage_pct, snapshot.tao_in, snapshot.alpha_in)

        amount = min(slot_cap, pool_cap, conc_cap, slip_cap, free_after_reserve)
        amount *= signal.strength

        if amount < MIN_TRADE_TAO:
            return False, f"position size {amount:.4f} TAO below minimum", 0.0

        # Final safety: ensure we don't spend more than free_tao in hand
        if amount > portfolio.free_tao:
            amount = portfolio.free_tao
            if amount < MIN_TRADE_TAO:
                return False, "insufficient free TAO", 0.0

        return True, "", amount

    def check_exit(
        self,
        position,
        features,
        snapshot: Snapshot,
    ) -> tuple[bool, str]:
        hours_held = position.hold_duration_hours(snapshot.timestamp)
        if hours_held >= self.config.max_hold_hours:
            return True, f"max hold ({self.config.max_hold_hours}h) exceeded"
        return False, ""

    def compute_daily_pnl(
        self, portfolio: PortfolioState, start_of_day_value: float
    ) -> float:
        if start_of_day_value <= 0:
            return 0.0
        return (portfolio.total_value_tao - start_of_day_value) / start_of_day_value
