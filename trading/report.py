"""Console reports + JSON serialization for backtest results."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .models import Direction, Snapshot
from .portfolio import PortfolioTracker

if TYPE_CHECKING:
    from .backtester import BacktestResult


def print_backtest_report(result: "BacktestResult") -> None:
    print()
    print("=" * 72)
    print("BITTENSOR SUBNET TRADING, BACKTEST REPORT")
    print("=" * 72)
    print(f"Time range:      {result.start_time}  ->  {result.end_time}")
    print(f"Initial capital: {result.initial_capital:.2f} TAO")
    print(f"Strategies:      {', '.join(result.strategies_used)}")
    if result.regime_filter:
        print(f"Regime filter:   {result.regime_filter}")
    print()
    print("-- Performance ---------------------------------------------------")
    print(f"  Final value:        {result.final_value:.3f} TAO  ({result.total_return_pct*100:+.2f}%)")
    print(f"  Total return (TAO): {result.total_return_tao:+.3f}")
    print(f"  Annualized:         {result.annualized_return_pct*100:+.2f}%")
    print(f"  Max drawdown:       {result.max_drawdown_pct*100:.2f}%  ({result.max_drawdown_tao:.3f} TAO)")
    print(f"  Sharpe:             {result.sharpe_ratio:.2f}")
    print(f"  Sortino:            {result.sortino_ratio:.2f}")
    print(f"  Calmar:             {result.calmar_ratio:.2f}")
    print()
    print("-- Trade stats ---------------------------------------------------")
    print(f"  Total trades:       {result.total_trades}  (buys={result.buy_trades}, sells={result.sell_trades})")
    print(f"  Win rate:           {result.win_rate*100:.1f}%  ({result.winning_trades}W/{result.losing_trades}L)")
    print(f"  Avg win:            {result.avg_win_pct*100:+.2f}%")
    print(f"  Avg loss:           {result.avg_loss_pct*100:+.2f}%")
    print(f"  Best trade:         {result.best_trade_pct*100:+.2f}%")
    print(f"  Worst trade:        {result.worst_trade_pct*100:+.2f}%")
    pf = result.profit_factor
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"  Profit factor:      {pf_str}")
    print(f"  Avg hold:           {result.avg_hold_hours:.1f}h  (median {result.median_hold_hours:.1f}h)")
    print()
    if result.strategy_stats:
        print("-- By strategy ---------------------------------------------------")
        header = f"  {'strategy':<18} {'buys':>6} {'sells':>6} {'wins':>5} {'winrate':>8} {'avg%':>8} {'pnl':>10}"
        print(header)
        for name, st in sorted(result.strategy_stats.items(), key=lambda kv: -kv[1].get("total_pnl_tao", 0)):
            print(
                f"  {name:<18} {st['buys']:>6} {st['sells']:>6} {st['wins']:>5} "
                f"{st['win_rate']*100:>7.1f}% {st['avg_return_pct']*100:>+7.2f}% "
                f"{st['total_pnl_tao']:>+10.3f}"
            )
        print()
    print("-- Slippage analysis --------------------------------------------")
    print(f"  Avg entry slippage: {result.avg_entry_slippage_pct*100:.3f}%")
    print(f"  Avg exit slippage:  {result.avg_exit_slippage_pct*100:.3f}%")
    print(f"  Total slip cost:    {result.total_slippage_cost_tao:.4f} TAO")
    print(f"  Max slip trade:     {result.max_slippage_trade_pct*100:.3f}%")
    print()
    if result.trades:
        sells = [t for t in result.trades if t.direction == Direction.SELL]
        sells_sorted = sorted(sells, key=lambda t: -(t.pnl_pct or 0))
        best = sells_sorted[:5]
        worst = sells_sorted[-5:][::-1]
        print("-- Top 5 best trades --------------------------------------------")
        for t in best:
            print(f"  SN{t.netuid:<4} {t.strategy.value:<18} pnl {t.pnl_pct*100:+.2f}%  held {t.hold_duration_hours:.1f}h")
        print("-- Top 5 worst trades -------------------------------------------")
        for t in worst:
            print(f"  SN{t.netuid:<4} {t.strategy.value:<18} pnl {t.pnl_pct*100:+.2f}%  held {t.hold_duration_hours:.1f}h")
        print()
    if result.monthly_returns:
        print("-- Monthly returns ----------------------------------------------")
        for month in sorted(result.monthly_returns.keys()):
            r = result.monthly_returns[month]
            print(f"  {month}:  {r*100:+.2f}%")
        print()
    if result.blocked_trades:
        print("-- Blocked trades -----------------------------------------------")
        print(f"  Total blocked: {result.blocked_trades}")
        for reason, count in sorted(result.blocked_reasons.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {count:>5}  {reason}")
        print()
    print("=" * 72)


def save_backtest_json(result: "BacktestResult", path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_json(), f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def print_paper_status(
    portfolio: PortfolioTracker,
    current_snapshots: dict[int, Snapshot],
) -> None:
    now = datetime.now()
    state = portfolio.get_state(now, current_snapshots)
    print()
    print("PAPER TRADING STATUS")
    print("-" * 72)
    print(f"  Total value:   {state.total_value_tao:.3f} TAO")
    print(f"  Free TAO:      {state.free_tao:.3f}")
    print(f"  P&L:           {state.total_pnl_tao:+.3f} TAO  ({state.total_pnl_pct*100:+.2f}%)")
    print(f"  Peak:          {state.peak_value_tao:.3f}")
    print(f"  Drawdown:      {state.drawdown_pct*100:+.2f}%")
    print(f"  Trades:        {state.num_trades}")
    print(f"  Open:          {len(state.positions)}")
    print()
    if state.positions:
        print("  OPEN POSITIONS")
        print(f"  {'SN':<5} {'Strategy':<18} {'Entry':>10} {'Curr':>10} {'Alpha':>12} {'P&L%':>8} {'Held(h)':>9}")
        for netuid, pos in state.positions.items():
            snap = current_snapshots.get(netuid)
            if snap and snap.tao_in > 0 and snap.alpha_in > 0:
                cur_price = snap.tao_in / snap.alpha_in
                pnl_pct = pos.unrealized_pnl_pct(snap.tao_in, snap.alpha_in) * 100
            else:
                cur_price = 0.0
                pnl_pct = 0.0
            held = pos.hold_duration_hours(now)
            print(
                f"  {netuid:<5} {pos.strategy.value:<18} "
                f"{pos.entry_price:>10.6f} {cur_price:>10.6f} {pos.alpha_amount:>12.2f} "
                f"{pnl_pct:>+7.2f}% {held:>9.1f}"
            )
        print()
