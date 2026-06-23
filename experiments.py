"""Parallel experiments to push LAM+RCB ensemble toward super-profitable.

Approach:
  - Each experiment is a small Python config override on top of the
    winning LAM+RCB ensemble (RCB hold=18h tuned).
  - Run them in a ProcessPoolExecutor for parallelism on this 8-core box.
  - Collect topline metrics into one comparison table at the end.

Not committed to the repo — this is one-shot experiment scaffolding.
"""

from __future__ import annotations

import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from trading.backtester import Backtester
from trading.config import TradingConfig


DB_PATH = "/home/ryan/bittensor/OpenTaoAPI/TaoOpenAPI/data/opentao.db"
START = "2026-04-15"
END = "2026-06-21"
BASE_CAP = 100.0
BASE_STRATEGIES = ["lam", "rcb"]

# Subnets that lost the most money in the baseline ensemble — pruning candidates.
LOSER_SUBNETS = [38, 24, 74, 59, 31, 21, 79, 54, 80, 27]


def run_one(name: str, overrides: dict, strategies=None, capital=None) -> dict:
    """Spawn one backtest with config overrides; return topline metrics."""
    cfg = TradingConfig(db_path=DB_PATH)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if capital is not None:
        cfg.initial_capital_tao = capital
    bt = Backtester(cfg)
    t0 = time.time()
    res = bt.run(start=START, end=END, strategies=strategies or BASE_STRATEGIES)
    elapsed = time.time() - t0
    real_ret_pct = (res.final_value / res.initial_capital - 1) * 100
    from datetime import datetime
    s = datetime.fromisoformat(res.start_time.replace("Z","+00:00"))
    e = datetime.fromisoformat(res.end_time.replace("Z","+00:00"))
    days = (e - s).total_seconds() / 86400
    annual = ((1 + real_ret_pct/100) ** (365/days) - 1) * 100 if days > 0 else float("nan")
    return {
        "name": name,
        "return_pct": real_ret_pct,
        "annual_pct": annual,
        "sharpe": res.sharpe_ratio,
        "sortino": res.sortino_ratio,
        "calmar": res.calmar_ratio,
        "max_dd_pct": res.max_drawdown_pct * 100,
        "trades": res.buy_trades + res.sell_trades,
        "pf": res.profit_factor,
        "final": res.final_value,
        "capital": res.initial_capital,
        "elapsed_s": elapsed,
    }


# Experiment configurations.
EXPERIMENTS = [
    # Baseline (reference)
    ("baseline (no tweaks)", {"rcb_hold_hours": 18.0}, None, None),

    # === Blacklist losers ===
    ("blacklist bottom-10 losers", {"rcb_hold_hours": 18.0, "exclude_netuids": [0] + LOSER_SUBNETS}, None, None),

    # === max_positions sweep ===
    ("max_positions=3", {"rcb_hold_hours": 18.0, "max_positions": 3}, None, None),
    ("max_positions=5", {"rcb_hold_hours": 18.0, "max_positions": 5}, None, None),
    ("max_positions=7", {"rcb_hold_hours": 18.0, "max_positions": 7}, None, None),
    ("max_positions=10 (default)", {"rcb_hold_hours": 18.0, "max_positions": 10}, None, None),
    ("max_positions=15", {"rcb_hold_hours": 18.0, "max_positions": 15}, None, None),

    # === concentration cap sweep ===
    ("max_single_pct=0.20 (default)", {"rcb_hold_hours": 18.0, "max_single_position_pct": 0.20}, None, None),
    ("max_single_pct=0.30", {"rcb_hold_hours": 18.0, "max_single_position_pct": 0.30}, None, None),
    ("max_single_pct=0.40", {"rcb_hold_hours": 18.0, "max_single_position_pct": 0.40}, None, None),
    ("max_single_pct=0.50", {"rcb_hold_hours": 18.0, "max_single_position_pct": 0.50}, None, None),

    # === capital scale ===
    ("capital=500", {"rcb_hold_hours": 18.0}, None, 500.0),
    ("capital=1000", {"rcb_hold_hours": 18.0}, None, 1000.0),

    # === aggressive combos: blacklist + concentrated + smaller position count ===
    ("blacklist + max_pos=5 + conc=0.30",
     {"rcb_hold_hours": 18.0, "exclude_netuids": [0] + LOSER_SUBNETS,
      "max_positions": 5, "max_single_position_pct": 0.30}, None, None),
    ("blacklist + max_pos=3 + conc=0.40",
     {"rcb_hold_hours": 18.0, "exclude_netuids": [0] + LOSER_SUBNETS,
      "max_positions": 3, "max_single_position_pct": 0.40}, None, None),
    ("blacklist + max_pos=5 + conc=0.40",
     {"rcb_hold_hours": 18.0, "exclude_netuids": [0] + LOSER_SUBNETS,
      "max_positions": 5, "max_single_position_pct": 0.40}, None, None),
]


def main():
    print(f"Running {len(EXPERIMENTS)} experiments in parallel...")
    results = []
    # spawn + max_tasks_per_child=1 isolates each experiment to its own fresh
    # process so module-level caches in cross-sectional strategies (xstmc, eyc,
    # xmvb, pmr) can't leak between experiments and corrupt their signals.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {}
        for name, overrides, strategies, capital in EXPERIMENTS:
            fut = pool.submit(run_one, name, overrides, strategies, capital)
            futures[fut] = name
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                print(f"  ✓ {res['name']}: ret={res['return_pct']:+.2f}% "
                      f"sortino={res['sortino']:+.2f} ({res['elapsed_s']:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"  ✗ {futures[fut]}: {e}", flush=True)

    # Sort by sortino (primary) × calmar (secondary)
    results.sort(key=lambda r: (-r["sortino"], -r["calmar"]))

    print("\n" + "=" * 110)
    print(f'{"experiment":<42} {"ret%":>7} {"annual%":>9} {"sortino":>8} {"calmar":>7} {"DD%":>7} {"trades":>6} {"PF":>5}')
    print("-" * 110)
    for r in results:
        print(f'{r["name"]:<42} {r["return_pct"]:>+7.2f} {r["annual_pct"]:>+9.1f} '
              f'{r["sortino"]:>+8.2f} {r["calmar"]:>+7.2f} {r["max_dd_pct"]:>+7.2f} '
              f'{r["trades"]:>6} {r["pf"]:>5.2f}')

    Path("results/experiments_round1.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: results/experiments_round1.json")


if __name__ == "__main__":
    main()
