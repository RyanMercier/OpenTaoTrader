"""Round 2 — apples-to-apples parameter tuning on the LAM+RCB ensemble.

Round 1 told us:
  - The baseline at 100 TAO is the winner (+12.52%)
  - The strategy doesn't scale (slippage kills 500/1000 TAO runs)
  - max_positions=10 is the sweet spot
  - Concentration cap isn't the binding constraint at 100 TAO

Round 2 focuses on parameters we haven't tested apples-to-apples on the
Apr15-Jun21 window:

  - LAM entry threshold (0.005 / 0.01 / 0.02 / 0.04)
  - LAM hold hours (12 / 24 / 36 / 48)
  - LAM stop-loss / take-profit
  - RCB compression max
  - RCB stop-loss / take-profit
  - RCB hold hours (re-confirm 18 is optimal)
  - min_pool_depth_tao (universe filter; tighter = cleaner)

Also: small-capital scaling. At 100 TAO we win; what about 50, 200?
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading.backtester import Backtester
from trading.config import TradingConfig


DB_PATH = "/home/ryan/bittensor/OpenTaoAPI/TaoOpenAPI/data/opentao.db"
START = "2026-04-15"
END = "2026-06-21"
BASE_STRATEGIES = ["lam", "rcb"]


def run_one(name: str, overrides: dict, strategies=None, capital=None) -> dict:
    cfg = TradingConfig(db_path=DB_PATH)
    cfg.rcb_hold_hours = 18.0  # round-1 winner; carry through
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


EXPERIMENTS = [
    # Re-baseline so all results are head-to-head
    ("BASELINE", {}, None, None),

    # === LAM tweaks ===
    ("lam_entry=0.005", {"lam_entry_threshold": 0.005}, None, None),
    ("lam_entry=0.01",  {"lam_entry_threshold": 0.01}, None, None),
    ("lam_entry=0.04",  {"lam_entry_threshold": 0.04}, None, None),
    ("lam_hold=12h",    {"lam_hold_hours": 12.0}, None, None),
    ("lam_hold=36h",    {"lam_hold_hours": 36.0}, None, None),
    ("lam_hold=48h",    {"lam_hold_hours": 48.0}, None, None),
    ("lam_sl=-0.03",    {"lam_stop_loss_pct": -0.03}, None, None),
    ("lam_sl=-0.08",    {"lam_stop_loss_pct": -0.08}, None, None),
    ("lam_tp=0.06",     {"lam_take_profit_pct": 0.06}, None, None),
    ("lam_tp=0.15",     {"lam_take_profit_pct": 0.15}, None, None),
    ("lam_tp=0.20",     {"lam_take_profit_pct": 0.20}, None, None),

    # === RCB tweaks ===
    ("rcb_hold=12h",    {"rcb_hold_hours": 12.0}, None, None),
    ("rcb_hold=24h",    {"rcb_hold_hours": 24.0}, None, None),
    ("rcb_hold=36h",    {"rcb_hold_hours": 36.0}, None, None),
    ("rcb_comp=0.20",   {"rcb_compression_max": 0.20}, None, None),
    ("rcb_comp=0.40",   {"rcb_compression_max": 0.40}, None, None),
    ("rcb_tp=0.05",     {"rcb_take_profit_pct": 0.05}, None, None),
    ("rcb_tp=0.12",     {"rcb_take_profit_pct": 0.12}, None, None),
    ("rcb_sl=-0.02",    {"rcb_stop_loss_pct": -0.02}, None, None),
    ("rcb_sl=-0.06",    {"rcb_stop_loss_pct": -0.06}, None, None),

    # === Universe filter (pool depth) ===
    ("min_depth=500",   {"min_pool_depth_tao": 500.0}, None, None),
    ("min_depth=1000",  {"min_pool_depth_tao": 1000.0}, None, None),
    ("min_depth=2000",  {"min_pool_depth_tao": 2000.0}, None, None),

    # === Capital sub-100 ===
    ("capital=50",      {}, None, 50.0),
    ("capital=200",     {}, None, 200.0),
    ("capital=300",     {}, None, 300.0),

    # === LAM-only / RCB-only at exact window (sanity, since baseline includes drain_exit safety) ===
    ("LAM only",        {}, ["lam"], None),
    ("RCB only",        {}, ["rcb"], None),
]


def main():
    print(f"Running {len(EXPERIMENTS)} experiments with full process isolation...")
    results = []
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

    results.sort(key=lambda r: (-r["sortino"], -r["calmar"]))

    print("\n" + "=" * 110)
    print(f'{"experiment":<26} {"ret%":>7} {"annual%":>9} {"sortino":>8} {"calmar":>7} {"DD%":>7} {"trades":>6} {"PF":>5}')
    print("-" * 110)
    for r in results:
        print(f'{r["name"]:<26} {r["return_pct"]:>+7.2f} {r["annual_pct"]:>+9.1f} '
              f'{r["sortino"]:>+8.2f} {r["calmar"]:>+7.2f} {r["max_dd_pct"]:>+7.2f} '
              f'{r["trades"]:>6} {r["pf"]:>5.2f}')

    Path("results/experiments_round2.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: results/experiments_round2.json")


if __name__ == "__main__":
    main()
