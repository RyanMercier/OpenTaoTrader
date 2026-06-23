"""Round 4 — push LAM+WSI past +20.61%.

Round 3 surfaced LAM+WSI at +20.61% / +178% annualized / Sortino +2.33.
Round 4:
  1. Confirm reproducibility (rerun baseline)
  2. Sweep WSI params (percentile, hold hours, stop/TP, min_pool_depth)
  3. Try LAM+WSI+X triplets (X = best diversifier from round 3)
  4. Sub-window robustness: re-test on the first vs last half of the window
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
FULL_START = "2026-04-15"
FULL_END = "2026-06-21"
HALF1_START = "2026-04-15"
HALF1_END = "2026-05-18"
HALF2_START = "2026-05-18"
HALF2_END = "2026-06-21"


def run_one(name: str, strategies: list[str], overrides: dict = None,
            start: str = FULL_START, end: str = FULL_END) -> dict:
    cfg = TradingConfig(db_path=DB_PATH)
    cfg.rcb_hold_hours = 18.0
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    bt = Backtester(cfg)
    t0 = time.time()
    res = bt.run(start=start, end=end, strategies=strategies)
    elapsed = time.time() - t0
    real_ret_pct = (res.final_value / res.initial_capital - 1) * 100
    from datetime import datetime
    s = datetime.fromisoformat(res.start_time.replace("Z","+00:00"))
    e = datetime.fromisoformat(res.end_time.replace("Z","+00:00"))
    days = (e - s).total_seconds() / 86400
    annual = ((1 + real_ret_pct/100) ** (365/days) - 1) * 100 if days > 0 else float("nan")
    return {
        "name": name,
        "strategies": ",".join(strategies),
        "return_pct": real_ret_pct,
        "annual_pct": annual,
        "sharpe": res.sharpe_ratio,
        "sortino": res.sortino_ratio,
        "calmar": res.calmar_ratio,
        "max_dd_pct": res.max_drawdown_pct * 100,
        "trades": res.buy_trades + res.sell_trades,
        "pf": res.profit_factor,
        "final": res.final_value,
        "elapsed_s": elapsed,
    }


EXPERIMENTS = [
    # === Reproducibility ===
    ("BASELINE LAM+WSI (reproduce)", ["lam", "wsi"], None, FULL_START, FULL_END),
    ("LAM+RCB ref", ["lam", "rcb"], None, FULL_START, FULL_END),

    # === WSI percentile sweep (cleaner whale signal = higher percentile) ===
    ("LAM+WSI percentile=0.90", ["lam", "wsi"], {"wsi_percentile": 0.90}, FULL_START, FULL_END),
    ("LAM+WSI percentile=0.95", ["lam", "wsi"], {"wsi_percentile": 0.95}, FULL_START, FULL_END),
    ("LAM+WSI percentile=0.97 (default)", ["lam", "wsi"], {"wsi_percentile": 0.97}, FULL_START, FULL_END),
    ("LAM+WSI percentile=0.98", ["lam", "wsi"], {"wsi_percentile": 0.98}, FULL_START, FULL_END),
    ("LAM+WSI percentile=0.99", ["lam", "wsi"], {"wsi_percentile": 0.99}, FULL_START, FULL_END),

    # === WSI hold hours sweep ===
    ("LAM+WSI hold=24h", ["lam", "wsi"], {"wsi_hold_hours": 24.0}, FULL_START, FULL_END),
    ("LAM+WSI hold=48h (default)", ["lam", "wsi"], {"wsi_hold_hours": 48.0}, FULL_START, FULL_END),
    ("LAM+WSI hold=72h", ["lam", "wsi"], {"wsi_hold_hours": 72.0}, FULL_START, FULL_END),
    ("LAM+WSI hold=96h", ["lam", "wsi"], {"wsi_hold_hours": 96.0}, FULL_START, FULL_END),

    # === WSI take-profit / stop-loss sweep ===
    ("LAM+WSI tp=0.08", ["lam", "wsi"], {"wsi_take_profit_pct": 0.08}, FULL_START, FULL_END),
    ("LAM+WSI tp=0.12", ["lam", "wsi"], {"wsi_take_profit_pct": 0.12}, FULL_START, FULL_END),
    ("LAM+WSI tp=0.20", ["lam", "wsi"], {"wsi_take_profit_pct": 0.20}, FULL_START, FULL_END),
    ("LAM+WSI sl=-0.04", ["lam", "wsi"], {"wsi_stop_loss_pct": -0.04}, FULL_START, FULL_END),
    ("LAM+WSI sl=-0.08", ["lam", "wsi"], {"wsi_stop_loss_pct": -0.08}, FULL_START, FULL_END),

    # === WSI pool depth ===
    ("LAM+WSI depth=500", ["lam", "wsi"], {"wsi_min_pool_depth": 500.0}, FULL_START, FULL_END),
    ("LAM+WSI depth=1000", ["lam", "wsi"], {"wsi_min_pool_depth": 1000.0}, FULL_START, FULL_END),

    # === LAM+WSI triplets ===
    ("LAM+WSI+RCB", ["lam", "wsi", "rcb"], None, FULL_START, FULL_END),
    ("LAM+WSI+SFB", ["lam", "wsi", "sfb"], None, FULL_START, FULL_END),
    ("LAM+WSI+PMR", ["lam", "wsi", "pmr"], None, FULL_START, FULL_END),
    ("LAM+WSI+stake_velocity", ["lam", "wsi", "stake_velocity"], None, FULL_START, FULL_END),
    ("LAM+WSI+pdmr", ["lam", "wsi", "pdmr"], None, FULL_START, FULL_END),

    # === OOS robustness: split window in half ===
    ("LAM+WSI first half", ["lam", "wsi"], None, HALF1_START, HALF1_END),
    ("LAM+WSI second half", ["lam", "wsi"], None, HALF2_START, HALF2_END),
    ("LAM+RCB first half (ref)", ["lam", "rcb"], None, HALF1_START, HALF1_END),
    ("LAM+RCB second half (ref)", ["lam", "rcb"], None, HALF2_START, HALF2_END),

    # === Smaller universe: tighter min_pool_depth_tao at backtester level ===
    ("LAM+WSI bt_min_depth=500", ["lam", "wsi"], {"min_pool_depth_tao": 500.0}, FULL_START, FULL_END),
    ("LAM+WSI bt_min_depth=1000", ["lam", "wsi"], {"min_pool_depth_tao": 1000.0}, FULL_START, FULL_END),
]


def main():
    print(f"Running {len(EXPERIMENTS)} experiments (process-isolated)...")
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {}
        for name, strategies, overrides, start, end in EXPERIMENTS:
            fut = pool.submit(run_one, name, strategies, overrides, start, end)
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
    print(f'{"experiment":<36} {"ret%":>7} {"annual%":>9} {"sortino":>8} {"calmar":>7} {"DD%":>7} {"trades":>6} {"PF":>5}')
    print("-" * 110)
    for r in results:
        print(f'{r["name"]:<36} {r["return_pct"]:>+7.2f} {r["annual_pct"]:>+9.1f} '
              f'{r["sortino"]:>+8.2f} {r["calmar"]:>+7.2f} {r["max_dd_pct"]:>+7.2f} '
              f'{r["trades"]:>6} {r["pf"]:>5.2f}')

    Path("results/experiments_round4.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: results/experiments_round4.json")


if __name__ == "__main__":
    main()
