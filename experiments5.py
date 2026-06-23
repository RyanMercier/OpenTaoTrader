"""Round 5 — walk-forward validation of LAM+WSI with proper warmup.

Round 4 showed split-half OOS gave 0.25% in the second half. Per-trade
analysis on the FULL backtest showed LAM made +9.85 TAO in that same
window. The discrepancy is feature warmup — features (z-scores,
momentum) need ~7d of prior bars to compute. A backtest that starts at
May 18 wastes that warmup; the per-trade view of a backtest that
started Apr 15 has all the history it needs.

To verify the strategy is robust (not overfit), we slide 14-day test
windows across the dataset but always load 14 days of prior data for
warmup. The backtester sees the full warmup+test window; we then
compute returns only against the test slice via the per-trade record.

Final rounds:
  1. 5 sliding 14-day test windows with 14-day warmup each
  2. Confirm best-config LAM+WSI is consistently profitable
  3. Save the final config and report annualized expectation
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading.backtester import Backtester
from trading.config import TradingConfig


DB_PATH = "/home/ryan/bittensor/OpenTaoAPI/TaoOpenAPI/data/opentao.db"
DATA_START = datetime(2026, 4, 15, tzinfo=timezone.utc)
DATA_END = datetime(2026, 6, 21, tzinfo=timezone.utc)
WARMUP_DAYS = 14
TEST_DAYS = 14


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def slice_test_pnl(trades: list, test_start: datetime, test_end: datetime,
                   initial_capital: float) -> dict:
    """Compute returns from sell trades whose timestamp falls inside the
    test window. Treats it as if you started fresh at test_start with the
    initial capital and only the trades that happen in-window count."""
    in_window = []
    for t in trades:
        if t.get("direction") != "sell" or t.get("pnl_tao") is None:
            continue
        try:
            ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
        except Exception:
            continue
        if test_start <= ts < test_end:
            in_window.append(t)
    pnl = sum(t["pnl_tao"] for t in in_window)
    wins = sum(1 for t in in_window if t["pnl_pct"] > 0)
    n = len(in_window)
    return {
        "trades_in_window": n,
        "wins": wins,
        "win_rate": wins / max(n, 1),
        "pnl_tao": pnl,
        "return_pct": pnl / initial_capital * 100,
    }


def run_walk(name: str, strategies: list[str], test_start: datetime,
             test_end: datetime, overrides: dict = None) -> dict:
    """Backtest from (test_start - warmup) to test_end; compute returns
    from trades inside [test_start, test_end)."""
    warmup_start = max(test_start - timedelta(days=WARMUP_DAYS), DATA_START)
    cfg = TradingConfig(db_path=DB_PATH)
    cfg.rcb_hold_hours = 18.0
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    bt = Backtester(cfg)
    res = bt.run(
        start=_iso(warmup_start),
        end=_iso(test_end),
        strategies=strategies,
    )
    test_metrics = slice_test_pnl(
        [t.to_dict() if hasattr(t, "to_dict") else t for t in res.trades],
        test_start, test_end, res.initial_capital,
    )
    days = (test_end - test_start).total_seconds() / 86400
    annual = ((1 + test_metrics["return_pct"]/100) ** (365/days) - 1) * 100 if days > 0 else float("nan")
    return {
        "name": name,
        "strategies": ",".join(strategies),
        "warmup_start": _iso(warmup_start),
        "test_start": _iso(test_start),
        "test_end": _iso(test_end),
        "test_days": days,
        "test_return_pct": test_metrics["return_pct"],
        "test_annual_pct": annual,
        "test_trades": test_metrics["trades_in_window"],
        "test_win_rate": test_metrics["win_rate"],
        "test_pnl_tao": test_metrics["pnl_tao"],
    }


# 5 sliding 14-day test windows. First window starts 14 days into the data
# (so warmup is fully inside the dataset).
TEST_WINDOWS = []
cursor = DATA_START + timedelta(days=WARMUP_DAYS)
while cursor + timedelta(days=TEST_DAYS) <= DATA_END:
    TEST_WINDOWS.append((cursor, cursor + timedelta(days=TEST_DAYS)))
    cursor += timedelta(days=10)  # slide forward 10 days per fold


CONFIGS = [
    ("LAM+WSI default", ["lam", "wsi"], None),
    ("LAM+WSI pct=0.98", ["lam", "wsi"], {"wsi_percentile": 0.98}),
    ("LAM+RCB ref", ["lam", "rcb"], None),
    ("LAM solo", ["lam"], None),
]


def main():
    jobs = []
    for cfg_name, strats, overrides in CONFIGS:
        for ts, te in TEST_WINDOWS:
            jobs.append((cfg_name, strats, ts, te, overrides))
    print(f"Running {len(jobs)} walk-forward folds ({len(CONFIGS)} configs × {len(TEST_WINDOWS)} windows)...")
    print(f"Test windows: {[(_iso(s), _iso(e)) for s, e in TEST_WINDOWS]}")
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {}
        for name, strats, ts, te, overrides in jobs:
            fut = pool.submit(run_walk, name, strats, ts, te, overrides)
            futures[fut] = (name, ts, te)
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                print(f"  ✓ {res['name']} [{res['test_start']}→{res['test_end']}]: "
                      f"ret={res['test_return_pct']:+.2f}% ({res['test_trades']} trades, "
                      f"wr={res['test_win_rate']*100:.0f}%)", flush=True)
            except Exception as e:
                n, ts, te = futures[fut]
                print(f"  ✗ {n} [{_iso(ts)}→{_iso(te)}]: {e}", flush=True)

    # Aggregate per config
    from collections import defaultdict
    agg = defaultdict(lambda: {"rets": [], "trades": 0, "pnl": 0.0})
    for r in results:
        a = agg[r["name"]]
        a["rets"].append(r["test_return_pct"])
        a["trades"] += r["test_trades"]
        a["pnl"] += r["test_pnl_tao"]

    print("\n" + "=" * 110)
    print(f'{"config":<24} {"folds":>5} {"avg_ret%":>9} {"med_ret%":>9} {"min%":>7} {"max%":>7} {"+%folds":>8} {"tot_trades":>10}')
    print("-" * 110)
    rows = []
    for name, a in agg.items():
        rets = a["rets"]
        if not rets: continue
        rets_sorted = sorted(rets)
        med = rets_sorted[len(rets)//2]
        avg = sum(rets) / len(rets)
        pos = sum(1 for r in rets if r > 0)
        rows.append((name, len(rets), avg, med, min(rets), max(rets), pos/len(rets)*100, a["trades"]))
    rows.sort(key=lambda r: -r[2])  # by avg return
    for r in rows:
        print(f'{r[0]:<24} {r[1]:>5} {r[2]:>+9.2f} {r[3]:>+9.2f} {r[4]:>+7.2f} {r[5]:>+7.2f} {r[6]:>7.0f}% {r[7]:>10}')

    Path("results/experiments_round5_walkforward.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: results/experiments_round5_walkforward.json")


if __name__ == "__main__":
    main()
