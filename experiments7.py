"""Round 7 — walk-forward at 3 TAO to confirm robustness.

Round 6 found LAM solo at 3 TAO = +26.85% on full backtest. Need to
verify it's not regime-specific (same trap that took LAM+WSI from
+20.61% headline to 50% walk-forward fold rate).
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
CAPITAL = 3.0


def _iso(d): return d.strftime("%Y-%m-%d")


def slice_test_pnl(trades, test_start, test_end, initial_capital):
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


def run_walk(name: str, strategies: list[str], test_start, test_end, overrides=None):
    warmup_start = max(test_start - timedelta(days=WARMUP_DAYS), DATA_START)
    cfg = TradingConfig(db_path=DB_PATH)
    cfg.initial_capital_tao = CAPITAL
    cfg.rcb_hold_hours = 18.0
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    bt = Backtester(cfg)
    res = bt.run(start=_iso(warmup_start), end=_iso(test_end), strategies=strategies)
    test_metrics = slice_test_pnl(
        [t.to_dict() if hasattr(t, "to_dict") else t for t in res.trades],
        test_start, test_end, res.initial_capital,
    )
    days = (test_end - test_start).total_seconds() / 86400
    annual = ((1 + test_metrics["return_pct"]/100) ** (365/days) - 1) * 100 if days > 0 else float("nan")
    return {
        "name": name,
        "test_start": _iso(test_start),
        "test_end": _iso(test_end),
        "test_days": days,
        "test_return_pct": test_metrics["return_pct"],
        "test_annual_pct": annual,
        "test_trades": test_metrics["trades_in_window"],
        "test_win_rate": test_metrics["win_rate"],
        "test_pnl_tao": test_metrics["pnl_tao"],
    }


TEST_WINDOWS = []
cursor = DATA_START + timedelta(days=WARMUP_DAYS)
while cursor + timedelta(days=TEST_DAYS) <= DATA_END:
    TEST_WINDOWS.append((cursor, cursor + timedelta(days=TEST_DAYS)))
    cursor += timedelta(days=10)


CONFIGS = [
    ("LAM solo @ 3 TAO", ["lam"], None),
    ("LAM+WSI @ 3 TAO", ["lam", "wsi"], None),
    ("LAM+RCB @ 3 TAO", ["lam", "rcb"], None),
    ("momentum solo @ 3 TAO max_pos=2", ["momentum"],
     {"reserve_pct": 0.0, "max_positions": 2, "min_trade_tao": 0.05}),
]


def main():
    jobs = []
    for name, strats, overrides in CONFIGS:
        for ts, te in TEST_WINDOWS:
            jobs.append((name, strats, ts, te, overrides))
    print(f"Running {len(jobs)} walk-forward folds @ {CAPITAL} TAO...")
    print(f"Test windows: {[(_iso(s), _iso(e)) for s, e in TEST_WINDOWS]}")
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_walk, n, s, ts, te, o): n for n, s, ts, te, o in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']} [{r['test_start']}→{r['test_end']}]: "
                      f"ret={r['test_return_pct']:+.2f}% ({r['test_trades']} trades, "
                      f"wr={r['test_win_rate']*100:.0f}%)", flush=True)
            except Exception as e:
                print(f"  ✗ {futures[fut]}: {e}", flush=True)

    from collections import defaultdict
    agg = defaultdict(lambda: {"rets": [], "trades": 0})
    for r in results:
        agg[r["name"]]["rets"].append(r["test_return_pct"])
        agg[r["name"]]["trades"] += r["test_trades"]

    print("\n" + "=" * 110)
    print(f'{"config":<36} {"folds":>5} {"avg%":>7} {"med%":>7} {"min%":>7} {"max%":>7} {"+folds":>7} {"trades":>7}')
    print("-" * 110)
    rows = []
    for name, a in agg.items():
        rets = sorted(a["rets"])
        if not rets: continue
        rows.append((name, len(rets), sum(rets)/len(rets), rets[len(rets)//2],
                     min(rets), max(rets), sum(1 for r in rets if r > 0)/len(rets)*100,
                     a["trades"]))
    rows.sort(key=lambda r: -r[2])
    for r in rows:
        print(f'{r[0]:<36} {r[1]:>5} {r[2]:>+7.2f} {r[3]:>+7.2f} {r[4]:>+7.2f} {r[5]:>+7.2f} {r[6]:>6.0f}% {r[7]:>7}')

    Path("results/experiments_round7_walkforward_3tao.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: results/experiments_round7_walkforward_3tao.json")


if __name__ == "__main__":
    main()
