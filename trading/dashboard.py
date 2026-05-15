"""Self-contained HTML dashboards. All data is embedded inline as JSON; only
Chart.js is loaded from a CDN.
"""

from __future__ import annotations

import json
import os
import webbrowser
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .models import Direction, Snapshot, Signal
from .portfolio import PortfolioTracker

if TYPE_CHECKING:
    from .backtester import BacktestResult


CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _month_heatmap_html(monthly_returns: dict) -> str:
    if not monthly_returns:
        return "<p style='color:#888'>No monthly data.</p>"
    rows = []
    for month in sorted(monthly_returns.keys()):
        r = monthly_returns[month]
        pct = r * 100
        color = "#1b5e20" if pct > 0 else ("#b71c1c" if pct < 0 else "#555")
        rows.append(
            f"<tr><td>{month}</td><td style='color:#fff;background:{color};text-align:right;padding:4px 12px'>{pct:+.2f}%</td></tr>"
        )
    return (
        "<table style='border-collapse:collapse'>"
        "<thead><tr><th style='text-align:left'>Month</th><th style='text-align:right'>Return</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _strategy_table_html(strategy_stats: dict) -> str:
    if not strategy_stats:
        return "<p style='color:#888'>No strategy breakdown.</p>"
    rows = []
    for name, st in sorted(strategy_stats.items(), key=lambda kv: -kv[1].get("total_pnl_tao", 0)):
        rows.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td style='text-align:right'>{st['buys']}</td>"
            f"<td style='text-align:right'>{st['sells']}</td>"
            f"<td style='text-align:right'>{st['win_rate']*100:.1f}%</td>"
            f"<td style='text-align:right'>{st['avg_return_pct']*100:+.2f}%</td>"
            f"<td style='text-align:right'>{st['total_pnl_tao']:+.3f}</td>"
            f"</tr>"
        )
    return (
        "<table style='border-collapse:collapse;width:100%'>"
        "<thead><tr>"
        "<th style='text-align:left'>Strategy</th>"
        "<th style='text-align:right'>Buys</th>"
        "<th style='text-align:right'>Sells</th>"
        "<th style='text-align:right'>Win%</th>"
        "<th style='text-align:right'>AvgRet</th>"
        "<th style='text-align:right'>PnL (TAO)</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _compute_benchmark(db_path: str, start_iso: str, end_iso: str) -> list[dict]:
    """Equal-weight buy-and-hold benchmark across all subnets in the window.

    At time T, the benchmark value is the average of (price_at_T / price_at_start)
    across all subnets that have data. Normalized to the strategy's initial
    capital at rendering time.
    """
    import sqlite3
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, netuid, alpha_price_tao FROM subnet_snapshots "
                "WHERE timestamp >= ? AND timestamp <= ? AND alpha_price_tao > 0 "
                "ORDER BY timestamp ASC",
                (start_iso, end_iso),
            ).fetchall()
    except Exception:
        return []
    if not rows:
        return []
    # Baseline per subnet = first observed price
    baselines: dict[int, float] = {}
    series: dict[str, dict[int, float]] = {}
    for r in rows:
        n = r["netuid"]
        p = r["alpha_price_tao"]
        ts = r["timestamp"]
        baselines.setdefault(n, p)
        series.setdefault(ts, {})[n] = p / baselines[n]
    # Average across subnets at each timestamp (equal-weight)
    benchmark = []
    for ts in sorted(series.keys()):
        ratios = list(series[ts].values())
        if ratios:
            benchmark.append({"t": ts, "v": sum(ratios) / len(ratios)})
    return benchmark


def _trade_histogram(trades: list, bins: list[float]) -> list[dict]:
    """Histogram of trade P&L percentages. `bins` is the lower edges of buckets
    (all in pct units, e.g. [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05])."""
    counts = [0] * (len(bins) + 1)
    for t in trades:
        pnl = t.pnl_pct if hasattr(t, "pnl_pct") else t.get("pnl_pct")
        if pnl is None:
            continue
        placed = False
        for i, edge in enumerate(bins):
            if pnl < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    # Emit human-readable bucket labels
    out = []
    labels = []
    prev = None
    for edge in bins:
        label = f"< {edge*100:+.1f}%" if prev is None else f"{prev*100:+.1f}% to {edge*100:+.1f}%"
        labels.append(label)
        prev = edge
    labels.append(f">= {bins[-1]*100:+.1f}%")
    for lab, c in zip(labels, counts):
        out.append({"bucket": lab, "count": c})
    return out


def _hold_histogram(sells: list) -> list[dict]:
    buckets = [0, 1, 2, 4, 8, 24, 72, 168, 9999]
    names = ["<1h", "1-2h", "2-4h", "4-8h", "8-24h", "1-3d", "3-7d", "7d+"]
    counts = [0] * len(names)
    for t in sells:
        h = t.hold_duration_hours if hasattr(t, "hold_duration_hours") else t.get("hold_duration_hours")
        if h is None:
            continue
        for i in range(len(names)):
            if h < buckets[i + 1]:
                counts[i] += 1
                break
    return [{"bucket": n, "count": c} for n, c in zip(names, counts)]


def _entry_exit_matrix(sells: list) -> dict:
    """Build a 2D table: rows=entry strategy, cols=exit strategy, cells={n, winrate, total_pnl}."""
    from collections import defaultdict
    cells: dict = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_tao": 0.0, "pnl_pct_sum": 0.0})
    entries = set()
    exits = set()
    for t in sells:
        entry = t.entry_strategy.value if hasattr(t, "entry_strategy") and t.entry_strategy else (
            t.get("entry_strategy") if isinstance(t, dict) else "?"
        )
        exitn = t.strategy.value if hasattr(t, "strategy") and hasattr(t.strategy, "value") else (
            t.get("strategy") if isinstance(t, dict) else "?"
        )
        entry = entry or "?"
        exitn = exitn or "?"
        entries.add(entry)
        exits.add(exitn)
        key = (entry, exitn)
        pnl_tao = t.pnl_tao if hasattr(t, "pnl_tao") else t.get("pnl_tao")
        pnl_pct = t.pnl_pct if hasattr(t, "pnl_pct") else t.get("pnl_pct")
        cells[key]["n"] += 1
        if (pnl_tao or 0) > 0:
            cells[key]["wins"] += 1
        cells[key]["pnl_tao"] += pnl_tao or 0.0
        cells[key]["pnl_pct_sum"] += pnl_pct or 0.0
    return {
        "entries": sorted(entries),
        "exits": sorted(exits),
        "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
    }


def _per_subnet_pnl(trades: list) -> dict:
    """Cumulative P&L per subnet over time (only sells contribute to realized P&L)."""
    from collections import defaultdict
    series: dict[int, list[dict]] = defaultdict(list)
    running: dict[int, float] = defaultdict(float)
    for t in trades:
        direction = t.direction.value if hasattr(t, "direction") and hasattr(t.direction, "value") else t.get("direction")
        if direction != "sell":
            continue
        pnl = t.pnl_tao if hasattr(t, "pnl_tao") else t.get("pnl_tao")
        ts = t.timestamp.isoformat() if hasattr(t, "timestamp") and hasattr(t.timestamp, "isoformat") else (t.get("timestamp") if isinstance(t, dict) else None)
        netuid = t.netuid if hasattr(t, "netuid") else t.get("netuid")
        if pnl is None or ts is None or netuid is None:
            continue
        running[netuid] += pnl
        series[netuid].append({"t": ts, "v": running[netuid]})
    return {str(k): v for k, v in series.items()}


def _rolling_returns(equity: list, window_days: int = 30) -> list[dict]:
    """30-day (or configurable) rolling return at each tick.

    equity is a list of (iso_ts, value). Returns list of {t, v} where v is
    the pct change from (t - window_days) to t.
    """
    if not equity:
        return []
    # Build time -> value map, keep chronological order
    from datetime import datetime as _dt, timedelta
    pts = []
    for ts, v in equity:
        try:
            dt = _dt.fromisoformat(ts.replace("Z", "+00:00") if ts.endswith("Z") else ts)
            pts.append((dt, v))
        except Exception:
            continue
    out = []
    j = 0
    for i, (t, v) in enumerate(pts):
        target = t - timedelta(days=window_days)
        while j < i and pts[j][0] < target:
            j += 1
        if j >= i:
            continue
        base = pts[j][1]
        if base > 0:
            out.append({"t": t.isoformat(), "v": (v / base - 1) * 100})
    return out


def generate_backtest_dashboard(
    result: "BacktestResult",
    output_path: str = "data/backtest_dashboard.html",
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    equity_data = [
        {"t": ts, "v": v} for ts, v in result.portfolio_values
    ]
    drawdown_data = [
        {"t": ts, "v": v * 100} for ts, v in result.drawdown_series
    ]

    # Benchmark (equal-weight buy-and-hold across all subnets in scope)
    cfg_raw = getattr(result, "config", None)
    if isinstance(cfg_raw, dict):
        db_path = cfg_raw.get("db_path")
    else:
        db_path = getattr(cfg_raw, "db_path", None) if cfg_raw is not None else None
    benchmark_ratios = _compute_benchmark(db_path, result.start_time, result.end_time) if db_path else []
    initial = result.initial_capital
    benchmark_equity = [{"t": b["t"], "v": b["v"] * initial} for b in benchmark_ratios]

    # Rolling 30-day returns for strategy + benchmark
    rolling_strategy = _rolling_returns(result.portfolio_values, window_days=30)
    rolling_benchmark = _rolling_returns([(b["t"], b["v"] * initial) for b in benchmark_ratios], window_days=30)

    # Per-subnet cumulative P&L
    per_subnet = _per_subnet_pnl(result.trades)

    # Trade P&L histogram, 11 buckets
    sells = [t for t in result.trades if (t.direction.value if hasattr(t.direction, "value") else t.direction) == "sell"]
    pnl_hist = _trade_histogram(sells, [-0.05, -0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03, 0.05])
    hold_hist = _hold_histogram(sells)
    ee_matrix = _entry_exit_matrix(sells)

    trade_markers = []
    for t in result.trades:
        trade_markers.append({
            "t": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else t.timestamp,
            "netuid": t.netuid,
            "direction": t.direction.value if hasattr(t.direction, "value") else t.direction,
            "strategy": t.strategy.value if hasattr(t.strategy, "value") else t.strategy,
            "entry_strategy": t.entry_strategy.value if hasattr(t, "entry_strategy") and t.entry_strategy else None,
            "tao": t.tao_amount,
            "alpha": t.alpha_amount,
            "slip": t.slippage_pct,
            "pnl_pct": t.pnl_pct,
            "pnl_tao": t.pnl_tao,
            "hold_hours": t.hold_duration_hours,
        })

    slippage_scatter = []
    for t in result.trades:
        if t.direction == Direction.BUY and t.tao_amount > 0:
            # pool depth at entry: we don't have it here directly; approximate
            # using spot_price * alpha (tao_in = spot * alpha_in; we know
            # effective price and tao_amount). Skip if uncertain.
            pool_pct = None
            if t.spot_price > 0:
                # alpha_in ~= tao_amount / slippage * ..., punt and use
                # tao_amount / (tao_amount/slip) as a rough relative size.
                # More honestly: slippage_pct = tao_amount / tao_in for buys
                # in our no-fee AMM, so tao_in ≈ tao_amount / slippage_pct.
                if t.slippage_pct > 0:
                    pool_pct = t.slippage_pct  # by derivation above
            if pool_pct is not None:
                slippage_scatter.append({
                    "x": pool_pct * 100,  # trade as % of pool
                    "y": t.slippage_pct * 100,
                    "netuid": t.netuid,
                })

    # Position timeline: each held period becomes a bar
    position_timeline = []
    # Build from sells joined with the most recent preceding buy for the same netuid
    open_entries: dict = {}
    for t in result.trades:
        if t.direction == Direction.BUY:
            open_entries[t.netuid] = t
        elif t.direction == Direction.SELL and t.netuid in open_entries:
            entry = open_entries.pop(t.netuid)
            position_timeline.append({
                "netuid": t.netuid,
                "start": entry.timestamp.isoformat() if hasattr(entry.timestamp, "isoformat") else entry.timestamp,
                "end": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else t.timestamp,
                "pnl_pct": (t.pnl_pct or 0.0) * 100,
                "strategy": entry.strategy.value if hasattr(entry.strategy, "value") else entry.strategy,
            })

    # Benchmark return summary
    if benchmark_equity:
        bh_final = benchmark_equity[-1]["v"]
        bh_return_pct = (bh_final / initial - 1)
    else:
        bh_final = initial
        bh_return_pct = 0.0
    alpha_vs_bh = result.total_return_pct - bh_return_pct

    cfg_obj = getattr(result, "config", None)
    cfg_dict = {}
    if cfg_obj is not None:
        try:
            from dataclasses import asdict as _asdict, is_dataclass
            cfg_dict = _asdict(cfg_obj) if is_dataclass(cfg_obj) else (cfg_obj if isinstance(cfg_obj, dict) else {})
        except Exception:
            cfg_dict = {}

    payload = {
        "summary": {
            "start_time": result.start_time,
            "end_time": result.end_time,
            "initial_capital": result.initial_capital,
            "final_value": result.final_value,
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "calmar": result.calmar_ratio,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "avg_hold_hours": result.avg_hold_hours,
            "total_slippage_cost": result.total_slippage_cost_tao,
            "strategies": result.strategies_used,
            "regime_filter": result.regime_filter,
            "blocked_trades": result.blocked_trades,
            "benchmark_return_pct": bh_return_pct,
            "alpha_vs_benchmark": alpha_vs_bh,
            "num_hotkeys": cfg_dict.get("num_hotkeys"),
            "max_positions": cfg_dict.get("max_positions"),
            "max_single_position_pct": cfg_dict.get("max_single_position_pct"),
            "reserve_pct": cfg_dict.get("reserve_pct"),
        },
        "equity": equity_data,
        "benchmark": benchmark_equity,
        "rolling_strategy": rolling_strategy,
        "rolling_benchmark": rolling_benchmark,
        "drawdown": drawdown_data,
        "trades": trade_markers,
        "slippage": slippage_scatter,
        "positions": position_timeline,
        "per_subnet": per_subnet,
        "pnl_hist": pnl_hist,
        "hold_hist": hold_hist,
        "ee_matrix": ee_matrix,
    }
    monthly_table = _month_heatmap_html(result.monthly_returns)

    data_json = json.dumps(payload, default=_json_default)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bittensor Trading Backtest</title>
<script src="{CHART_JS_CDN}"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0b0e14; color: #e0e6ed; margin: 0; padding: 20px; max-width: 1600px; }}
  h1, h2 {{ color: #80cbc4; }}
  h2 {{ font-size: 18px; margin-top: 0; }}
  .banner {{ background: linear-gradient(90deg, #0f3b36 0%, #122228 100%);
             padding: 16px 20px; border-radius: 6px; margin-bottom: 20px;
             display: flex; justify-content: space-between; align-items: center; }}
  .banner .hl {{ font-size: 26px; font-weight: 600; color: #80cbc4; }}
  .banner .sub {{ color: #aaa; margin-top: 4px; font-size: 13px; }}
  .banner code {{ background: #0b0e14; padding: 6px 10px; border-radius: 4px;
                  font-family: 'SF Mono', Menlo, monospace; color: #80cbc4; }}
  .grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 20px; }}
  .stat {{ background: #1a1f28; padding: 12px; border-radius: 6px; border-left: 3px solid #80cbc4; }}
  .stat .label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat .value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .stat .sub   {{ font-size: 11px; color: #888; margin-top: 2px; }}
  .stat .good {{ color: #4caf50; }}
  .stat .bad {{ color: #e57373; }}
  .chart-container {{ background: #1a1f28; padding: 14px; border-radius: 6px; margin-bottom: 16px; }}
  canvas {{ max-height: 380px; }}
  table {{ width: 100%; color: #e0e6ed; border-collapse: collapse; }}
  th, td {{ padding: 5px 10px; border-bottom: 1px solid #2a2f38; font-size: 13px; }}
  th {{ color: #80cbc4; text-align: left; }}
  .cols2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .cols3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }}
  .cols2-2-1 {{ display: grid; grid-template-columns: 2fr 2fr 1fr; gap: 14px; }}
  .matrix td {{ text-align: right; padding: 6px 10px; }}
  .matrix td.label {{ text-align: left; color: #80cbc4; font-weight: 600; }}
  .cell-pos {{ background: rgba(76,175,80,0.18); }}
  .cell-neg {{ background: rgba(229,115,115,0.18); }}
</style>
</head>
<body>
<h1>Bittensor Subnet Trading, Backtest Dashboard</h1>
<p style="color:#888">{{summary.start_time}} -> {{summary.end_time}} · strategies: {{summary.strategies}}</p>

<div class="banner">
  <div>
    <div class="hl" id="bannerReturn"></div>
    <div class="sub" id="bannerSub"></div>
  </div>
  <div>
    <div class="sub">Deploy:</div>
    <code id="deployCmd"></code>
  </div>
</div>

<div class="grid" id="stats"></div>

<div class="chart-container">
  <h2>Equity Curve vs Equal-Weight Buy-&-Hold</h2>
  <canvas id="equityChart"></canvas>
</div>

<div class="cols2">
  <div class="chart-container">
    <h2>Rolling 30-Day Return</h2>
    <canvas id="rollingChart"></canvas>
  </div>
  <div class="chart-container">
    <h2>Drawdown</h2>
    <canvas id="drawdownChart"></canvas>
  </div>
</div>

<div class="chart-container">
  <h2>Cumulative Realized P&L by Subnet</h2>
  <canvas id="perSubnetChart"></canvas>
</div>

<div class="cols3">
  <div class="chart-container">
    <h2>Trade P&L Distribution</h2>
    <canvas id="pnlHistChart"></canvas>
  </div>
  <div class="chart-container">
    <h2>Hold Duration</h2>
    <canvas id="holdHistChart"></canvas>
  </div>
  <div class="chart-container">
    <h2>Monthly Returns</h2>
    {monthly_table}
  </div>
</div>

<div class="cols2">
  <div class="chart-container">
    <h2>Entry -> Exit Attribution</h2>
    <div id="eeMatrix"></div>
    <p style="color:#888;font-size:12px;margin-top:8px">
      Row = strategy that opened the position. Column = strategy that closed it.
      Cell shows trade count, win rate, total P&L (TAO). Hold_timeout is a time-based exit.
    </p>
  </div>
  <div class="chart-container">
    <h2>Slippage vs Trade Size</h2>
    <canvas id="slippageChart"></canvas>
  </div>
</div>

<div class="chart-container">
  <h2>Position Timeline</h2>
  <canvas id="positionsChart"></canvas>
</div>

<script>
const DATA = {data_json};

const COLORS = ['#80cbc4', '#ffb74d', '#ba68c8', '#aed581', '#4fc3f7', '#e57373', '#f06292', '#dce775'];

function pct(x) {{ return (x*100).toFixed(2) + '%'; }}
function taoStr(x) {{ return Number(x).toFixed(3); }}

function renderBanner() {{
  const s = DATA.summary;
  const ret = s.total_return_pct;
  const alpha = s.alpha_vs_benchmark;
  const alphaStr = alpha != null ? `${{(alpha*100).toFixed(2)}}%` : 'n/a';
  const bnh = s.benchmark_return_pct != null ? `${{(s.benchmark_return_pct*100).toFixed(2)}}%` : 'n/a';
  document.getElementById('bannerReturn').innerHTML =
    `<span class="${{ret>=0?'good':'bad'}}">${{(ret*100).toFixed(2)}}%</span>` +
    ` over ${{Math.round(durationDays())}}d  (annualized ${{(s.annualized_return_pct*100).toFixed(1)}}%, Sharpe ${{s.sharpe.toFixed(2)}})`;
  const alphaClass = alpha >= 0 ? 'good' : 'bad';
  document.getElementById('bannerSub').innerHTML =
    `vs equal-weight B&amp;H ${{bnh}}  ->  <span class="${{alphaClass}}">alpha ${{alphaStr}}</span>` +
    ` · max DD ${{pct(s.max_drawdown_pct)}} · ${{s.total_trades}} trades, ${{pct(s.win_rate)}} win rate`;
  const strats = (s.strategies || []).filter(x => x !== 'drain_exit').join(',');
  const hk = s.num_hotkeys || 2;
  const mp = s.max_positions || 2;
  document.getElementById('deployCmd').textContent =
    `python -m trading.cli paper --capital ${{s.initial_capital}} --strategies ${{strats || 'momentum,stmc'}} --hotkeys ${{hk}}`;
}}

function durationDays() {{
  const a = new Date(DATA.summary.start_time);
  const b = new Date(DATA.summary.end_time);
  return (b - a) / 86400000;
}}

function renderStats() {{
  const s = DATA.summary;
  const stats = [
    {{ label: 'Return',         value: pct(s.total_return_pct),         good: s.total_return_pct >= 0 }},
    {{ label: 'Annualized',     value: pct(s.annualized_return_pct),    good: s.annualized_return_pct >= 0 }},
    {{ label: 'vs B&H',         value: pct(s.alpha_vs_benchmark),       good: s.alpha_vs_benchmark >= 0,
                                sub: `bench ${{pct(s.benchmark_return_pct)}}` }},
    {{ label: 'Final Value',    value: taoStr(s.final_value) + ' TAO',  sub: `+${{taoStr(s.final_value - s.initial_capital)}}` }},
    {{ label: 'Sharpe',         value: s.sharpe.toFixed(2) }},
    {{ label: 'Sortino',        value: s.sortino.toFixed(2) }},
    {{ label: 'Max DD',         value: pct(s.max_drawdown_pct),         good: false }},
    {{ label: 'Win Rate',       value: pct(s.win_rate),                 sub: `${{s.total_trades}} trades` }},
    {{ label: 'Avg Hold',       value: s.avg_hold_hours.toFixed(1) + 'h' }},
    {{ label: 'Slip Cost',      value: taoStr(s.total_slippage_cost) + ' TAO',
                                sub: `${{s.initial_capital>0?((s.total_slippage_cost/s.initial_capital)*100).toFixed(2):'0'}}% of capital` }},
    {{ label: 'Blocked',        value: s.blocked_trades,                sub: 'rate-limit skips' }},
    {{ label: 'Hotkeys',        value: s.num_hotkeys || '?',            sub: `max_pos=${{s.max_positions || '?'}}, conc=${{s.max_single_position_pct?(s.max_single_position_pct*100).toFixed(0):'?'}}%` }},
  ];
  document.getElementById('stats').innerHTML = stats.map(st => {{
    const cls = st.good === true ? ' good' : (st.good === false ? ' bad' : '');
    const sub = st.sub ? `<div class="sub">${{st.sub}}</div>` : '';
    return `<div class="stat"><div class="label">${{st.label}}</div><div class="value${{cls}}">${{st.value}}</div>${{sub}}</div>`;
  }}).join('');
}}

function renderEquity() {{
  const labels = DATA.equity.map(p => p.t);
  const values = DATA.equity.map(p => p.v);
  const initial = DATA.summary.initial_capital;
  const benchmark = (DATA.benchmark || []);
  const buys = DATA.trades.filter(t => t.direction === 'buy');
  const sells = DATA.trades.filter(t => t.direction === 'sell');

  // Align benchmark timestamps to nearest equity tick (approximate)
  const benchPoints = benchmark.map(b => ({{ x: b.t, y: b.v }}));

  new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{
          label: 'Strategy',
          data: values.map((v, i) => ({{ x: labels[i], y: v }})),
          borderColor: '#80cbc4',
          backgroundColor: 'rgba(128,203,196,0.1)',
          fill: true,
          pointRadius: 0,
          tension: 0.1,
          parsing: false,
        }},
        {{
          label: 'Equal-weight B&H',
          data: benchPoints,
          borderColor: '#ffb74d',
          borderDash: [4,4],
          pointRadius: 0,
          fill: false,
          parsing: false,
        }},
        {{
          label: 'Initial',
          data: labels.map(t => ({{x: t, y: initial}})),
          borderColor: '#555',
          borderDash: [6,4],
          pointRadius: 0,
          fill: false,
          parsing: false,
        }},
        {{
          label: 'Buys',
          type: 'scatter',
          data: buys.map(b => ({{ x: b.t, y: initial }})),
          backgroundColor: '#4caf50',
          pointStyle: 'triangle',
          pointRadius: 4,
        }},
        {{
          label: 'Sells',
          type: 'scatter',
          data: sells.map(s => ({{ x: s.t, y: initial }})),
          backgroundColor: '#e57373',
          pointStyle: 'triangle',
          rotation: 180,
          pointRadius: 4,
        }},
      ],
    }},
    options: {{
      scales: {{ x: {{ type: 'time', time: {{ parser: 'iso' }}, ticks: {{ autoSkip: true, maxTicksLimit: 12 }} }} }},
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              if (ctx.dataset.type === 'scatter') {{
                const ix = ctx.dataIndex;
                const src = ctx.dataset.label === 'Buys' ? buys[ix] : sells[ix];
                const pnl = src.pnl_pct != null ? ` pnl ${{(src.pnl_pct*100).toFixed(2)}}%` : '';
                const entry = src.entry_strategy ? `${{src.entry_strategy}}->${{src.strategy}}` : src.strategy;
                return `${{ctx.dataset.label}} SN${{src.netuid}} [${{entry}}] slip ${{(src.slip*100).toFixed(3)}}%${{pnl}}`;
              }}
              return `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(3)}} TAO`;
            }}
          }}
        }}
      }},
    }},
  }});
}}

function renderRolling() {{
  new Chart(document.getElementById('rollingChart'), {{
    type: 'line',
    data: {{
      datasets: [
        {{
          label: 'Strategy 30d return',
          data: (DATA.rolling_strategy || []).map(p => ({{ x: p.t, y: p.v }})),
          borderColor: '#80cbc4',
          pointRadius: 0,
          tension: 0.1,
          parsing: false,
        }},
        {{
          label: 'B&H 30d return',
          data: (DATA.rolling_benchmark || []).map(p => ({{ x: p.t, y: p.v }})),
          borderColor: '#ffb74d',
          borderDash: [4,4],
          pointRadius: 0,
          tension: 0.1,
          parsing: false,
        }},
      ]
    }},
    options: {{
      scales: {{
        x: {{ type: 'time', time: {{ parser: 'iso' }}, ticks: {{ autoSkip: true, maxTicksLimit: 8 }} }},
        y: {{ title: {{ display: true, text: 'Trailing 30-day return (%)' }} }}
      }}
    }},
  }});
}}

function renderDrawdown() {{
  new Chart(document.getElementById('drawdownChart'), {{
    type: 'line',
    data: {{
      datasets: [{{
        label: 'Drawdown (%)',
        data: DATA.drawdown.map(p => ({{x: p.t, y: p.v}})),
        borderColor: '#e57373',
        backgroundColor: 'rgba(229,115,115,0.25)',
        fill: true,
        pointRadius: 0,
        tension: 0.1,
        parsing: false,
      }}],
    }},
    options: {{ scales: {{ x: {{ type: 'time', time: {{ parser: 'iso' }}, ticks: {{ autoSkip: true, maxTicksLimit: 8 }} }} }} }},
  }});
}}

function renderPerSubnet() {{
  const datasets = Object.entries(DATA.per_subnet || {{}}).map(([netuid, pts], i) => ({{
    label: `SN${{netuid}}`,
    data: pts.map(p => ({{ x: p.t, y: p.v }})),
    borderColor: COLORS[i % COLORS.length],
    pointRadius: 0,
    tension: 0.1,
    parsing: false,
    fill: false,
  }}));
  new Chart(document.getElementById('perSubnetChart'), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      scales: {{
        x: {{ type: 'time', time: {{ parser: 'iso' }}, ticks: {{ autoSkip: true, maxTicksLimit: 12 }} }},
        y: {{ title: {{ display: true, text: 'Cumulative realized P&L (TAO)' }} }},
      }}
    }},
  }});
}}

function renderPnlHist() {{
  const hist = DATA.pnl_hist || [];
  new Chart(document.getElementById('pnlHistChart'), {{
    type: 'bar',
    data: {{
      labels: hist.map(h => h.bucket),
      datasets: [{{
        label: 'Trades',
        data: hist.map(h => h.count),
        backgroundColor: hist.map(h =>
          h.bucket.startsWith('<') || h.bucket.startsWith('-') ? 'rgba(229,115,115,0.7)' : 'rgba(76,175,80,0.7)'
        ),
      }}]
    }},
    options: {{
      scales: {{ x: {{ ticks: {{ maxRotation: 45, minRotation: 45 }} }} }},
      plugins: {{ legend: {{ display: false }} }},
    }},
  }});
}}

function renderHoldHist() {{
  const hist = DATA.hold_hist || [];
  new Chart(document.getElementById('holdHistChart'), {{
    type: 'bar',
    data: {{
      labels: hist.map(h => h.bucket),
      datasets: [{{
        label: 'Trades',
        data: hist.map(h => h.count),
        backgroundColor: 'rgba(128,203,196,0.7)',
      }}]
    }},
    options: {{ plugins: {{ legend: {{ display: false }} }} }},
  }});
}}

function renderEEMatrix() {{
  const m = DATA.ee_matrix || {{entries: [], exits: [], cells: {{}}}};
  if (!m.entries.length) {{ document.getElementById('eeMatrix').innerHTML = '<p style="color:#888">No closed trades.</p>'; return; }}
  let html = '<table class="matrix"><thead><tr><th>entry \\\\ exit</th>';
  for (const ex of m.exits) html += `<th style="text-align:right">${{ex}}</th>`;
  html += '<th style="text-align:right">total</th></tr></thead><tbody>';
  for (const en of m.entries) {{
    html += `<tr><td class="label">${{en}}</td>`;
    let rowN = 0, rowPnl = 0, rowWins = 0;
    for (const ex of m.exits) {{
      const c = m.cells[`${{en}}|${{ex}}`];
      if (!c) {{ html += '<td>·</td>'; continue; }}
      const wr = c.n > 0 ? c.wins / c.n : 0;
      const cls = c.pnl_tao > 0 ? 'cell-pos' : (c.pnl_tao < 0 ? 'cell-neg' : '');
      html += `<td class="${{cls}}">${{c.n}} · ${{(wr*100).toFixed(0)}}% · ${{c.pnl_tao.toFixed(3)}}T</td>`;
      rowN += c.n; rowPnl += c.pnl_tao; rowWins += c.wins;
    }}
    const rowWR = rowN > 0 ? rowWins / rowN : 0;
    const rowCls = rowPnl > 0 ? 'cell-pos' : (rowPnl < 0 ? 'cell-neg' : '');
    html += `<td class="${{rowCls}}"><b>${{rowN}} · ${{(rowWR*100).toFixed(0)}}% · ${{rowPnl.toFixed(3)}}T</b></td></tr>`;
  }}
  html += '</tbody></table>';
  document.getElementById('eeMatrix').innerHTML = html;
}}

function renderSlippage() {{
  new Chart(document.getElementById('slippageChart'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: 'Buy slippage',
        data: DATA.slippage.map(p => ({{ x: p.x, y: p.y, netuid: p.netuid }})),
        backgroundColor: '#80cbc4',
      }}]
    }},
    options: {{
      scales: {{
        x: {{ title: {{ display: true, text: 'Trade size (% of pool)' }} }},
        y: {{ title: {{ display: true, text: 'Slippage (%)' }} }},
      }},
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const p = ctx.raw;
              return `SN${{p.netuid}}: size ${{p.x.toFixed(2)}}%, slip ${{p.y.toFixed(3)}}%`;
            }}
          }}
        }}
      }},
    }},
  }});
}}

function renderPositions() {{
  if (!DATA.positions.length) return;
  const grouped = {{}};
  DATA.positions.forEach(p => {{
    grouped[p.netuid] = grouped[p.netuid] || [];
    grouped[p.netuid].push(p);
  }});
  const netuids = Object.keys(grouped).sort((a,b) => Number(a) - Number(b));
  const datasets = netuids.map(n => ({{
    label: `SN${{n}}`,
    data: grouped[n].map(p => ({{ x: [p.start, p.end], y: Number(n), pnl: p.pnl_pct, strategy: p.strategy }})),
    backgroundColor: grouped[n].map(p => p.pnl_pct >= 0 ? 'rgba(76,175,80,0.7)' : 'rgba(229,115,115,0.7)'),
    borderWidth: 0,
  }}));
  new Chart(document.getElementById('positionsChart'), {{
    type: 'bar',
    data: {{ datasets }},
    options: {{
      indexAxis: 'y',
      parsing: false,
      scales: {{
        x: {{ type: 'time', time: {{ parser: 'iso' }} }},
        y: {{ ticks: {{ callback: v => `SN${{v}}` }} }},
      }},
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const d = ctx.raw;
              return `SN${{d.y}} ${{d.strategy}}: pnl ${{d.pnl.toFixed(2)}}%`;
            }}
          }}
        }}
      }},
    }},
  }});
}}

renderBanner();
renderStats();
renderEquity();
renderRolling();
renderDrawdown();
renderPerSubnet();
renderPnlHist();
renderHoldHist();
renderEEMatrix();
renderSlippage();
renderPositions();
</script>
</body>
</html>
"""
    # Replace placeholder tokens in the summary subtitle (we used {{}} above
    # for Chart.js template literals, so we hand-inject here).
    html = html.replace("{summary.start_time}", str(result.start_time))
    html = html.replace("{summary.end_time}", str(result.end_time))
    html = html.replace("{summary.strategies}", ", ".join(result.strategies_used))
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def generate_paper_dashboard(
    portfolio: PortfolioTracker,
    current_snapshots: Optional[dict[int, Snapshot]] = None,
    pending_entries: Optional[list[Signal]] = None,
    trade_log_path: str = "data/paper_trades.json",
    output_path: str = "data/paper_dashboard.html",
) -> str:
    from datetime import timezone
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    now = datetime.now(timezone.utc)
    current_snapshots = current_snapshots or {}

    state = portfolio.get_state(now, current_snapshots)

    value_series = [
        {"t": s.timestamp.isoformat(), "v": s.total_value_tao}
        for s in portfolio.value_history
    ]

    positions = []
    for netuid, pos in state.positions.items():
        snap = current_snapshots.get(netuid)
        if snap and snap.tao_in > 0 and snap.alpha_in > 0:
            cur_price = snap.tao_in / snap.alpha_in
            pnl_tao = pos.unrealized_pnl_tao(snap.tao_in, snap.alpha_in)
            pnl_pct = pos.unrealized_pnl_pct(snap.tao_in, snap.alpha_in)
        else:
            cur_price = 0.0
            pnl_tao = 0.0
            pnl_pct = 0.0
        positions.append({
            "netuid": netuid,
            "strategy": pos.strategy.value,
            "entry_price": pos.entry_price,
            "current_price": cur_price,
            "alpha_amount": pos.alpha_amount,
            "tao_invested": pos.tao_invested,
            "pnl_tao": pnl_tao,
            "pnl_pct": pnl_pct,
            "hold_hours": pos.hold_duration_hours(now),
        })

    pending = []
    for sig in (pending_entries or [])[:25]:
        pending.append({
            "netuid": sig.netuid,
            "strategy": sig.strategy.value,
            "strength": sig.strength,
            "reason": sig.reason,
            "direction": sig.direction.value,
        })

    trade_rows = []
    for t in portfolio.trades[-50:][::-1]:
        trade_rows.append({
            "timestamp": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else t.timestamp,
            "netuid": t.netuid,
            "direction": t.direction.value,
            "strategy": t.strategy.value,
            "tao": t.tao_amount,
            "alpha": t.alpha_amount,
            "slip": t.slippage_pct,
            "pnl_pct": t.pnl_pct,
        })

    payload = {
        "now": now.isoformat(),
        "state": {
            "total_value_tao": state.total_value_tao,
            "free_tao": state.free_tao,
            "total_pnl_tao": state.total_pnl_tao,
            "total_pnl_pct": state.total_pnl_pct,
            "peak_value_tao": state.peak_value_tao,
            "drawdown_pct": state.drawdown_pct,
            "num_trades": state.num_trades,
        },
        "value_series": value_series,
        "positions": positions,
        "pending": pending,
        "trades": trade_rows,
    }

    data_json = json.dumps(payload, default=_json_default)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Paper Trading Dashboard</title>
<script src="{CHART_JS_CDN}"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0b0e14; color: #e0e6ed; margin: 0; padding: 20px; }}
  h1, h2 {{ color: #80cbc4; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .stat {{ background: #1a1f28; padding: 14px; border-radius: 6px; border-left: 3px solid #80cbc4; }}
  .stat .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
  .stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .chart-container {{ background: #1a1f28; padding: 14px; border-radius: 6px; margin-bottom: 18px; }}
  table {{ width: 100%; color: #e0e6ed; border-collapse: collapse; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #2a2f38; }}
  th {{ color: #80cbc4; text-align: left; }}
  .positive {{ color: #4caf50; }}
  .negative {{ color: #e57373; }}
</style>
</head>
<body>
<h1>Paper Trading Dashboard</h1>
<p style="color:#888">Updated: {now.isoformat()}</p>
<div class="grid" id="stats"></div>

<div class="chart-container">
  <h2>Portfolio Value</h2>
  <canvas id="valueChart"></canvas>
</div>

<div class="chart-container">
  <h2>Open Positions</h2>
  <div id="positions"></div>
</div>

<div class="chart-container">
  <h2>Pending Signals (not yet traded)</h2>
  <div id="pending"></div>
</div>

<div class="chart-container">
  <h2>Recent Trades</h2>
  <div id="trades"></div>
</div>

<script>
const DATA = {data_json};

function pct(x) {{ return (x*100).toFixed(2) + '%'; }}
function tao(x) {{ return Number(x).toFixed(3); }}
function cls(x) {{ return x >= 0 ? 'positive' : 'negative'; }}

function renderStats() {{
  const s = DATA.state;
  const stats = [
    {{ label: 'Portfolio Value', value: tao(s.total_value_tao) + ' TAO' }},
    {{ label: 'Free TAO',        value: tao(s.free_tao) }},
    {{ label: 'Total P&L',       value: tao(s.total_pnl_tao) + ' TAO', cls: cls(s.total_pnl_tao) }},
    {{ label: 'P&L %',           value: pct(s.total_pnl_pct), cls: cls(s.total_pnl_pct) }},
    {{ label: 'Peak',            value: tao(s.peak_value_tao) }},
    {{ label: 'Drawdown',        value: pct(s.drawdown_pct), cls: 'negative' }},
    {{ label: 'Trades',          value: s.num_trades }},
    {{ label: 'Open Positions',  value: DATA.positions.length }},
  ];
  document.getElementById('stats').innerHTML = stats.map(st =>
    `<div class="stat"><div class="label">${{st.label}}</div>`
    + `<div class="value ${{st.cls || ''}}">${{st.value}}</div></div>`
  ).join('');
}}

function renderValue() {{
  if (!DATA.value_series.length) return;
  new Chart(document.getElementById('valueChart'), {{
    type: 'line',
    data: {{
      labels: DATA.value_series.map(p => p.t),
      datasets: [{{
        label: 'Portfolio (TAO)',
        data: DATA.value_series.map(p => p.v),
        borderColor: '#80cbc4',
        backgroundColor: 'rgba(128,203,196,0.1)',
        fill: true,
        pointRadius: 0,
        tension: 0.1,
      }}],
    }},
    options: {{ scales: {{ x: {{ ticks: {{ autoSkip: true, maxTicksLimit: 10 }} }} }} }},
  }});
}}

function renderTable(root, rows, cols) {{
  const el = document.getElementById(root);
  if (!rows.length) {{ el.innerHTML = '<p style="color:#888">None.</p>'; return; }}
  const head = `<tr>${{cols.map(c => `<th>${{c.h}}</th>`).join('')}}</tr>`;
  const body = rows.map(r =>
    `<tr>${{cols.map(c => `<td style="text-align:${{c.a||'left'}}">${{c.fmt ? c.fmt(r) : r[c.k]}}</td>`).join('')}}</tr>`
  ).join('');
  el.innerHTML = `<table><thead>${{head}}</thead><tbody>${{body}}</tbody></table>`;
}}

renderStats();
renderValue();
renderTable('positions', DATA.positions, [
  {{ h: 'SN', k: 'netuid' }},
  {{ h: 'Strategy', k: 'strategy' }},
  {{ h: 'Entry', a: 'right', fmt: r => r.entry_price.toFixed(6) }},
  {{ h: 'Current', a: 'right', fmt: r => r.current_price.toFixed(6) }},
  {{ h: 'Alpha', a: 'right', fmt: r => r.alpha_amount.toFixed(2) }},
  {{ h: 'Invested (TAO)', a: 'right', fmt: r => r.tao_invested.toFixed(3) }},
  {{ h: 'P&L (TAO)', a: 'right', fmt: r => `<span class="${{r.pnl_tao>=0?'positive':'negative'}}">${{r.pnl_tao.toFixed(3)}}</span>` }},
  {{ h: 'P&L %', a: 'right', fmt: r => `<span class="${{r.pnl_pct>=0?'positive':'negative'}}">${{(r.pnl_pct*100).toFixed(2)}}%</span>` }},
  {{ h: 'Hold (h)', a: 'right', fmt: r => r.hold_hours.toFixed(1) }},
]);
renderTable('pending', DATA.pending, [
  {{ h: 'SN', k: 'netuid' }},
  {{ h: 'Direction', k: 'direction' }},
  {{ h: 'Strategy', k: 'strategy' }},
  {{ h: 'Strength', a: 'right', fmt: r => r.strength.toFixed(2) }},
  {{ h: 'Reason', k: 'reason' }},
]);
renderTable('trades', DATA.trades, [
  {{ h: 'Time', k: 'timestamp' }},
  {{ h: 'SN', k: 'netuid' }},
  {{ h: 'Dir', k: 'direction' }},
  {{ h: 'Strategy', k: 'strategy' }},
  {{ h: 'TAO', a: 'right', fmt: r => r.tao.toFixed(3) }},
  {{ h: 'Alpha', a: 'right', fmt: r => r.alpha.toFixed(2) }},
  {{ h: 'Slip', a: 'right', fmt: r => (r.slip*100).toFixed(3) + '%' }},
  {{ h: 'PnL %', a: 'right', fmt: r => r.pnl_pct == null ? '' : `<span class="${{r.pnl_pct>=0?'positive':'negative'}}">${{(r.pnl_pct*100).toFixed(2)}}%</span>` }},
]);
</script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def open_dashboard(path: str) -> None:
    webbrowser.open(f"file://{os.path.abspath(path)}")
