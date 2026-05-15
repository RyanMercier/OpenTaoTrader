"""Multi-config comparison dashboard.

Takes a list of backtest result JSONs and produces a single HTML page with
a dropdown to switch between configs. All data is embedded inline.

Use via the CLI:
    python -m trading.cli compare --summary results/compare/summary.json --open
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from .dashboard import (
    CHART_JS_CDN,
    _compute_benchmark,
    _entry_exit_matrix,
    _hold_histogram,
    _json_default,
    _per_subnet_pnl,
    _rolling_returns,
    _trade_histogram,
)


def _trade_from_dict(t: dict):
    """Build a lightweight object with attribute access for helper functions."""
    from types import SimpleNamespace
    from .models import Direction, StrategyName
    ns = SimpleNamespace(**t)
    # Coerce enums where helpers expect `.value`
    if isinstance(ns.direction, str):
        ns.direction = Direction(ns.direction)
    if isinstance(ns.strategy, str):
        ns.strategy = StrategyName(ns.strategy)
    if isinstance(getattr(ns, "entry_strategy", None), str):
        ns.entry_strategy = StrategyName(ns.entry_strategy)
    if isinstance(getattr(ns, "timestamp", None), str):
        try:
            ns.timestamp = datetime.fromisoformat(ns.timestamp)
        except Exception:
            pass
    return ns


def _config_payload(result_path: str, label: str) -> dict:
    """Build the per-config payload the JS layer expects."""
    with open(result_path, "r") as f:
        d = json.load(f)

    cfg = d.get("config") or {}
    db_path = cfg.get("db_path") if isinstance(cfg, dict) else None

    # Rehydrate trades into objects for the helper functions
    trades = [_trade_from_dict(t) for t in d.get("trades", [])]
    from .models import Direction
    sells = [t for t in trades if t.direction == Direction.BUY.__class__.SELL] if trades else []
    # fix: simpler comparison
    sells = [t for t in trades if t.direction == Direction.SELL]

    initial = d["initial_capital"]
    # Benchmark (equal-weight B&H across in-scope subnets)
    benchmark_ratios = _compute_benchmark(db_path, d["start_time"], d["end_time"]) if db_path else []
    benchmark_equity = [{"t": b["t"], "v": b["v"] * initial} for b in benchmark_ratios]

    bh_final = benchmark_equity[-1]["v"] if benchmark_equity else initial
    bh_return_pct = (bh_final / initial - 1) if initial > 0 else 0.0

    rolling_strategy = _rolling_returns(d["portfolio_values"], window_days=30)
    rolling_benchmark = _rolling_returns(
        [(b["t"], b["v"] * initial) for b in benchmark_ratios], window_days=30
    )

    per_subnet = _per_subnet_pnl(trades)
    pnl_hist = _trade_histogram(
        sells, [-0.05, -0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03, 0.05]
    )
    hold_hist = _hold_histogram(sells)
    ee_matrix = _entry_exit_matrix(sells)

    equity_data = [{"t": ts, "v": v} for ts, v in d["portfolio_values"]]
    drawdown_data = [{"t": ts, "v": v * 100} for ts, v in d["drawdown_series"]]

    trade_markers = []
    for t in trades:
        trade_markers.append({
            "t": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else t.timestamp,
            "netuid": t.netuid,
            "direction": t.direction.value,
            "strategy": t.strategy.value,
            "entry_strategy": t.entry_strategy.value if getattr(t, "entry_strategy", None) else None,
            "tao": t.tao_amount,
            "alpha": t.alpha_amount,
            "slip": t.slippage_pct,
            "pnl_pct": t.pnl_pct,
            "pnl_tao": t.pnl_tao,
            "hold_hours": t.hold_duration_hours,
        })

    # Position timeline
    position_timeline = []
    open_entries: dict = {}
    for t in trades:
        if t.direction == Direction.BUY:
            open_entries[t.netuid] = t
        elif t.direction == Direction.SELL and t.netuid in open_entries:
            entry = open_entries.pop(t.netuid)
            position_timeline.append({
                "netuid": t.netuid,
                "start": entry.timestamp.isoformat() if hasattr(entry.timestamp, "isoformat") else entry.timestamp,
                "end": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else t.timestamp,
                "pnl_pct": (t.pnl_pct or 0.0) * 100,
                "strategy": entry.strategy.value,
            })

    # Slippage scatter
    slippage_scatter = []
    for t in trades:
        if t.direction == Direction.BUY and t.tao_amount > 0 and t.slippage_pct and t.slippage_pct > 0:
            slippage_scatter.append({
                "x": t.slippage_pct * 100,
                "y": t.slippage_pct * 100,
                "netuid": t.netuid,
            })

    return {
        "label": label,
        "summary": {
            "start_time": d["start_time"],
            "end_time": d["end_time"],
            "initial_capital": initial,
            "final_value": d["final_value"],
            "total_return_pct": d["total_return_pct"],
            "annualized_return_pct": d["annualized_return_pct"],
            "max_drawdown_pct": d["max_drawdown_pct"],
            "sharpe": d["sharpe_ratio"],
            "sortino": d["sortino_ratio"],
            "total_trades": d["total_trades"],
            "win_rate": d["win_rate"],
            "avg_hold_hours": d["avg_hold_hours"],
            "total_slippage_cost": d["total_slippage_cost_tao"],
            "strategies": d["strategies_used"],
            "blocked_trades": d["blocked_trades"],
            "benchmark_return_pct": bh_return_pct,
            "alpha_vs_benchmark": d["total_return_pct"] - bh_return_pct,
            "num_hotkeys": cfg.get("num_hotkeys") if isinstance(cfg, dict) else None,
            "max_positions": cfg.get("max_positions") if isinstance(cfg, dict) else None,
            "max_single_position_pct": cfg.get("max_single_position_pct") if isinstance(cfg, dict) else None,
            "reserve_pct": cfg.get("reserve_pct") if isinstance(cfg, dict) else None,
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
        "monthly_returns": d.get("monthly_returns", {}),
    }


def generate_comparison_dashboard(
    summary_path: str = "results/compare/summary.json",
    output_path: str = "results/comparison_dashboard.html",
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    configs = [_config_payload(c["path"], c["label"]) for c in summary]

    # Summary leaderboard
    leaderboard = [{
        "id": i,
        "label": c["label"],
        "return_pct": c["summary"]["total_return_pct"],
        "annualized": c["summary"]["annualized_return_pct"],
        "sharpe": c["summary"]["sharpe"],
        "max_dd": c["summary"]["max_drawdown_pct"],
        "win_rate": c["summary"]["win_rate"],
        "trades": c["summary"]["total_trades"],
        "alpha": c["summary"]["alpha_vs_benchmark"],
    } for i, c in enumerate(configs)]

    data_json = json.dumps({"configs": configs, "leaderboard": leaderboard}, default=_json_default)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bittensor Trading, Strategy Comparison</title>
<script src="__CDN__"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0b0e14; color: #e0e6ed; margin: 0; padding: 20px; max-width: 1600px; }
  h1, h2 { color: #80cbc4; }
  h2 { font-size: 18px; margin-top: 0; }
  .selector-bar { background: #1a1f28; padding: 14px 18px; border-radius: 6px;
                  margin-bottom: 18px; display: flex; align-items: center; gap: 14px; }
  .selector-bar select { background: #0b0e14; color: #e0e6ed; border: 1px solid #2a2f38;
                         padding: 10px 14px; border-radius: 4px; font-size: 15px;
                         font-family: inherit; min-width: 340px; }
  .banner { background: linear-gradient(90deg, #0f3b36 0%, #122228 100%);
            padding: 16px 20px; border-radius: 6px; margin-bottom: 18px;
            display: flex; justify-content: space-between; align-items: center; }
  .banner .hl { font-size: 26px; font-weight: 600; color: #80cbc4; }
  .banner .sub { color: #aaa; margin-top: 4px; font-size: 13px; }
  .banner code { background: #0b0e14; padding: 6px 10px; border-radius: 4px;
                 font-family: 'SF Mono', Menlo, monospace; color: #80cbc4; font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 18px; }
  .stat { background: #1a1f28; padding: 12px; border-radius: 6px; border-left: 3px solid #80cbc4; }
  .stat .label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat .value { font-size: 20px; font-weight: 600; margin-top: 4px; }
  .stat .sub { font-size: 11px; color: #888; margin-top: 2px; }
  .stat .good { color: #4caf50; }
  .stat .bad { color: #e57373; }
  .chart-container { background: #1a1f28; padding: 14px; border-radius: 6px; margin-bottom: 14px; }
  canvas { max-height: 360px; }
  table { width: 100%; color: #e0e6ed; border-collapse: collapse; }
  th, td { padding: 5px 10px; border-bottom: 1px solid #2a2f38; font-size: 13px; }
  th { color: #80cbc4; text-align: left; cursor: pointer; user-select: none; }
  tr.active { background: #0f3b36; }
  tr:hover { background: #12202a; cursor: pointer; }
  .cols2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .matrix td { text-align: right; padding: 6px 10px; }
  .matrix td.label { text-align: left; color: #80cbc4; font-weight: 600; }
  .cell-pos { background: rgba(76,175,80,0.18); }
  .cell-neg { background: rgba(229,115,115,0.18); }
  .num-pos { color: #4caf50; }
  .num-neg { color: #e57373; }
</style>
</head>
<body>
<h1>Strategy Comparison, Bittensor Subnet Trading</h1>

<div class="selector-bar">
  <label>Config:</label>
  <select id="configSelect"></select>
  <span style="color:#888">Click a row in the leaderboard to switch. Currently viewing: <b id="currentLabel">-</b></span>
</div>

<div class="chart-container">
  <h2>Leaderboard (all configs, 3 TAO, same data window)</h2>
  <table id="leaderboard">
    <thead><tr>
      <th>#</th><th>Config</th>
      <th style="text-align:right">Return</th>
      <th style="text-align:right">Annualized</th>
      <th style="text-align:right">Sharpe</th>
      <th style="text-align:right">Max DD</th>
      <th style="text-align:right">Win%</th>
      <th style="text-align:right">Trades</th>
      <th style="text-align:right">vs B&H</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="banner">
  <div>
    <div class="hl" id="bannerReturn"></div>
    <div class="sub" id="bannerSub"></div>
  </div>
  <div style="text-align:right">
    <div class="sub">Deploy:</div>
    <code id="deployCmd"></code>
  </div>
</div>

<div class="grid" id="stats"></div>

<div class="chart-container">
  <h2>Equity Curve vs Equal-Weight B&H</h2>
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
    <div id="monthlyTable"></div>
  </div>
</div>

<div class="chart-container">
  <h2>Entry -> Exit Attribution</h2>
  <div id="eeMatrix"></div>
  <p style="color:#888;font-size:12px;margin-top:6px">
    Rows = entry strategy, columns = exit strategy. Cell: trade count · win rate · total P&L TAO.
  </p>
</div>

<div class="chart-container">
  <h2>Position Timeline</h2>
  <canvas id="positionsChart"></canvas>
</div>

<script>
const STATE = __DATA__;
const COLORS = ['#80cbc4','#ffb74d','#ba68c8','#aed581','#4fc3f7','#e57373','#f06292','#dce775'];

let currentIdx = 0;
let charts = {};

function pct(x) { return (x*100).toFixed(2) + '%'; }
function taoStr(x) { return Number(x).toFixed(3); }

function destroyCharts() {
  for (const k of Object.keys(charts)) {
    try { charts[k].destroy(); } catch(e){}
    delete charts[k];
  }
}

function renderLeaderboard() {
  const tbody = document.querySelector('#leaderboard tbody');
  tbody.innerHTML = STATE.leaderboard.map((r, i) => {
    const retCls = r.return_pct >= 0 ? 'num-pos' : 'num-neg';
    const aCls = r.alpha >= 0 ? 'num-pos' : 'num-neg';
    return `<tr data-idx="${i}" ${i===currentIdx?'class="active"':''}>
      <td>${i+1}</td>
      <td>${r.label}</td>
      <td style="text-align:right" class="${retCls}">${pct(r.return_pct)}</td>
      <td style="text-align:right">${pct(r.annualized)}</td>
      <td style="text-align:right">${r.sharpe.toFixed(2)}</td>
      <td style="text-align:right" class="num-neg">${pct(r.max_dd)}</td>
      <td style="text-align:right">${pct(r.win_rate)}</td>
      <td style="text-align:right">${r.trades}</td>
      <td style="text-align:right" class="${aCls}">${pct(r.alpha)}</td>
    </tr>`;
  }).join('');
  tbody.querySelectorAll('tr').forEach(row => {
    row.addEventListener('click', () => selectConfig(Number(row.dataset.idx)));
  });
}

function renderSelector() {
  const sel = document.getElementById('configSelect');
  sel.innerHTML = STATE.configs.map((c, i) => `<option value="${i}">${c.label}</option>`).join('');
  sel.addEventListener('change', () => selectConfig(Number(sel.value)));
}

function selectConfig(idx) {
  currentIdx = idx;
  document.getElementById('configSelect').value = idx;
  document.querySelectorAll('#leaderboard tr').forEach((tr, i) => {
    tr.classList.toggle('active', Number(tr.dataset.idx) === idx);
  });
  document.getElementById('currentLabel').textContent = STATE.configs[idx].label;
  destroyCharts();
  renderAll();
}

function renderBanner() {
  const d = STATE.configs[currentIdx];
  const s = d.summary;
  const ret = s.total_return_pct;
  const alpha = s.alpha_vs_benchmark;
  const alphaStr = alpha != null ? `${(alpha*100).toFixed(2)}%` : 'n/a';
  const bnh = s.benchmark_return_pct != null ? `${(s.benchmark_return_pct*100).toFixed(2)}%` : 'n/a';
  const days = Math.round((new Date(s.end_time) - new Date(s.start_time)) / 86400000);
  document.getElementById('bannerReturn').innerHTML =
    `<span class="${ret>=0?'good':'bad'}">${(ret*100).toFixed(2)}%</span>` +
    ` over ${days}d (annualized ${(s.annualized_return_pct*100).toFixed(1)}%, Sharpe ${s.sharpe.toFixed(2)})`;
  const alphaClass = alpha >= 0 ? 'good' : 'bad';
  document.getElementById('bannerSub').innerHTML =
    `vs B&amp;H ${bnh} -> <span class="${alphaClass}">alpha ${alphaStr}</span>` +
    ` · max DD ${pct(s.max_drawdown_pct)} · ${s.total_trades} trades, ${pct(s.win_rate)} win rate`;
  const strats = (s.strategies || []).filter(x => x !== 'drain_exit').join(',');
  const hk = s.num_hotkeys || 2;
  document.getElementById('deployCmd').textContent =
    `python -m trading.cli paper --capital ${s.initial_capital} --strategies ${strats || 'momentum,stmc'} --hotkeys ${hk}`;
}

function renderStats() {
  const s = STATE.configs[currentIdx].summary;
  const stats = [
    { label: 'Return',      value: pct(s.total_return_pct), good: s.total_return_pct>=0 },
    { label: 'Annualized',  value: pct(s.annualized_return_pct), good: s.annualized_return_pct>=0 },
    { label: 'vs B&H',      value: pct(s.alpha_vs_benchmark), good: s.alpha_vs_benchmark>=0,
                            sub: `bench ${pct(s.benchmark_return_pct)}` },
    { label: 'Final Value', value: taoStr(s.final_value) + ' TAO',
                            sub: `+${taoStr(s.final_value - s.initial_capital)}` },
    { label: 'Sharpe',      value: s.sharpe.toFixed(2) },
    { label: 'Sortino',     value: s.sortino.toFixed(2) },
    { label: 'Max DD',      value: pct(s.max_drawdown_pct), good: false },
    { label: 'Win Rate',    value: pct(s.win_rate), sub: `${s.total_trades} trades` },
    { label: 'Avg Hold',    value: s.avg_hold_hours.toFixed(1) + 'h' },
    { label: 'Slip Cost',   value: taoStr(s.total_slippage_cost) + ' TAO' },
    { label: 'Blocked',     value: s.blocked_trades, sub: 'rate-limit skips' },
    { label: 'Sizing',      value: (s.max_single_position_pct? (s.max_single_position_pct*100).toFixed(0)+'%': '?'),
                            sub: `${s.num_hotkeys||'?'}hk, ${s.max_positions||'?'}pos` },
  ];
  document.getElementById('stats').innerHTML = stats.map(st => {
    const cls = st.good === true ? ' good' : (st.good === false ? ' bad' : '');
    const sub = st.sub ? `<div class="sub">${st.sub}</div>` : '';
    return `<div class="stat"><div class="label">${st.label}</div><div class="value${cls}">${st.value}</div>${sub}</div>`;
  }).join('');
}

function renderEquity() {
  const d = STATE.configs[currentIdx];
  const initial = d.summary.initial_capital;
  const buys = d.trades.filter(t => t.direction === 'buy');
  const sells = d.trades.filter(t => t.direction === 'sell');
  charts.equity = new Chart(document.getElementById('equityChart'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'Strategy', data: d.equity.map(p => ({x:p.t,y:p.v})),
          borderColor: '#80cbc4', backgroundColor:'rgba(128,203,196,0.1)', fill:true, pointRadius:0, tension:0.1, parsing:false },
        { label: 'Equal-weight B&H', data: (d.benchmark||[]).map(b=>({x:b.t,y:b.v})),
          borderColor:'#ffb74d', borderDash:[4,4], pointRadius:0, fill:false, parsing:false },
        { label: 'Buys', type:'scatter', data: buys.map(b=>({x:b.t,y:initial})),
          backgroundColor:'#4caf50', pointStyle:'triangle', pointRadius:4 },
        { label: 'Sells', type:'scatter', data: sells.map(s=>({x:s.t,y:initial})),
          backgroundColor:'#e57373', pointStyle:'triangle', rotation:180, pointRadius:4 },
      ]
    },
    options: {
      scales:{ x:{ type:'time', time:{parser:'iso'}, ticks:{autoSkip:true, maxTicksLimit:12} } },
      plugins:{
        tooltip:{
          callbacks:{
            label:(ctx)=>{
              if (ctx.dataset.type==='scatter'){
                const arr = ctx.dataset.label==='Buys'?buys:sells;
                const src = arr[ctx.dataIndex];
                const pnl = src.pnl_pct!=null?` pnl ${(src.pnl_pct*100).toFixed(2)}%`:'';
                const e = src.entry_strategy?`${src.entry_strategy}->${src.strategy}`:src.strategy;
                return `${ctx.dataset.label} SN${src.netuid} [${e}]${pnl}`;
              }
              return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)} TAO`;
            }
          }
        }
      }
    }
  });
}

function renderRolling() {
  const d = STATE.configs[currentIdx];
  charts.rolling = new Chart(document.getElementById('rollingChart'), {
    type:'line', data:{ datasets:[
      { label:'Strategy 30d', data:(d.rolling_strategy||[]).map(p=>({x:p.t,y:p.v})),
        borderColor:'#80cbc4', pointRadius:0, tension:0.1, parsing:false },
      { label:'B&H 30d', data:(d.rolling_benchmark||[]).map(p=>({x:p.t,y:p.v})),
        borderColor:'#ffb74d', borderDash:[4,4], pointRadius:0, tension:0.1, parsing:false },
    ]},
    options:{ scales:{ x:{type:'time', time:{parser:'iso'}, ticks:{autoSkip:true, maxTicksLimit:8}},
                      y:{title:{display:true, text:'Trailing 30d return (%)'}} } }
  });
}

function renderDrawdown() {
  const d = STATE.configs[currentIdx];
  charts.drawdown = new Chart(document.getElementById('drawdownChart'), {
    type:'line', data:{ datasets:[
      { label:'Drawdown (%)', data:d.drawdown.map(p=>({x:p.t,y:p.v})),
        borderColor:'#e57373', backgroundColor:'rgba(229,115,115,0.25)', fill:true, pointRadius:0, tension:0.1, parsing:false }
    ]},
    options:{ scales:{ x:{type:'time', time:{parser:'iso'}, ticks:{autoSkip:true, maxTicksLimit:8}} } }
  });
}

function renderPerSubnet() {
  const d = STATE.configs[currentIdx];
  const datasets = Object.entries(d.per_subnet||{}).map(([n, pts], i)=>({
    label:`SN${n}`, data:pts.map(p=>({x:p.t,y:p.v})),
    borderColor: COLORS[i%COLORS.length], pointRadius:0, tension:0.1, parsing:false, fill:false
  }));
  charts.perSubnet = new Chart(document.getElementById('perSubnetChart'), {
    type:'line', data:{datasets},
    options:{
      scales:{ x:{type:'time', time:{parser:'iso'}, ticks:{autoSkip:true, maxTicksLimit:12}},
               y:{title:{display:true, text:'Cumulative realized P&L (TAO)'}} }
    }
  });
}

function renderPnlHist() {
  const d = STATE.configs[currentIdx];
  const hist = d.pnl_hist||[];
  charts.pnl = new Chart(document.getElementById('pnlHistChart'), {
    type:'bar',
    data:{ labels:hist.map(h=>h.bucket), datasets:[{
      label:'Trades', data:hist.map(h=>h.count),
      backgroundColor:hist.map(h=>h.bucket.startsWith('<')||h.bucket.startsWith('-')?'rgba(229,115,115,0.7)':'rgba(76,175,80,0.7)')
    }]},
    options:{ scales:{x:{ticks:{maxRotation:45,minRotation:45}}}, plugins:{legend:{display:false}} }
  });
}

function renderHoldHist() {
  const d = STATE.configs[currentIdx];
  const hist = d.hold_hist||[];
  charts.hold = new Chart(document.getElementById('holdHistChart'), {
    type:'bar',
    data:{ labels:hist.map(h=>h.bucket), datasets:[{label:'Trades', data:hist.map(h=>h.count), backgroundColor:'rgba(128,203,196,0.7)'}]},
    options:{ plugins:{legend:{display:false}} }
  });
}

function renderMonthly() {
  const mr = STATE.configs[currentIdx].monthly_returns || {};
  const months = Object.keys(mr).sort();
  if (!months.length) { document.getElementById('monthlyTable').innerHTML='<p style="color:#888">No data.</p>'; return; }
  let html = '<table><thead><tr><th>Month</th><th style="text-align:right">Return</th></tr></thead><tbody>';
  for (const m of months){
    const v = mr[m]*100;
    const color = v>0?'#1b5e20':(v<0?'#b71c1c':'#555');
    html += `<tr><td>${m}</td><td style="color:#fff;background:${color};text-align:right;padding:4px 12px">${v>=0?'+':''}${v.toFixed(2)}%</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('monthlyTable').innerHTML = html;
}

function renderEEMatrix() {
  const m = STATE.configs[currentIdx].ee_matrix || {entries:[], exits:[], cells:{}};
  if (!m.entries.length){ document.getElementById('eeMatrix').innerHTML='<p style="color:#888">No closed trades.</p>'; return; }
  let html = '<table class="matrix"><thead><tr><th>entry \\ exit</th>';
  for (const ex of m.exits) html += `<th style="text-align:right">${ex}</th>`;
  html += '<th style="text-align:right">total</th></tr></thead><tbody>';
  for (const en of m.entries){
    html += `<tr><td class="label">${en}</td>`;
    let rowN=0, rowPnl=0, rowWins=0;
    for (const ex of m.exits){
      const c = m.cells[`${en}|${ex}`];
      if (!c) { html += '<td>·</td>'; continue; }
      const wr = c.n>0?c.wins/c.n:0;
      const cls = c.pnl_tao>0?'cell-pos':(c.pnl_tao<0?'cell-neg':'');
      html += `<td class="${cls}">${c.n} · ${(wr*100).toFixed(0)}% · ${c.pnl_tao.toFixed(3)}T</td>`;
      rowN+=c.n; rowPnl+=c.pnl_tao; rowWins+=c.wins;
    }
    const rowWR = rowN>0?rowWins/rowN:0;
    const rowCls = rowPnl>0?'cell-pos':(rowPnl<0?'cell-neg':'');
    html += `<td class="${rowCls}"><b>${rowN} · ${(rowWR*100).toFixed(0)}% · ${rowPnl.toFixed(3)}T</b></td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('eeMatrix').innerHTML = html;
}

function renderPositions() {
  const d = STATE.configs[currentIdx];
  if (!d.positions.length) return;
  const grouped = {};
  d.positions.forEach(p=>{ grouped[p.netuid]=grouped[p.netuid]||[]; grouped[p.netuid].push(p); });
  const netuids = Object.keys(grouped).sort((a,b)=>Number(a)-Number(b));
  const datasets = netuids.map(n=>({
    label:`SN${n}`,
    data: grouped[n].map(p=>({x:[p.start,p.end], y:Number(n), pnl:p.pnl_pct, strategy:p.strategy})),
    backgroundColor: grouped[n].map(p=>p.pnl_pct>=0?'rgba(76,175,80,0.7)':'rgba(229,115,115,0.7)'),
    borderWidth:0
  }));
  charts.positions = new Chart(document.getElementById('positionsChart'), {
    type:'bar', data:{datasets},
    options:{
      indexAxis:'y', parsing:false,
      scales:{x:{type:'time', time:{parser:'iso'}}, y:{ticks:{callback:v=>`SN${v}`}}},
      plugins:{ tooltip:{callbacks:{ label:(ctx)=>{ const d=ctx.raw; return `SN${d.y} ${d.strategy}: pnl ${d.pnl.toFixed(2)}%`; }}} }
    }
  });
}

function renderAll() {
  renderBanner();
  renderStats();
  renderEquity();
  renderRolling();
  renderDrawdown();
  renderPerSubnet();
  renderPnlHist();
  renderHoldHist();
  renderMonthly();
  renderEEMatrix();
  renderPositions();
}

// Default selection: find the "BEST" tagged config if present, else idx 0
let defaultIdx = 0;
STATE.configs.forEach((c, i) => { if (c.label.includes('[BEST]')) defaultIdx = i; });
currentIdx = defaultIdx;

renderSelector();
renderLeaderboard();
document.getElementById('configSelect').value = currentIdx;
document.getElementById('currentLabel').textContent = STATE.configs[currentIdx].label;
renderAll();
</script>
</body>
</html>
"""
    html = html.replace("__CDN__", CHART_JS_CDN).replace("__DATA__", data_json)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path
