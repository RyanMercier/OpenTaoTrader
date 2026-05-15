# OpenTaoTrader

Paper and live trading runner for Bittensor subnet alpha tokens. Reads
chain data from a running [OpenTaoAPI](https://github.com/ryanmercier/OpenTaoAPI)
instance, runs the same signal pipeline whether you're paper or live,
and writes its own SQLite of portfolios, positions, trades, and
value-history for the dashboard.

> Built solo by Ryan Mercier. Sister repo to OpenTaoAPI.

## What it does

- **Paper trading** against the upstream API's live snapshot feed.
  AMM-aware: constant-product slippage on every trade, per-hotkey rate
  limit enforcement (1 stake per 360 blocks), zero-lookahead causal
  feature computation.
- **Live trading** from the CLI. Loads a Bittensor wallet on disk,
  prompts for the coldkey password on stdin, submits `add_stake` /
  `unstake` extrinsics. Keys never enter the server process.
- **Pluggable strategies.** Drop a Python file anywhere, decorate the
  class with `@register_strategy("name")`, point
  `OPENTAO_EXTERNAL_STRATEGIES` at it. The runner picks it up alongside
  the four built-ins (`stake_velocity`, `mean_reversion`, `momentum`,
  `drain_exit`).
- **Same dashboard for both modes.** Equity curve, pool-weighted
  buy-and-hold benchmark, drawdown, win rate, Sharpe, Sortino, profit
  factor. A LIVE badge marks live portfolios.

## Quick start

You need an OpenTaoAPI instance running somewhere. Either run it locally
or point at the hosted demo at `https://opentao.rpmsystems.io`.

```bash
conda create -n taotrader python=3.11 -y
conda activate taotrader
pip install -r requirements.txt

# Point at an OpenTaoAPI instance:
export OPENTAO_API_URL=https://opentao.rpmsystems.io

# Run the server. Dashboard at http://localhost:8009/paper
uvicorn server.main:app --host 0.0.0.0 --port 8009
```

Set `PAPER_TRADING_ENABLED=true` to actually advance paper portfolios on
their configured cadence. Read endpoints work either way so you can
browse the dashboard without running anyone's bot.

## Create a paper portfolio

```bash
curl -X POST http://localhost:8009/api/v1/paper/portfolios \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "demo",
    "initial_capital_tao": 100,
    "strategies": ["mean_reversion", "momentum"],
    "poll_interval_seconds": 1800
  }'
```

Then visit `http://localhost:8009/paper/1` to see the equity curve,
positions, trades, and stats.

## Live trading

CLI-only, by design. The server process never sees a coldkey.

```bash
# Create the portfolio first via the web UI or API (it starts in paper mode).
curl -X POST http://localhost:8009/api/v1/paper/portfolios \
  -H 'Content-Type: application/json' \
  -d '{"name": "live-demo", "initial_capital_tao": 10, "strategies": ["mean_reversion"]}'

# Promote it to live and run the loop. Password prompted on stdin.
python -m trading.cli live \
  --portfolio live-demo \
  --wallet my_wallet \
  --hotkey default
```

Pre-flight fires on every trade: re-fetches pool reserves, re-checks
slippage at the actual current state, verifies free balance. A daily
kill-switch trips the portfolio inactive if intraday loss breaches
`daily_loss_limit_pct`. Use `--dry-run` to test the pipeline without
submitting extrinsics.

## Plugin strategies

Built-ins live in `trading/strategies/`. The simplest is
[mean_reversion.py](trading/strategies/mean_reversion.py); copy it as a
starting point. The registry contract is:

```python
from trading.models import Direction, Signal, StrategyName
from trading.strategies import register_strategy
from trading.strategies.base import Strategy

@register_strategy("buy_dips")
class BuyDipsStrategy(Strategy):
    """Buy on strong negative price momentum, hold 24h."""

    def name(self): return StrategyName.EXTERNAL

    def generate_entry_signal(self, netuid, features, snapshot):
        pm = features.price_momentum_24h
        if pm is None or pm > -0.05:
            return None
        return Signal(
            timestamp=snapshot.timestamp, netuid=netuid,
            direction=Direction.BUY, strategy=self.name(),
            strength=min(abs(pm) / 0.10, 1.0),
            reason=f"24h dip {pm * 100:.1f}% on SN{netuid}",
            features=features.to_dict(),
        )

    def generate_exit_signal(self, netuid, features, snapshot, position):
        if position.hold_duration_hours(snapshot.timestamp) >= 24:
            return Signal(
                timestamp=snapshot.timestamp, netuid=netuid,
                direction=Direction.SELL, strategy=StrategyName.HOLD_TIMEOUT,
                strength=1.0, reason="24h hold", features=features.to_dict(),
            )
        return None
```

```bash
OPENTAO_EXTERNAL_STRATEGIES=/opt/my_strategies uvicorn server.main:app
```

The "create portfolio" form picks up the new key automatically.

## Backtesting

The backtester reads subnet snapshots straight from OpenTaoAPI's SQLite
for speed. Point at it via `--db`:

```bash
python -m trading.cli backtest \
  --db /path/to/OpenTaoAPI/data/opentao.db \
  --capital 100 \
  --strategies mean_reversion,momentum \
  --output results/run1.json
```

Generates a self-contained HTML dashboard with equity curve, monthly
heatmap, slippage analysis, trade attribution, and per-strategy P&L.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENTAO_API_URL` | `http://localhost:8000` | URL of the upstream OpenTaoAPI |
| `OPENTAO_DB_PATH` | _(empty)_ | Path to OpenTaoAPI's SQLite, for backtester |
| `DATABASE_PATH` | `data/opentaotrader.db` | Trader's own SQLite |
| `BITTENSOR_NETWORK` | `finney` | Network used by the live CLI |
| `SUBTENSOR_ENDPOINT` | _(empty)_ | Override websocket endpoint |
| `PAPER_TRADING_ENABLED` | `false` | Whether the runner advances cycles |
| `OPENTAO_EXTERNAL_STRATEGIES` | _(empty)_ | Colon-separated paths to strategy files |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8009` | Bind address |

## How it talks to OpenTaoAPI

At boot the trader hits `GET /api/v1/history/{netuid}/snapshots?hours=720`
once per active subnet to seed feature history, then subscribes to
`GET /api/v1/stream` over SSE for live updates. The benchmark series
uses the same `/history` endpoint. Live trading uses Bittensor's
AsyncSubtensor directly for pre-flight pool refresh and extrinsic
submission, since latency matters there. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## License

MIT
