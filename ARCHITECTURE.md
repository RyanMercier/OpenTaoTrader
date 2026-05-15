# Architecture

OpenTaoTrader is a downstream consumer of OpenTaoAPI. The data layer
(subnet snapshots, live ticks, pool state, the live demo dashboard)
lives in OpenTaoAPI. The trader reads from it over HTTP/SSE, holds its
own SQLite for trading state, and submits chain extrinsics directly
when in live mode.

```
   +-------------------+               +--------------------+
   |   OpenTaoAPI      |  HTTP / SSE   |   OpenTaoTrader    |
   |   (data layer)    | <--- reads -- |   server + CLI     |
   +---------+---------+               +--------+-----------+
             |                                  |
             v                                  v
   +-------------------+               +--------------------+
   |  opentao.db       |               | opentaotrader.db   |
   |  subnet_snapshots |               | paper_portfolios   |
   |  + webhooks etc.  |               | + positions/trades |
   +-------------------+               +--------------------+

   Live CLI keeps its own AsyncSubtensor for signing add_stake /
   unstake. Server process never sees a coldkey.
```

## Why split

Two repos = two pitches. OpenTaoAPI is the public data layer that
anyone can self-host or hit via the demo; OpenTaoTrader is the agent
layer that consumes it. The trader can run against any OpenTaoAPI
instance (your own self-hosted node, the public demo, or a colleague's
deployment) without coupling its release cycle to the API's.

## Snapshot seeding + SSE buffer

At startup `OpenTaoAPIClient.seed_snapshots(hours=720)` fetches the
last 30 days of history for every active subnet. Then `start_sse()`
spawns a background task that consumes `/api/v1/stream` and folds each
event into a per-netuid rolling buffer. The fold respects 30-minute
bar boundaries: events inside the current bar overwrite the last
sample, new bars append. This gives the strategy engine sub-second
fresh data without ever touching the chain.

`run_once` snapshots the buffer under a lock before computing
features, so a concurrent SSE write doesn't tear a half-updated
history.

## Stale upstream gating

Before each cycle the runner calls `OpenTaoAPIClient.is_stale()`,
which checks `/health`. If the upstream poller is behind (the API's
own snapshot pipeline is stuck) the trader skips the cycle entirely.
Trading on dead data is worse than skipping a cycle.

## Live trading boundary

The server process is paper-only. The live runner is launched via
`python -m trading.cli live`, lives in its own process, loads the
wallet from `~/.bittensor/wallets/<name>/`, prompts for the coldkey
password on stdin, then loops calling `LiveTrader.run_once()`.

`LiveTrader` keeps two clients side by side:

- `api_client` (HTTP+SSE to OpenTaoAPI) for snapshot history and
  feature input. Same buffer the paper trader uses.
- `chain_client` (AsyncSubtensor) for the pre-flight pool refresh, the
  coldkey balance check, the stake-before / stake-after delta, and
  the actual `add_stake` / `unstake` submission. Latency matters at
  trade time, so we go direct.

Both processes write to the same `opentaotrader.db`. SQLite's file
lock + aiosqlite's per-connection `asyncio.Lock` keep concurrent
writes safe.

## Daily kill-switch

Every realized sell evaluates `_check_kill_switch`. If
`(start_of_day_value - current_value) / start_of_day_value` exceeds
`daily_loss_limit_pct`, the trader flips the portfolio inactive in
the DB (`active=0`) and refuses further entries. The operator has to
explicitly resume. Reset baseline at UTC midnight.

## Backtester

Backtests don't go over HTTP. They use `DataLoader(db_path)` to read
`subnet_snapshots` straight from OpenTaoAPI's SQLite on the same
disk. Pulling 30 days of full history for 100+ subnets over HTTP would
be slow and pointless when the file is sitting right there. The
trader's `--db` flag (or `OPENTAO_DB_PATH`) points it at the API's DB.
