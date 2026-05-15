"""Trader runtime config. Pulled from environment via pydantic-settings.

The trader is a downstream consumer of OpenTaoAPI. Almost every piece of
data it needs (subnet snapshots, current pool state, live ticks) comes
from the API over HTTP/SSE. The exceptions are the live-trading code path
(it talks to chain directly to sign extrinsics) and the backtester (it
reads snapshots straight from the API's SQLite for speed).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # URL of the OpenTaoAPI instance this trader reads from.
    opentao_api_url: str = "http://localhost:8000"

    # Path to OpenTaoAPI's SQLite, used by the backtester for fast bulk
    # reads of subnet_snapshots. Optional; leave empty to disable backtests
    # in environments where the DB isn't on the same disk.
    opentao_db_path: str = ""

    # Trader's own SQLite for paper/live trade state.
    database_path: str = "data/opentaotrader.db"

    # Bittensor network used by the live-trading CLI for signing.
    bittensor_network: str = "finney"
    subtensor_endpoint: str = ""

    # Cache TTLs for the chain client (live CLI only).
    cache_ttl_metagraph: int = 300
    cache_ttl_price: int = 30
    cache_ttl_dynamic_info: int = 120
    cache_ttl_balance: int = 60
    rpc_timeout: float = 20.0

    # Paper-trader runner. Default off so a fresh install doesn't
    # immediately run anyone's bot.
    paper_trading_enabled: bool = False

    # Colon-separated paths to user-defined strategy files.
    opentao_external_strategies: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8009


settings = Settings()
