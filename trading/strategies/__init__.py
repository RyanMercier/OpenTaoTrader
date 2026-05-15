"""Strategy plugin registry.

Built-in strategies live in this package. To register your own, subclass
``Strategy`` and decorate it with ``@register_strategy("your_key")``. Drop
the file into this directory or point ``OPENTAO_EXTERNAL_STRATEGIES`` at
its location.

The runner, the API, and the UI all discover strategies by reading
``STRATEGIES``, so anything you register here is immediately pickable from
the paper-trading create form.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Type

from .base import Strategy

logger = logging.getLogger(__name__)

STRATEGIES: dict[str, Type[Strategy]] = {}
STRATEGY_SOURCES: dict[str, str] = {}  # key -> "builtin" or "external:<path>"


def register_strategy(key: str, *, source: str = "builtin"):
    """Add a Strategy subclass to ``STRATEGIES`` under ``key``. Re-using
    the same key overwrites the prior entry; if the source changed we log
    a warning so collisions don't silently replace each other."""
    def deco(cls: Type[Strategy]) -> Type[Strategy]:
        prev = STRATEGY_SOURCES.get(key)
        if prev is not None and prev != source:
            logger.warning(
                "Strategy key %s already registered (was %s), overwriting with %s",
                key, prev, source,
            )
        STRATEGIES[key] = cls
        STRATEGY_SOURCES[key] = source
        return cls
    return deco


def list_strategies() -> list[dict]:
    """Return registry contents in JSON-friendly form for the API."""
    out = []
    for key in sorted(STRATEGIES):
        cls = STRATEGIES[key]
        first_line = ""
        if cls.__doc__:
            for ln in cls.__doc__.strip().splitlines():
                if ln.strip():
                    first_line = ln.strip()
                    break
        out.append({
            "name": key,
            "source": STRATEGY_SOURCES.get(key, "unknown"),
            "doc": first_line,
        })
    return out


def load_external_strategies(paths_env: str | None = None) -> int:
    """Import strategy files listed in ``OPENTAO_EXTERNAL_STRATEGIES`` (or
    the explicit ``paths_env`` argument). Entries are colon-separated; each
    can be a single ``.py`` file or a directory whose ``*.py`` files are
    imported in order. Returns the number of files loaded."""
    raw = paths_env if paths_env is not None else os.environ.get(
        "OPENTAO_EXTERNAL_STRATEGIES", ""
    )
    if not raw:
        return 0
    loaded = 0
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            logger.warning("External strategy path does not exist: %s", path)
            continue
        files = [path] if path.is_file() else sorted(path.glob("*.py"))
        for f in files:
            if f.name.startswith("_"):
                continue
            try:
                _import_external_file(f)
                loaded += 1
            except Exception:
                logger.exception("Failed to import external strategy %s", f)
    return loaded


def _import_external_file(path: Path) -> None:
    """Import a single strategy file. Anything it registers gets the file
    path tagged as its source so the UI can show provenance."""
    pre_keys = set(STRATEGIES)
    spec = importlib.util.spec_from_file_location(
        f"opentao_external.{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    new_keys = set(STRATEGIES) - pre_keys
    for key in new_keys:
        STRATEGY_SOURCES[key] = f"external:{path}"


# Built-in imports run last so their @register_strategy decorators see
# the symbols defined above.
from .drain_detector import DrainDetector              # noqa: E402,F401
from .mean_reversion import MeanReversionStrategy      # noqa: E402,F401
from .momentum import MomentumStrategy                  # noqa: E402,F401
from .stake_velocity import StakeVelocityStrategy      # noqa: E402,F401

__all__ = [
    "Strategy",
    "STRATEGIES",
    "STRATEGY_SOURCES",
    "register_strategy",
    "list_strategies",
    "load_external_strategies",
    "DrainDetector",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "StakeVelocityStrategy",
]
