"""Structured, human-readable logging setup shared by the API, CLI, and pipeline.

Kept in one place so every entrypoint (uvicorn, the Typer CLI, pytest) logs
identically instead of each script reinventing `logging.basicConfig`.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. by a previous call or by uvicorn) -- just
        # make sure the level matches and bail out, so we never double-log.
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers unless we're at DEBUG.
    if level != "DEBUG":
        for noisy in ("httpx", "urllib3", "sqlalchemy.engine"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
