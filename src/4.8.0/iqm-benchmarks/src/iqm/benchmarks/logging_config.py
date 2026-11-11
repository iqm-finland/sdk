# Copyright 2024 IQM Benchmarks developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to initialize logger.

Should be used in every module to output information/warnings. Handler and level setup is left to the application,
for example using :func:`configure_logging`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import logging.config
from typing import Any

# Create the logger
qcvv_logger = logging.getLogger(__name__)

PACKAGE_LOGGER_NAME = "iqm.benchmarks"
"""Name of the logger that is the ancestor of all loggers in this package."""

BASIC_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class _BelowWarningFilter(logging.Filter):
    """Pass only records below WARNING, so that they can go to stdout while WARNING and above go to stderr."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def build_logging_config(
    level: str = "INFO",
    loggers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a :func:`logging.config.dictConfig` dictionary for iqm-benchmarks.

    Only the ``iqm.benchmarks`` logger namespace (plus any extra ``loggers``) is configured. The root logger and
    existing loggers are left untouched, so third-party log output is not affected.

    Records below WARNING are written to stdout and WARNING and above to stderr. This keeps routine INFO output from
    being rendered as an error in Jupyter notebooks.

    Args:
        level: Case-insensitive logging level for the whole ``iqm.benchmarks`` package.
        loggers: Optional mapping from logger name to case-insensitive level. Names inside the ``iqm.benchmarks``
            package (e.g. ``"iqm.benchmarks.randomized_benchmarking"``) override ``level`` for that sub-tree. Other
            names (e.g. ``"qiskit"``) get the same handlers as the package logger.

    Returns:
        Configuration dictionary.

    Raises:
        ValueError: If any level is not a valid logging level.

    """
    loggers_config: dict[str, dict[str, Any]] = {}
    # An explicit entry for the package logger overrides ``level`` but keeps the package handlers.
    for name, logger_level in {PACKAGE_LOGGER_NAME: level, **(loggers or {})}.items():
        logger_level = logger_level.upper()  # noqa: PLW2901
        if not isinstance(logging.getLevelName(logger_level), int):
            raise ValueError(f"Logger '{name}' has an incorrect level value '{logger_level}'.")
        # Package children inherit the package handlers, so they must have none of their own to avoid duplicate
        # records. The package logger and foreign loggers own their handlers and must not also emit through the
        # application's root logger.
        inherits = name.startswith(f"{PACKAGE_LOGGER_NAME}.")
        loggers_config[name] = {
            "level": logger_level,
            "handlers": [] if inherits else ["stdout", "stderr"],
            "propagate": inherits,
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"below_warning": {"()": _BelowWarningFilter}},
        "formatters": {"basic": {"format": BASIC_FORMAT}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "basic",
                "stream": "ext://sys.stdout",
                "filters": ["below_warning"],
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "basic",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": loggers_config,
    }


def configure_logging(level: str = "INFO", loggers: dict[str, str] | None = None) -> None:
    """Configure logging for iqm-benchmarks, see :func:`build_logging_config` for the arguments.

    Safe to call repeatedly, for example to change the level in a notebook.
    """
    logging.config.dictConfig(build_logging_config(level=level, loggers=loggers))


@contextmanager
def temporary_log_level(*loggers: logging.Logger, level: int) -> Iterator[None]:
    """Temporarily set logger levels, restoring their previous levels on exit, also after an exception.

    Args:
        loggers: Loggers whose levels are changed.
        level: Level to use inside the ``with`` block.

    """
    previous_levels = [(logger, logger.level) for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(level)
        yield
    finally:
        for logger, previous_level in reversed(previous_levels):
            logger.setLevel(previous_level)
