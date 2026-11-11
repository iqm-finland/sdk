# Copyright 2024 IQM
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

from collections.abc import Callable
import logging
import logging.config
from typing import Any

from exa.common.errors.iqm_error import ValidationError

WARNINGS_LOGGER_NAME = "py.warnings"
"""Name of the logger that :func:`logging.captureWarnings` sends warnings to."""

BRIEF_DATEFMT = "%m-%d %H:%M:%S"

BRIEF = "[{asctime};{levelname}{extra_info}] {message}"
VERBOSE = "[{asctime};{levelname};{processName}({process});{threadName}({thread});{name};{filename}:{lineno}:{funcName}{extra_info}] {message}"  # noqa: E501


class ExtraFormatter(logging.Formatter):
    """Helper formatter class to pass in arbitrary extra information to log messages."""

    def __init__(self, *args, extra_info_getter: Callable[[], str] | None = None, **kwargs) -> None:
        self.extra_info_getter = extra_info_getter if extra_info_getter is not None else lambda: ""
        super().__init__(*args, **kwargs)

    def format(self, record):  # noqa: ANN001, ANN201
        extra_info = self.extra_info_getter()
        record.__dict__.update(extra_info=extra_info)
        return super().format(record)


class MaxLevelFilter(logging.Filter):
    """Drops messages at or above a given level.

    Used to send the milder messages to stdout and the rest to stderr.
    """

    def __init__(self, exclusive_maximum: int = logging.WARNING) -> None:
        self.exclusive_maximum = exclusive_maximum
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.exclusive_maximum


InfoFilter = MaxLevelFilter
"""Old name, kept for backwards compatibility."""


def init_loggers(
    loggers: dict[str, str | None] | None = None,
    default_level: str = "INFO",
    verbose: bool = False,
    extra_info_getter: Callable[[], str] | None = None,
    disable_existing_loggers: bool = True,
) -> None:
    """Set the log level of given logger names.

    Logs below WARNING level are written into stdout, and logs at WARNING level or above are written into stderr.

    By default, the root logger uses WARNING level. If ``default_level`` is above WARNING, the root logger uses it
    instead.

    See Python's logging module for possible logging levels.

    Args:
        loggers: A mapping from logger name to (case insensitive) logging level. If logging level is None or empty
            string, ``default_level`` will be used for that logger. It is possible to fine tune logging for individual
            modules, since logger propagation is turned off. Overriding works both ways: a parent logger can have higher
            logging level than its descendants, and vice versa.
            For example, ``{"sqlalchemy": None, "sqlalchemy.engine": "debug"}`` will put
            "sqlalchemy" and its descendants (i.e. "sqlalchemy.dialects", "sqlalchemy.dialects.sqlite", etc.) into
            ``default_level``, except for "sqlalchemy.engine" for which DEBUG level is defined.
            For the root logger, use empty string key, for example: ``{"": "INFO"}``.
            If empty dict or None is given, only the root logger will be initialized to its default WARNING level.
            Warnings from :mod:`warnings` are always sent to the logs, under a logger named ``py.warnings``. Its level
            cannot be configured, since warnings are always at WARNING level, like ``logger.warning``.
        default_level: The default level (case insensitive) to be used for logger names given with ``loggers``
            for which a level is not specified.
        verbose: If False, :const:``BRIEF`` format will be used for log messages, otherwise :const:``VERBOSE``.
        extra_info_getter: Optional callable to convey extra information to log messages. It will get called before
            each log message emission and the output will get appended to the log message.
        disable_existing_loggers: If True, loggers that already exist and are not listed in ``loggers`` are
            disabled, so they drop all records. Pass False to leave them as they are.

    """
    loggers = loggers or {}
    if WARNINGS_LOGGER_NAME in loggers:
        raise ValidationError(
            f"The level of logger '{WARNINGS_LOGGER_NAME}' cannot be configured. Python warnings are always logged "
            "at WARNING level."
        )
    # py.warnings must always be listed here, or disable_existing_loggers below turns it off and warnings from
    # warnings.warn stop showing up at all.
    loggers = {**loggers, WARNINGS_LOGGER_NAME: None}

    lowest_level_num = logging.CRITICAL
    loggers_config: dict[str, Any] = {}
    for logger_name, level in loggers.items():
        if not level:
            level = default_level  # noqa: PLW2901
        level = level.upper()  # noqa: PLW2901
        level_num = logging.getLevelName(level)
        if not isinstance(level_num, int):
            raise ValidationError(f"Logger name '{logger_name}' has an incorrect level value '{level}'.")
        lowest_level_num = min(level_num, lowest_level_num)
        loggers_config[logger_name] = {
            "level": level,
            "handlers": ["stdout", "stderr"],
            "propagate": False,
        }

    lowest_level = logging.getLevelName(lowest_level_num)
    lowest_level_warning_or_above = lowest_level if lowest_level_num > logging.WARNING else "WARNING"
    # The root level follows the console level when that is above WARNING. Otherwise warnings would be built and
    # then discarded by the handlers.
    loggers_config.setdefault("", {"level": lowest_level_warning_or_above, "handlers": ["stderr"]})

    config = {
        "version": 1,
        "filters": {
            "infofilter": {
                "()": InfoFilter,
            }
        },
        "disable_existing_loggers": disable_existing_loggers,
        "formatters": {
            "brief": {
                "()": lambda: ExtraFormatter(
                    fmt=BRIEF, datefmt=BRIEF_DATEFMT, style="{", extra_info_getter=extra_info_getter
                )
            },
            "verbose": {"()": lambda: ExtraFormatter(fmt=VERBOSE, style="{", extra_info_getter=extra_info_getter)},
        },
        "handlers": {
            "stdout": {
                "level": lowest_level,
                "class": "logging.StreamHandler",
                "formatter": "verbose" if verbose else "brief",
                "stream": "ext://sys.stdout",
                "filters": ["infofilter"],
            },
            "stderr": {
                "level": lowest_level_warning_or_above,
                "class": "logging.StreamHandler",
                "formatter": "verbose" if verbose else "brief",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": loggers_config,
    }
    logging.config.dictConfig(config)
    # logging.captureWarnings(True) does nothing if it is already on. If something replaced
    # warnings.showwarning since then, such as pytest, warnings would quietly stop being sent to the logs.
    # Turning it off first makes sure it is set up again.
    logging.captureWarnings(False)
    logging.captureWarnings(True)
