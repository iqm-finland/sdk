#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright (c) 2019-2025 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
"""YAML utilities."""

import logging
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML, YAMLError

logger = logging.getLogger(__name__)
yaml = YAML(typ="safe", pure=True)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file from the given path and raise error if the file can't be loaded."""
    path = path if isinstance(path, Path) else Path(path)
    with path.open(encoding="utf-8") as file:
        try:
            data = yaml.load(file)
        except YAMLError as err:
            raise ValueError(f"Failed to load YAML file from {path}.") from err
    logger.debug("Loaded a YAML file from %s.", path)
    return data


def dump_yaml(data: dict[str, Any], path: Path | str) -> None:
    """Dump a YAML data to the given path. Create missing directories if necessary."""
    path = path if isinstance(path, Path) else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file)
    logger.debug("Saved a YAML file to %s.", path)
