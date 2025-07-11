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

"""Helper to create a SweepOptions instance from a dict."""

import ast
from typing import Any

from exa.common.control.sweep.option import (
    CenterSpanBaseOptions,
    CenterSpanOptions,
    FixedOptions,
    StartStopBaseOptions,
    StartStopOptions,
)
from exa.common.control.sweep.option.constants import OPTIONS_TYPE
from exa.common.control.sweep.option.sweep_options import SweepOptions


def convert_to_options(config: dict[str, Any]) -> SweepOptions:
    """Creates one of the options object based on configuration dictionary.

    - If configuration has keys ``start`` and ``stop``, :class:`.StartStopOptions` is created.
    - If configuration has keys ``start_exp`` and ``stop_exp``, :class:`.StartStopBaseOptions` is created.
    - If configuration has keys ``center`` and ``span``, :class:`.CenterSpanOptions` is created.
    - If configuration has keys ``center_exp`` and ``span_exp``, :class:`.CenterSpanBaseOptions` is created
    - If configuration has keys ``fixed``, :class:`.FixedOptions` is created.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: Error is raised if config has unsupported structure

    """
    config = {k.lower(): v for k, v in config.items()}
    if {OPTIONS_TYPE.get("start"), OPTIONS_TYPE.get("stop")}.issubset(set(config)):  # type:ignore[arg-type]
        updated_config = __update_config(config, OPTIONS_TYPE.get("start_stop_list"))  # type:ignore[arg-type]
        return StartStopOptions(**updated_config)
    elif {OPTIONS_TYPE.get("start_exp"), OPTIONS_TYPE.get("stop_exp")}.issubset(set(config)):  # type:ignore[arg-type]
        __rename_key(config, OPTIONS_TYPE.get("start_exp"), OPTIONS_TYPE.get("start"))  # type:ignore[arg-type]
        __rename_key(config, OPTIONS_TYPE.get("stop_exp"), OPTIONS_TYPE.get("stop"))  # type:ignore[arg-type]
        updated_config = __update_config(config, OPTIONS_TYPE.get("start_stop_base_list"))  # type:ignore[arg-type]
        return StartStopBaseOptions(**updated_config)
    elif {OPTIONS_TYPE.get("center"), OPTIONS_TYPE.get("span")}.issubset(set(config)):  # type:ignore[arg-type]
        updated_config = __update_config(config, OPTIONS_TYPE.get("center_span_list"))  # type:ignore[arg-type]
        return CenterSpanOptions(**updated_config)
    elif {OPTIONS_TYPE.get("center_exp"), OPTIONS_TYPE.get("span_exp")}.issubset(set(config)):  # type:ignore[arg-type]
        __rename_key(config, OPTIONS_TYPE.get("center_exp"), OPTIONS_TYPE.get("center"))  # type:ignore[arg-type]
        __rename_key(config, OPTIONS_TYPE.get("span_exp"), OPTIONS_TYPE.get("span"))  # type:ignore[arg-type]
        updated_config = __update_config(config, OPTIONS_TYPE.get("center_span_base_list"))  # type:ignore[arg-type]
        return CenterSpanBaseOptions(**updated_config)
    elif OPTIONS_TYPE.get("fixed") in set(config):
        updated_config = __update_config(config, OPTIONS_TYPE.get("fixed"))  # type:ignore[arg-type]
        return FixedOptions(**updated_config)
    else:
        raise ValueError(f"Config {config} cannot be converted to range options")


def __update_config(config: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: ast.literal_eval("".join(str(value).strip())) for key, value in config.items() if key in keys}


def __rename_key(config: dict[str, Any], old_key: str, new_key: str) -> None:
    config[new_key] = config.pop(old_key)
