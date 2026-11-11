#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Functions for scientific formatting."""

from decimal import Decimal, InvalidOperation
import re

import numpy as np

from exa.common.data import ObservationUncertainty, ObservationValue


def scientific_rounding(value: ObservationValue, uncertainty: ObservationUncertainty | None) -> str:
    """Round a `value` according to `uncertainty` using scientific rounding rules.

    Args:
        value: The value to be rounded.
        uncertainty: The uncertainty of the value.

    Returns:
        A string formatted like "<rounded value>±<rounded uncertainty>"

    """
    if uncertainty is None:
        return str(value)
    value_value: float | str
    if isinstance(value, np.ndarray):
        if np.ndim(np.squeeze(value)) != 0 or np.issubdtype(dtype := value.dtype, complex):
            raise TypeError(f"Observation with value {value} can't be converted to float or string.")
        value_value = str(np.squeeze(value)[()]) if np.issubdtype(dtype, str) else float(np.squeeze(value)[()])
    elif isinstance(value, str):
        value_value = value
    elif isinstance(value, complex):
        raise TypeError(f"Observation with value {value} can't be converted to float or string.")
    elif isinstance(value, bool):
        value_value = float(value)
    else:
        value_value = value

    uncertainty_value: float | str
    if isinstance(uncertainty, np.ndarray) and (
        np.ndim(np.squeeze(uncertainty)) != 0 or np.issubdtype(dtype := uncertainty.dtype, complex)
    ):
        raise TypeError(f"Observation with uncertainty {uncertainty} can't be converted to float or string.")
    if isinstance(uncertainty, np.ndarray):
        if np.ndim(np.squeeze(uncertainty)) != 0 or np.issubdtype(dtype := uncertainty.dtype, complex):
            raise TypeError(f"Observation with uncertainty {uncertainty} can't be converted to float or string.")
        uncertainty_value = (
            str(np.squeeze(uncertainty)[()]) if np.issubdtype(dtype, str) else float(np.squeeze(uncertainty)[()])
        )
    elif isinstance(uncertainty, complex):
        raise TypeError(f"Observation with uncertainty {uncertainty} can't be converted to float or string.")
    else:
        uncertainty_value = uncertainty

    try:
        return _scientific_rounding(value_value, uncertainty_value)
    except InvalidOperation as err:
        raise ValueError(f"Observation with value {value} and uncertainty {uncertainty} can't be converted.") from err


def _scientific_rounding(value: float | str, uncertainty: float | str) -> str:
    """Round a `value` according to `uncertainty` using scientific rounding rules.

    Args:
        value: value to be rounded
        uncertainty: uncertainty of the value

    Returns:
        A string formatted like "<rounded value>±<rounded uncertainty>"

    """
    val = Decimal(value)
    std = Decimal(uncertainty)

    if not val.is_finite():
        return "-∞" if val < 0 else "∞"

    if std == 0:
        return f"{format(val, '.2g')}±0"

    if val.is_nan():
        return "NaN"

    if not std.is_finite():
        return f"{val}±∞"

    exponent = std.adjusted()
    mantissa = std.scaleb(-exponent)
    if 1.0 < mantissa < 1.5:
        mantissa = mantissa.scaleb(1)
        exponent -= 1
    formatted_uncertainty = Decimal(round(mantissa)).scaleb(exponent)

    round_to = min(formatted_uncertainty.adjusted(), val.adjusted())
    formatted_value = round(val, -round_to)
    return f"{formatted_value}±{formatted_uncertainty}"


def get_scale(values: np.ndarray, step: int = 3) -> int:
    """Get the scale of a given NumPy array in given steps.

    Scale of 1000 is 3, scale of 1001 is 6 for step=3.

    Args:
        values (nd.array): A NumPy array containing the values which scale will be determined.
        step (int): A step in which the scale can grow.
    Returns: scale of the largest (absolute) value in values as int.

    """
    scale = np.max([np.abs(np.min(values.flatten())), np.abs(np.max(values.flatten()))])
    if scale <= 0:
        return 1
    power_of_ten = np.log10(scale) / step
    if np.sign(power_of_ten) < 0:
        scale = np.sign(power_of_ten) * np.ceil(np.abs(power_of_ten)) * step
    else:
        scale = np.sign(power_of_ten) * np.floor(np.abs(power_of_ten)) * step
    return scale


def get_scaling(values: np.ndarray) -> float:
    """Return the power of 10 value for given set of values."""
    return 10 ** -(get_scale(values))


def get_time_unit(time: float) -> tuple[float, str]:
    """Based on the time in seconds, get a unit and a multiplier for plots."""
    unit = "s"
    scaling = 1.0
    if 1e-1 >= time > 1e-4:
        scaling = 1e3
        unit = "ms"
    elif 1e-4 >= time > 1e-7:
        scaling = 1e6
        unit = r"$\mu$s"
    elif time <= 1e-7:
        scaling = 1e9
        unit = "ns"
    return scaling, unit


def get_prefix(values: np.ndarray) -> str:
    """Get multiple unit prefixes."""
    unit_prefixes = {
        -24: "y",  # yocto
        -21: "z",  # zepto
        -18: "a",  # atto
        -15: "f",  # femto
        -12: "p",  # piko
        -9: "n",  # nano
        -6: "u",  # mikro
        -3: "m",  # milli
        0: "",
        3: "k",  # kilo
        6: "M",  # mega
        9: "G",  # giga
        12: "T",  # tera
        15: "P",  # peta
        18: "E",  # exa
        21: "Z",  # zetta
        24: "Y",  # yotta
    }
    scale = get_scale(values)
    return unit_prefixes.get(scale, "")


def format_unit_string(unit_prefix: str, unit: str, enclosure: str = "(") -> str:
    """Return an unit string based on the enclosure sign.

    Args:
        unit_prefix: The unit prefix
        unit: The unit
        enclosure: Sign for the enclosure. Can be "(", "[" or "{". Defaults to "(".

    Returns:
        Formatted unit string

    """
    if not unit_prefix and not unit:
        return ""
    if enclosure == "(":
        return f"({unit_prefix}{unit})"
    if enclosure == "[":
        return f"[{unit_prefix}{unit}]"
    if enclosure == "{":
        return "{" + f"{unit_prefix}{unit}" + "}"
    return f"{unit_prefix}{unit}"


def format_names(name: str, delimiters: list[str] | None = None) -> str:
    """Format the variable names by removing '_' and '.'.

    Args:
        name: name to be formatted.
        delimiters: list of delimiters to be removed.

    Returns:
        The formatted name.

    """
    # Insert regexp magic here
    if delimiters is None:
        delimiters = ["_", r"\."]
    formatted = name
    for delimiter in delimiters:
        formatted = re.sub(delimiter, " ", formatted)
    return formatted
