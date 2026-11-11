#  *****************************************************************************
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
#  *****************************************************************************
"""Utility functions for QPUPlot, mostly to do with database query."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import floor, log10
import re
from typing import Any

import numpy as np
from scipy.special import logit
from typing_extensions import deprecated

from exa.common.helpers import format_deprecated
from exa.common.qcm_data import ChipTopology, sort_couplers
from exa.common.qcm_data.chad_model import CHAD
from exa.core.observation import LocalObservation
from exa.core.plotting.plot_styles import SI_PREFIXES
from exa.core.util.dut_field_decoder import decode_dut_field  # noqa: F401
from iqm.station_control.interface.models import ObservationData


class CrystalChipType(Enum):
    """Chip type based on a crystal architecture."""

    MINI_CRYSTAL_20 = "mini_crystal_20"
    MINI_CRYSTAL_54 = "mini_crystal_54"
    CRYSTAL_5 = "crystal_5"
    CRYSTAL_20 = "crystal_20"
    CRYSTAL_54 = "crystal_54"
    CRYSTAL_150 = "crystal_150"


@dataclass
class QPUPlotObservation:
    """Store component name and gate metric next to observation."""

    component: str
    """The component this observations belongs to, e.g., 'QB1', 'TC1', 'COMPR1'."""
    metric: str
    """The name of the observation parameter, e.g., 'rb.prx.{prx_implementation}.fidelity', 'ghz_state.population'."""
    observation: ObservationData | LocalObservation
    """The observation data."""


def print_observations(observations: list[ObservationData]) -> None:
    """Print observations list."""
    for idx, obs in enumerate(observations, start=1):
        print(
            f"[{idx}]  dut_field: {obs.dut_field}\n"
            f"{re.sub(r'.', ' ', str(idx))}    value: {obs.value}\n"
            f"{re.sub(r'.', ' ', str(idx))}    tags: {obs.tags}\n"
            f"{re.sub(r'.', ' ', str(idx))}    time-stamp-created: {obs.created_timestamp}"
        )


def get_exponent(val: float) -> int:
    """Get the exponent of the given value.

    E.g. find_exp(3.7e-13) returns -13, find_exp(7250.1) returns 3.

    Args:
        val: The value to get the exponent from.

    Returns:
        The exponent of the provided value.

    """
    if abs(val) > 0:
        return floor(log10(abs(val)))
    return 0


def fix_unit_prefixes(exponent: float, unit: str, observation_type: str) -> tuple[float, str]:
    """Identify appropriate prefix to the unit and provides corresponding conversion factor.

    Args:
        exponent: The exponent of value (in scientific notation).
        unit: The unit to change.
        observation_type: The name of observation/quantity (used to hardcode which quantities to scale).

    Returns:
        The factor to convert value with and the unit with appropriate prefix.

    """
    # Dictionary keys are the exponents (assuming base ten).
    # Greek characters are given in unicode

    if ("fidelity" in observation_type or "error" in observation_type) and unit == "":
        # Fidelity or error into percent
        if exponent > -4:
            new_unit = "%"
            conversion_factor = 100.0
        else:
            new_unit = "1e" + str(exponent)
            conversion_factor = 1 / (10**exponent)
    elif exponent != 0:
        if unit:
            multiple_of_three_lower = int(exponent // 3) * 3  # rounds down
            multiple_of_three_upper = (int(exponent // 3) + 1) * 3  # rounds up

            if abs(multiple_of_three_lower - exponent) <= abs(multiple_of_three_upper - exponent) + 1:
                closest_multiple_of_three = multiple_of_three_lower
            else:
                closest_multiple_of_three = multiple_of_three_upper
            magnitude = 10**closest_multiple_of_three
            prefix_str = SI_PREFIXES.get(closest_multiple_of_three, "")
            new_unit = prefix_str + unit
            conversion_factor = 1 / magnitude
        else:
            new_unit = "1e" + str(exponent)
            conversion_factor = 1 / (10**exponent)

    else:
        new_unit = unit
        conversion_factor = 1.0

    return conversion_factor, new_unit


def get_scaling_from_unit(unit: str) -> float:
    """Helper function to determine the applied scaling factor from the unit prefix. Currently, the function assumes that
    the first letter in the unit is the scaling factor. This is correct for our most commonly used units such as: A,
    V and Hz. Will not work as intended if the unit is kg.

    Args:
        unit: string of the unit.

    Returns:
        A float corresponding to the scaling applied to the base SI unit.

    """  # noqa: E501
    supported_prefixes = list(SI_PREFIXES.values())
    corresponding_exp = list(SI_PREFIXES.keys())
    exponent = 0
    exception_cases = {"%": -2, "rad": 0}
    # if the unit is radians, the initial r can also be interpreted as ronto, which we want to avoid in this case.
    e_notation = "1e"

    if unit in exception_cases:
        exponent += exception_cases[unit]
    elif e_notation in unit:
        e_notation_number = int(unit[2:])
        exponent += e_notation_number
    elif len(unit) > 1:
        prefix = unit[0]
        remaining_unit = unit[1:]
        if prefix in supported_prefixes:
            exponent += corresponding_exp[supported_prefixes.index(prefix)]
        if remaining_unit in exception_cases:
            exponent += exception_cases[remaining_unit]

    scaling_factor = float(10**exponent)

    return scaling_factor


def change_unit_magnitude(
    unit: str, quantity: list[float] | np.ndarray, observation_type: str, quantity_error: None | list[float] = None
) -> tuple[str, list[Any] | np.ndarray, None | np.ndarray]:
    """Change the magnitude of units.

    Changes some units into other common magnitudes. For instance,  Hz -> GHz, s -> microseconds etc.
    This function is similar to _fix_data_units function in topology.py, but works with lists instead of dictionaries.

    Args:
        unit: The unit to be scaled.
        quantity: Determines rescaling.
        observation_type: Certain types can rescale values for empty unit.
        quantity_error: Uncertainties of the quantity.

    Returns:
        The new unit, the converted data, and the converted error if provided.

    """
    exponent = get_exponent(max(np.abs(quantity)))  # the data sets the scale
    conversion_factor, new_unit = fix_unit_prefixes(exponent, unit, observation_type)

    conv_data = np.asarray(quantity) * conversion_factor
    if isinstance(quantity, list):
        conv_data = conv_data.tolist()

    new_errors = None
    if quantity_error is not None:
        new_errors = np.asarray(quantity_error) * conversion_factor

    return new_unit, conv_data, new_errors


@deprecated(
    format_deprecated(old="split_qubits_into_low_high", new="split_qubits_into_low_high_frequency", since="2026-09-17")
)
def split_qubits_into_low_high(chad: CHAD, qpu_type: str) -> tuple[list[str], list[str]]:
    """Output the high and low frequency qubits of the corresponding qpu type.

    Used primarily for the histogram plot method.

    Args:
        chad: The chip architecture definition (CHAD).
        qpu_type: The QPU type used to identify if chip supports splitting qubits into high and low frequency.

    Returns:
        The qubit component names. The tuple is ordered as
        ([low freq. qubits], [high freq. qubits]).

    """
    chip_topology = ChipTopology.from_chad(chad)

    grp1: list[str] = []
    grp2: list[str] = []

    qubit_name = "QB1"
    _search_qubit_connections(chip_topology, grp1, grp2, qubit_name, grp_index=1)

    if qpu_type.casefold() in [
        qpu_type.value.casefold()
        for qpu_type in [
            CrystalChipType.CRYSTAL_20,
            CrystalChipType.MINI_CRYSTAL_20,
            CrystalChipType.CRYSTAL_5,
            CrystalChipType.MINI_CRYSTAL_54,
        ]
    ]:
        # QB1 belongs to the low frequency group
        return (grp1, grp2) if qubit_name in grp1 else (grp2, grp1)
    if qpu_type.casefold() == CrystalChipType.CRYSTAL_54.value.casefold():
        # QB1 belongs to the high frequency group
        if qubit_name in grp2:
            low_freq_qbs = grp1
            high_freq_qbs = grp2
        else:
            low_freq_qbs = grp2
            high_freq_qbs = grp1
    else:
        raise Warning(
            f"splitting qubits into high and low frequency groups is not supported for this chip type {qpu_type}"
        )

    return low_freq_qbs, high_freq_qbs


def split_qubits_into_low_high_frequency(
    chip_design_record: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Partition the QPU qubits to a two-coloring of high and low frequency qubits.

    The two-coloring is determined from the QPU topology alone, and it is assumed the frequencies match this partition.
    Which subset is the high/low-frequency one is determined from the average qubit design frequency of the subset.

    Args:
        chip_design_record: Chip design record for the QPU.

    Returns:
        * Names of low-frequency qubits.
        * Names of high-frequency qubits.

    Warns:
        If the chip has no qubit two-coloring, or design frequency data is missing for some qubits,
        the high/low frequency split can't be determined.

    """
    chip_design_qubits = chip_design_record["content"]["components"]["qubit"]
    chip_topology = ChipTopology.from_chip_design_record(chip_design_record)

    grp1: list[str] = []
    grp2: list[str] = []

    qubit_name = "QB1"
    _search_qubit_connections(chip_topology, grp1, grp2, qubit_name, grp_index=1)

    if not grp1 or not grp2:
        raise Warning(
            "Splitting qubits into high and low frequency groups is not supported for this chip: "
            "could not form two coupled qubit groups."
        )

    design_frequencies = {
        qubit["name"]: freq
        for qubit in chip_design_qubits
        if (freq := qubit.get("properties", {}).get("design", {}).get("f_ge_bare")) is not None
    }

    if not all(name in design_frequencies for name in grp1 + grp2):
        raise Warning(
            "Splitting qubits into high and low frequency groups is not supported for this chip: "
            "design frequency (f_ge_bare) is missing for some qubits."
        )

    avg_freq_grp1 = sum(design_frequencies[name] for name in grp1) / len(grp1)
    avg_freq_grp2 = sum(design_frequencies[name] for name in grp2) / len(grp2)

    return (grp1, grp2) if avg_freq_grp1 < avg_freq_grp2 else (grp2, grp1)


def _search_qubit_connections(
    chip_topology: ChipTopology, group1_list: list[str], group2_list: list[str], qubit_name: str, grp_index: int
) -> None:
    """Sort qubit_name into group1_list or group2_list depending on the grp_index.

    Then it proceeds to do the same with the qubits coupled to it.

    Args:
        chip_topology: Topology of the chip, used to find qubits coupled to qubit_name.
        group1_list: List of qubits
        group2_list: Another list of qubits
        qubit_name: name of current qubit to sort and its connections
        grp_index: grp index of qubit_name. Determines which group it belongs to.

    """
    if grp_index == 1:
        group1_list.append(qubit_name)
    else:
        group2_list.append(qubit_name)

    neighbor_qubits = chip_topology.get_neighbor_locus_components([qubit_name]) & chip_topology.qubits

    for neighbor_qubit in sorted(neighbor_qubits):
        if neighbor_qubit not in group1_list + group2_list:
            _search_qubit_connections(chip_topology, group1_list, group2_list, neighbor_qubit, grp_index=-grp_index)


def sort_components(unsorted_unique_components: list[str] | set[str]) -> list[str]:
    """Given a list of component names, this function attempts to sort them in ascending order.

    Qubits come before couplers.

    Args:
        unsorted_unique_components: The component names to be sorted. Must be a list of unique names.

    Returns:
        sorted_component_names: The sorted (in ascending order, qubits first) component names.

    """
    # attempt to sort components
    single_qubits = [
        component
        for component in unsorted_unique_components
        if ("QB" in component and len(re.sub(r"[0-9a-zA-Z]", "", component)) == 0)
    ]
    cpl_components = [component for component in unsorted_unique_components if "QB" not in component]
    qb_qb_comp = [
        component
        for component in unsorted_unique_components
        if ("QB" in component and re.sub(r"[0-9a-zA-Z]", "", component))
    ]  # two-qubit Star components are of the form QB1_QB2

    sorted_qubits = sort_couplers(single_qubits)
    sorted_cpls = sort_couplers(cpl_components)
    sorted_qb_qb = sort_couplers(qb_qb_comp)

    return sorted_qubits + sorted_cpls + sorted_qb_qb


def get_unit(units: dict[str, str], components: Iterable[str]) -> str:
    """Return the unit for the data for the given component.

    The comp should be either 'QB', 'TC' or 'COMP'.

    Args:
        units: Mapping from component to unit.
        components: Determines if data is considered.

    Returns:
        The unit of the given component.

    """
    unit_list = [units[component] for component in components if component in units]
    if len(unit_list) > 0:
        return unit_list.pop(0)
    return ""


def shorten_label(quantity: str) -> str:
    """Provide a shortened unit name for axis labels, given the observation name.

    Args:
        quantity: A long descriptive name of the quantity.

    Returns:
        The shorthand name.

    """
    return next(
        iter(
            match.group(0).capitalize().replace("point", "bias").replace("Flux", "Applied flux")
            for key in ("time", "frequency", "fidelity", "error", "idling point", "flux")
            if (match := re.search(key, quantity, re.IGNORECASE))
        ),
        quantity,
    )


def logit_mean(x: np.ndarray) -> float:
    """Calculates the logit mean. Intended to replace median fidelity values. Idea provided by Matthew Sarsby.

    Args:
        x: Numpy array of values to calculate logit mean for.

    Returns:
        Value corresponding to the logit mean.

    """
    logit_values = logit(x)
    arithmetic_login_mean = np.mean(logit_values)
    return 1 / (1 + np.exp(-arithmetic_login_mean))


def check_same_unit(observations: list[ObservationData | LocalObservation]) -> str:
    """Check that all units in the observations are the same.

    Args:
        observations: The observations to retrieve the units from.

    Returns:
        The unit if they are the same.

    Raises:
        ValueError if units are not the same.

    """
    units = set()
    for obs in observations:
        units.add(obs.unit.casefold() if obs.unit else "")

    if len(units) > 1:
        raise ValueError(f"Observations have mismatching units, found {units} in {observations=}.")

    return units.pop()
