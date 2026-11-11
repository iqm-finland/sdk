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
"""Utilities for handling settings and options in Experiment."""

from __future__ import annotations

from collections.abc import Iterable
import copy
import logging
from typing import Any

from exa.common.data import CollectionType, DataType, Parameter, SettingNode
from exa.common.qcm_data import ChipTopology
from exa.core.experiment_definitions import (
    ComponentGrouping,
    ExperimentOptions,
    ReadoutMode,
)
from exa.core.observation import ObservationHandler

# BW COMPATIBILITY IMPORTS
from iqm.cpc.compiler._utils.settings import (  # noqa: F401
    COMPILER_DEFAULTS_SOURCE,
    GLOBAL_LOCUS_STRING_REPRESENTATION,
    gates_data_from,
    limit_components_by_settings,
)
from iqm.cpc.compiler._utils.stages import flatten_subscribed_components  # noqa: F401

logger = logging.getLogger(__name__)

COUNTER_RESULT = Parameter(
    "counter.result",
    "Multi-qubit state probabilities",
    data_type=DataType.FLOAT,
    collection_type=CollectionType.NDARRAY,
)


def validate_options(options: ExperimentOptions) -> None:
    """Validate the ExperimentOptions values.

    This is called as a part of `get_settings`.

    Args:
        options: The options to be validated.

    Raises:
        ValueError: if the options are invalid for the correct functioning of the experiment.

    """
    errors: list[str] = []
    valid_readout_modes = [member.value for member in ReadoutMode]
    if options.readout_mode not in valid_readout_modes:
        errors.append(f"- The readout_mode must be one of {valid_readout_modes}, but it is '{options.readout_mode}'")

    if options.readout_mode != ReadoutMode.COUNTER.value and isinstance(options.subscribed_components, dict):
        errors.append(
            f"- Grouped readout (i.e. options.subscribed_components being a dict) is only "
            f"supported in the readout_mode '{ReadoutMode.COUNTER.value}', while your readout_mode is "
            f"'{options.readout_mode}'. Please change the readout_mode or the"
            f" subscribed components to a list."
        )

    if len(errors) > 0:
        error_msg = "Invalid options. The following issues were found:\n" + "\n".join(errors)
        raise ValueError(error_msg)


def get_default_implementation_name(locus: str | Iterable[str], settings: SettingNode, gate: str) -> str:
    """Get the default implementation name for a given gate and locus.

    Takes into account the global default implementation and a possible locus specific implementation and also
    the symmetry properties of the gate.

    Args:
        locus: Individual qubits, couplers, or combinations.
        settings: The settings tree.
        gate: The name of the gate.

    Returns:
        The default implementation name.

    """
    return settings.get_default_implementation_name(gate, locus)


def get_gate_node_for_locus(
    locus: str | Iterable[str],
    settings: SettingNode,
    gate: str,
    implementation: str | None = None,
) -> SettingNode:
    """Get the gate :class:`settingNode` for the locus given as a parameter if it exists in the settings tree.

    Args:
        locus: Individual qubits, couplers, or combinations.
        settings: The gate settings tree.
        gate: The gate to retrieve the settings for.
        implementation: Using a custom rather than the default gate implementation.

    Returns:
        The settings of the specified locus and gate.

    """
    return settings.get_gate_node_for_locus(gate, locus, implementation)


def get_spot_dict(
    spot: tuple[tuple[Any, ...], ...],
    parameters_nested: list[list[Parameter]],
    full_calibration: dict[str | tuple, Any],
) -> dict[str | tuple, Any]:
    """Get gate calibration data dict for a sweep spot.

    Args:
        spot: tuple containing the sweep spot data.
        parameters_nested: The sweep parameters for the spot (formatted to reflect the structure of ``spot``).
        full_calibration: The full calibration data.

    Returns:
        The gate calibration data dict.

    """
    calib_dict: dict[str | tuple, Any] = {}
    for parameters, values in zip(parameters_nested, spot):
        for parameter, value in zip(parameters, values):
            if parameter.parent_name:  # element-wise parameters
                param_path = parameter.parent_name.split(".")[1:]
            else:
                param_path = parameter.name.split(".")[1:]
            node = calib_dict
            ref = full_calibration
            for p_fraq in param_path[:-1]:
                key = tuple(p_fraq.split("__")) if p_fraq not in ref else p_fraq  # handle the tuple keys
                if key not in node:
                    node[key] = {}
                node = node[key]
                ref = ref[key]
            if parameter.parent_name:
                new_array = node.get(param_path[-1], copy.copy(ref[param_path[-1]]))
                new_array[tuple(parameter.element_indices)] = value  # type: ignore[arg-type]
                value = new_array  # noqa: PLW2901
            node[param_path[-1]] = value
    return calib_dict


def get_flat_run_components_by_type(
    components: ComponentGrouping, chip_topology: ChipTopology
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Get flat lists of qubits, computational_resonators, couplers, and probe lines from component grouping.

    Args:
        components: The component grouping.
        chip_topology: The chip topology.

    Returns:
        Flat lists of qubits, computational_resonators, couplers, and probe lines from ``components``.

    """
    flattened_components = components.flatten()
    qubits = [q for q in flattened_components if q in chip_topology.qubits]
    computational_resonators = [r for r in flattened_components if r in chip_topology.computational_resonators]
    couplers = [c for c in flattened_components if c in chip_topology.couplers]
    for coupler in chip_topology.get_connecting_couplers(qubits + computational_resonators):
        if coupler not in couplers:
            couplers.append(coupler)
    probe_lines = [p for p in flattened_components if p in chip_topology.probe_lines]
    qb_clpr_res = qubits + couplers + computational_resonators
    for component, probe_line in chip_topology.component_to_probe_line.items():
        if component in qb_clpr_res and probe_line not in probe_lines:
            probe_lines.append(probe_line)
    return qubits, computational_resonators, couplers, probe_lines


def filter_components_with_gate(
    components: list[str],
    settings: SettingNode,
    gate_name: str,
) -> list[str]:
    """Filters out components that are not registered for any of the implementations of the given gate.

    Args:
        components: Components.
        settings: The settings tree.
        gate_name: The name of the gate (e.g. "measure_012").

    Returns:
        The filtered components that are registered for the gate.

    """
    if gate_name not in settings.gates.subtrees:
        return []

    gate_settings = settings.gates[gate_name]
    filtered_components = [
        component
        for component in components
        if any(component in impl.subtrees for impl in gate_settings.subtrees.values())
    ]

    return filtered_components


def filter_components_with_drive(
    components: list[str],
    settings: SettingNode,
) -> list[str]:
    """Filters out components without drive.

    Args:
        components: Components.
        settings: The settings tree.

    Returns:
        The filtered components that have drive capability.

    """
    filtered_components = [component for component in components if "drive" in settings.controllers[component].subtrees]

    return filtered_components


def filter_components_with_readout(
    components: Iterable[str],
    chip_topology: ChipTopology,
) -> list[str]:
    """Filters out components that are not connected to a probe line.

    Args:
        components: Computational components.
        chip_topology: The chip topology.

    Returns:
        The filtered components which are connected to a probe line.

    """
    filtered_components = [component for component in components if component in chip_topology.component_to_probe_line]

    return filtered_components


def get_components_without_readout(components: Iterable[str], chip_topology: ChipTopology) -> list[str]:
    """Returns only those components that are not connected to a probe line.

    Args:
        components: Components.
        chip_topology: The chip topology.

    Returns:
        The filtered components that are not connected to a probe line.

    """
    filtered_components = [
        component for component in components if component not in chip_topology.component_to_probe_line
    ]

    return filtered_components


def add_move_detuning_to_implementations(
    settings: SettingNode, observation_handler: ObservationHandler, computational_resonators: list[str]
) -> None:
    """Set the ``detuning`` parameter in MOVE gate calibration data for all implementations and loci.

    This function is temporary helper to be used until we can rely on the ``detuning``
    parameter having been set by calibration experiments.

    First we look for a ``detuning`` observation in ``observation_handler``.
    If not found there, we use (qubit drive frequency - resonator frequency) if both are
    found in ``settings``. That failing, ``detuning`` is set to zero.

    FIXME All "move" gate implementations must have a ``detuning`` parameter for this to work. Find a better way.

    Args:
        settings: settings tree to add ``detuning`` to
        observation_handler: source for the possible ``detuning`` observation
        computational_resonators: computational resonators on the QPU

    """
    # TODO: remove this function once the "detuning" parameter is always saved as an observation for each MOVE
    if not hasattr(settings.gates, "move"):
        return

    for implementation_name, implementation_node in settings.gates.move.child_nodes:
        for locus, _ in implementation_node.child_nodes:
            # TODO: not sure if it should check if "detuning" is there - we would require it there for some
            # experiments, but composite implementations are possible and are not calibrated directly

            detuning = observation_handler.get_value(f"gates.move.{implementation_name}.{locus}.detuning", load=True)
            if detuning is None:
                locus_elements = locus.split("__")

                # assuming qubit goes first
                move_qubit_frequency = settings.controllers[locus_elements[0]].drive.frequency.value
                resonator_frequency = None
                # TODO: MOVE is currently in the settings also for quare chips so we need to check this,
                # but it should not be in the settings at all for square chips
                if locus_elements[1] in computational_resonators:
                    resonator_frequency = settings.characterization.model[locus_elements[1]].frequency.value
                if move_qubit_frequency is None or resonator_frequency is None:
                    detuning = 0.0
                else:
                    detuning = move_qubit_frequency - resonator_frequency

            settings.gates.move[implementation_name][locus].detuning = detuning
            settings.gates.move[implementation_name][locus].detuning._source = COMPILER_DEFAULTS_SOURCE
