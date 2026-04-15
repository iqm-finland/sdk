# Copyright 2024-2025 IQM
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
from collections import defaultdict
from collections.abc import Iterable, Sequence
import copy
from enum import Enum
import inspect
from inspect import signature
from itertools import permutations
import logging
from types import UnionType
from typing import Any, get_args, get_origin

import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.errors.exa_error import UnknownSettingError
from exa.common.qcm_data.chip_topology import ChipTopology, sort_components
from iqm.cpc.compiler.compilation_stage import CompilationStage
from iqm.cpc.core.config import (
    ComponentGrouping,
    Components,
    set_gate_implementation_as_default,
)
from iqm.cpc.core.observation.observation_handler_base import ObservationHandlerBase
from iqm.cpc.core.observation.observation_parameters import (
    AC_STARK_QUBIT_TO_SWEEP,
    COMPONENT_FREQUENCY,
    CPHASE_DERIVATIVE,
    CPHASE_PI_DISTANCE,
    F0G1_PULSE_STARK_SHIFT_POLYNOMIAL_COEFFICIENTS,
    FLUX_GATE_DETUNING,
    FLUX_PULSE_GATE_CHARACTERIZATION_PARAMETERS,
    MEASURE_CHARACTERIZATION_PARAMETERS,
    RZ_ANGLE,
    SINGLE_COUPLER_MODEL_CHARACTERIZATION_PARAMETERS,
    SINGLE_QUBIT_MODEL_CHARACTERIZATION_PARAMETERS,
    SINGLE_QUBIT_READOUT_MODEL_CHARACTERIZATION_PARAMETERS,
)
from iqm.pulse.builder import build_quantum_ops
from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.gate_implementation import Locus as LocusTuple
from iqm.pulse.quantum_ops import QuantumOp
from iqm.station_control.client.qon import locus_str_to_locus, locus_to_locus_str

logger = logging.getLogger(__name__)


def _conf_source(configurator: str) -> dict[str, str]:
    """Configuration source for a setting."""
    # TODO just the configurator str would be enough in Setting._source
    return {"type": "configuration_source", "configurator": configurator}


DEFAULT_REPETITIONS = 1000
"""Default repetitions (shots)"""
STATION_CONFIGURATION_SOURCE = _conf_source("station_configuration")
"""Source for settings whose values come from station config."""
COMPILER_DEFAULTS_SOURCE = _conf_source("compiler_defaults")
"""Source for settings whose values come from compiler hardcoded defaults."""
GATE_DEFAULTS_SOURCE = _conf_source("gate_defaults")
"""Source for gate settings whose values come from iqm-pulse hardcoded defaults."""
QUANTUM_OP_DEFAULTS_SOURCE = _conf_source("quantum_op_defaults")
"""Source for QuantumOp default implementations."""

GLOBAL_LOCUS_STRING_REPRESENTATION = "_ANY_"
"""Represents the "whole QPU" or "any locus, as it does not matter in this context" locus in the settings tree paths
(in the calibration data this is represented by an empty tuple)."""
# TODO: implement a "transformer" abstraction in the ScheduleBuilder for these kind of entities that need calibration
# data but do not have a meaningful locus


def limit_components_by_settings(
    components: ComponentGrouping | Components, settings: SettingNode, chip_topology: ChipTopology
) -> ComponentGrouping:
    """Limit run components by settings.

    Only the components/groups/color groups found in the settings are returned.

    Args:
        components: The component grouping.
        settings: The settings tree.
        chip_topology: The chip topology.

    Returns:
        The component grouping limited by the settings.

    """
    if not isinstance(components, ComponentGrouping):
        components = ComponentGrouping(components)
    if not settings:
        return components
    components_in_settings = [c for c in chip_topology.all_components if c in settings.controllers.children]
    return components.limit_by_components(components_in_settings)


def get_controller_settings(
    components: list[str],
    chip_topology: ChipTopology,
    controller_mapping: dict[str, dict[str, str]],
    station_control_settings: SettingNode,
) -> tuple[SettingNode, list[str]]:
    """Create the controllers node in the settings tree.

    The controllers node is created as per the station control settings and populated with default values from the
    experiment configuration.

    Args:
        components: The list of components names.
        chip_topology: The chip topology.
        controller_mapping: The component to operations mapping.
        station_control_settings: The station control settings.

    Returns:
        The controllers node of the settings tree.

    """
    probe_lines = [
        pl for pl, pl_comps in chip_topology.probe_line_to_components.items() if set(pl_comps).intersection(components)
    ]
    all_components = components + probe_lines
    settings = _get_controller_settings(all_components, controller_mapping, station_control_settings)
    for probe_line in probe_lines:
        if hasattr(settings.controllers[probe_line].readout, "local_oscillator"):
            settings.controllers[probe_line].readout.local_oscillator.status = True
            settings.controllers[probe_line].readout.local_oscillator.status._source = COMPILER_DEFAULTS_SOURCE
    components_with_drive = [
        component for component in all_components if hasattr(settings.controllers[component], "drive")
    ]
    # drive settings
    override_drive_with = {
        "status": True,
        "continuous_wave_enabled": False,
    }
    _override_operation_settings_with_values_from_dict(
        settings=settings, operation="drive", components=components_with_drive, override_with=override_drive_with
    )
    # flux settings
    components_with_flux_and_awg = [
        component
        for component in all_components
        if hasattr(settings.controllers[component], "flux") and hasattr(settings.controllers[component].flux, "awg")
    ]
    override_flux_with = {"awg.status": True, "awg.continuous_wave.enabled": False}
    _override_operation_settings_with_values_from_dict(
        settings=settings,
        operation="flux",
        components=components_with_flux_and_awg,
        override_with=override_flux_with,
    )
    return settings, probe_lines


def update_settings_single_shot_threshold(
    qubits: list[str],
    couplers: list[str],
    settings: SettingNode,
    gate_definitions: dict[str, QuantumOp],
) -> None:
    """Set the default readout settings corresponding to thresholded single shot mode.

    Args:
        qubits: The list of qubits.
        couplers: The list of couplers.
        settings: The settings tree.
        gate_definitions: The gate definitions..

    """
    settings.controllers.options.playlist_repeats = DEFAULT_REPETITIONS
    settings.controllers.options.playlist_repeats._source = COMPILER_DEFAULTS_SOURCE

    settings.controllers.options.averaging_bins = DEFAULT_REPETITIONS
    settings.controllers.options.averaging_bins._source = COMPILER_DEFAULTS_SOURCE

    components = qubits + couplers
    for measure_gate_name in [
        gate_name for gate_name in ["measure", "measure_fidelity"] if gate_name in settings.gate_definitions.subtrees
    ]:
        for component in components:
            for impl_name in gate_definitions[measure_gate_name].implementations:
                if component in settings.gates[measure_gate_name][impl_name].children:
                    settings.gates[measure_gate_name][impl_name][component].acquisition_type = "threshold"
                    settings.gates[measure_gate_name][impl_name][
                        component
                    ].acquisition_type._source = COMPILER_DEFAULTS_SOURCE


def _get_controller_settings(
    components: list[str], controller_mapping: dict[str, dict[str, str]], station_control_settings: SettingNode
) -> SettingNode:
    """Get controller settings."""
    default_tree = station_control_settings
    pruned_tree = SettingNode("root", controllers=SettingNode("controllers", align_name=False))
    for component in components:
        if component in controller_mapping:
            controller_dict = controller_mapping[component]
        else:
            raise ValueError(f"The provided component name '{component}' doesn't exist in the configuration.")
        if controller_dict:
            controllers = {key: default_tree[controller] for key, controller in controller_dict.items()}
        else:
            controllers = {}
        pruned_tree.controllers[component] = SettingNode(component, align_name=False, **controllers)  # type: ignore[arg-type]
        pruned_tree.controllers[component].set_source(STATION_CONFIGURATION_SOURCE)

    for controller in ["counter", "options"]:
        pruned_tree.controllers[controller] = SettingNode(
            controller,
            align_name=False,
            **default_tree[controller].children,  # type: ignore[union-attr]
        )
        pruned_tree.controllers[controller].set_source(STATION_CONFIGURATION_SOURCE)
    return pruned_tree


def _override_operation_settings_with_values_from_dict(
    settings: SettingNode, operation: str, components: list[str], override_with: dict[str, Any]
) -> None:
    """Override settings of an operation with values defined in a dict of paths vs. values.

    Args:
        settings: A tree containing the settings.
        operation: The name of the operation to apply the settings for.
        components: The components for which to apply the rules.
        override_with: A dict mapping settings paths (relative to the operation node) to the override values.

    """
    for component in components:
        for path, value in override_with.items():
            _update_settings_value(settings.controllers[component][operation], path.split("."), value)


def _update_settings_value(settings: SettingNode, keys: list[str], value: Any) -> None:
    if len(keys) == 1:
        if keys[0] in settings.children:
            settings[keys[0]] = value
            settings[keys[0]]._source = COMPILER_DEFAULTS_SOURCE
        return None
    try:
        return _update_settings_value(settings[keys[0]], keys[1:], value)  # type: ignore[arg-type]
    except UnknownSettingError:
        return None


def _add_readout_options_settings(settings: SettingNode) -> None:
    """Add Settings common to stages that add readouts and to the circuits and subsrcibing them."""
    settings.stages["additionally_subscribed_components"] = Setting(
        Parameter(
            "additionally_subscribed_components",
            label="Always subscribe (measure and collect data from) these physical components,"
            " applies to all readout labels.",
            data_type=DataType.STRING,
            collection_type=CollectionType.LIST,
        ),
        [],  # type:ignore[arg-type]
    )
    settings.stages["additionally_probed_components"] = Setting(
        Parameter(
            "additionally_probed_components",
            label="Send terminal probe pulse to these physical components. Do not subscribe to their measurement data.",
            data_type=DataType.STRING,
            collection_type=CollectionType.LIST,
        ),
        [],  # type:ignore[arg-type]
    )
    settings.stages["unsubscribed_labels"] = Setting(
        Parameter(
            "unsubscribed_labels",
            label=(
                "Do not subscribe the readout data from these readout labels. Overrides "
                "`additionally_subscribed_components` where"
                " applicable."
            ),
            data_type=DataType.STRING,
            collection_type=CollectionType.LIST,
        ),
        [],  # type:ignore[arg-type]
    )
    # TODO: this is hacky. Setting should support dict values, in which case we would not have a node for these
    # settings.stages["subscribed_groups"] = SettingNode("subscribed_groups")


def _map_param_types(type_hint: type) -> tuple[DataType, CollectionType]:  # noqa: PLR0912
    """Map a python typehint into EXA Parameter's `(DataType, CollectionType)` tuple.

    Args:
        type_hint: python typehint.

    Returns:
        A `(DataType, CollectionType)` tuple

    """
    if type_hint is inspect._empty:  # bw compatibility for not requiring typing in circuit functions
        return DataType.ANYTHING, CollectionType.SCALAR

    if isinstance(type_hint, UnionType):
        type_hint = next(h for h in get_args(type_hint) if h is not None)

    if isinstance(type_hint, str):
        union_args = [h.strip() for h in type_hint.split("|")]
        if len(union_args) > 2:
            return DataType.ANYTHING, CollectionType.SCALAR
        type_hint = next(h for h in union_args if h != "None")  # type:ignore[assignment]

    if "Literal" in str(type_hint):  # hacky since Literal cannot be used in type checks
        type_hint = type(get_args(type_hint)[0])

    if hasattr(type_hint, "__iter__") and type_hint is not str and type(type_hint) is not str:
        if type_hint == np.ndarray:
            data_type = DataType.COMPLEX  # due to np.ndarray not being generic we assume complex numbers
            collection_type = CollectionType.NDARRAY
            return (data_type, collection_type)
        if get_origin(type_hint) is list:
            collection_type = CollectionType.LIST
            type_hint = get_args(type_hint)[0]
        else:
            return DataType.ANYTHING, CollectionType.SCALAR
    else:
        collection_type = CollectionType.SCALAR

    if isinstance(type_hint, Enum):
        type_hint = type(type_hint.value)

    if isinstance(type_hint, str) and "list" in type_hint:
        type_hint = type_hint.replace("list[", "").replace("]", "")  # type: ignore[assignment]

    if type_hint is float or type_hint == "float":
        data_type = DataType.FLOAT
    elif type_hint is int or type_hint == "int":
        data_type = DataType.INT
    elif type_hint is str or type_hint == "str":
        data_type = DataType.STRING
    elif type_hint is complex or type_hint == "complex":
        data_type = DataType.COMPLEX
    elif type_hint is bool or type_hint == "bool":
        data_type = DataType.BOOLEAN
    else:
        return DataType.ANYTHING, collection_type
    return (data_type, collection_type)


def add_stage_setting_options(
    settings: SettingNode, stages: list[CompilationStage], default_arg_names: list[str]
) -> None:
    """Add the "stages" node into the settings.

    The stages node contains common readout subscription settings under its root node and subnodes for each compilation
    stage. Under the stage nodes, there are nodes for each compiler pass for that stage, and the pass nodes contain
    the pass option settings (pass function args that are not one of the default args relating to the Compiler context).

    Compiler pass options must be of a data type supported by :class:`.Parameter`.

    Args:
        settings: the settings tree.
        stages: the compilation stages.
        default_arg_names: the default argument names.

    """
    settings["stages"] = SettingNode("stages", align_name=False)  # TODO: align_name=False is a hack
    _add_readout_options_settings(settings)
    common_readout_option_names = list(settings.stages.settings.keys())
    for stage in stages:
        align_name = False if stage.name == "circuit_generation" else True
        # FIXME: circuit generation cannot align name for historical reasons -- it would break all the experiment plots
        # for example, but this should be fixed eventually
        settings.stages[stage.name] = SettingNode(stage.name, align_name=align_name)
        for pass_func in stage.passes:
            settings.stages[stage.name][pass_func.__name__] = SettingNode(pass_func.__name__, align_name=align_name)
            sig = signature(pass_func).parameters
            # the first arg is the "data"
            for arg in list(sig.keys())[1:]:
                if arg not in default_arg_names and arg not in common_readout_option_names:
                    param = sig[arg]
                    data_type, collection_type = _map_param_types(param.annotation)
                    value = param.default if param.default is not param.empty else None
                    if isinstance(value, Enum):
                        value = value.value
                    settings.stages[stage.name][pass_func.__name__][arg] = Setting(
                        Parameter(arg, data_type=data_type, collection_type=collection_type),
                        value,
                        source=None if value is None else COMPILER_DEFAULTS_SOURCE,
                    )


def assign_default_gate_implementations_from(
    gate_definition_settings: SettingNode,
    gate_definitions: dict[str, QuantumOp],
    skip_missing: bool = False,
) -> dict[str, QuantumOp]:
    """Get a copy of ``self.gate_definitions`` with the default implementations assigned from settings.

    Args:
        gate_definition_settings: the ``gate_definitions`` subnode in the settings tree.
        gate_definitions: the `gate definitions dict.
        skip_missing: whether to skip gates missing from ``gate_definitions``.

    Returns:
        The gate definitions dict which is now up to date with the settings.

    """
    per_locus_defaults_dict: dict[str, dict[tuple[str, ...], list[str]]] = {}
    for gate_name in gate_definition_settings.children:
        if (gate_name not in gate_definitions) and skip_missing:
            continue
        if default_implementation := gate_definition_settings[gate_name].default_implementation.value:  # type: ignore[union-attr]
            if default_implementation in gate_definitions[gate_name].implementations or not skip_missing:
                set_gate_implementation_as_default(
                    gate_definitions,
                    gate_name,
                    default_implementation,  # type: ignore[union-attr]
                )
        for impl_name, impl_node in gate_definition_settings[gate_name].children.items():  # type: ignore[union-attr]
            if isinstance(impl_node, SettingNode) and (
                impl_name in gate_definitions[gate_name].implementations or not skip_missing
            ):  # type: ignore[index]
                for locus in gate_definition_settings[gate_name][impl_name].override_default_for_loci.value:  # type: ignore[index,union-attr]
                    locus_tuple = tuple(locus.split("__"))
                    gate_definitions[gate_name].set_default_implementation_for_locus(impl_name, locus_tuple)
                    if gate_name not in per_locus_defaults_dict:
                        per_locus_defaults_dict[gate_name] = {}
                    if locus_tuple not in per_locus_defaults_dict[gate_name]:
                        per_locus_defaults_dict[gate_name][locus_tuple] = [impl_name]
                    elif impl_name not in per_locus_defaults_dict[gate_name][locus_tuple]:
                        per_locus_defaults_dict[gate_name][locus_tuple].append(impl_name)

    for gate_name, gate_defaults in per_locus_defaults_dict.items():
        num_per_components_defaults: dict[frozenset[str], list[str]] = {}
        for locus, impls in gate_defaults.items():
            if len(impls) > 1:
                raise ValueError(
                    "Conflicting locus-specific default implementations for locus"
                    f" `{locus}`: the default implementation for operation `{gate_name}` has been set"
                    f" simultaneously to `{impls}`. Make sure you don't have the same locus set into"
                    " `settings.gates.<gate>.<implementation>.override_default_for_loci` for multiple "
                    f" different implementations of `{gate_name}`."
                )
            if gate_definitions[gate_name].arity > 1 and gate_definitions[gate_name].symmetric:
                locus_set = frozenset(locus)
                if locus_set not in num_per_components_defaults:
                    num_per_components_defaults[locus_set] = impls  # this must be now len=1
                elif impls[0] not in num_per_components_defaults[locus_set]:
                    raise ValueError(
                        "Conflicting locus-specific default implementations for components in a locus"
                        f" `{locus}`: for a symmetric gate such as {gate_name}, there cannot be more than one"
                        " locus-specific defaults for any permutation of the locus components, but here we have"
                        f" has at least two: {num_per_components_defaults[locus_set][0]} and {impls[0]}."
                    )

    return gate_definitions


def update_settings_from_observations(node: SettingNode, keys: Sequence[str], obs: Any) -> None:
    """Update a Settings tree recursively from an observation.

    Sets the observation as the ``source`` for the Setting.

    Args:
        node: Settings node.
        keys: ``obs`` dut_field fragments.
        obs: The observation to use.

    """
    if len(keys) == 1:
        if (s := node.settings.get(keys[0])) and not s.read_only:
            node[keys[0]] = obs.value
            node[keys[0]]._source = obs
            # node[keys[0]] = s.update(obs.value, source=obs)  # TODO more correct but currently maybe slower
        return None
    try:
        return update_settings_from_observations(node[keys[0]], keys[1:], obs)  # type: ignore[arg-type]
    except UnknownSettingError:
        return None


def add_default_charaterization_settings(
    settings: SettingNode, components: list[str], chip_topology: ChipTopology
) -> None:
    """Create and populate the default characterization nodes in the settings.

    Note: this function should be called only after :func:`add_pulse_settings` is called (or the gate calibration
    nodes are otherwise created).

    Args:
        settings: the experiment settings node.
        components: the component names.
        chip_topology: the chip topology.

    """
    # create the default gate characterization nodes
    _DEFAULT_CHARACTERIZATION_GATE_PROPERTIES_RECIPES = [
        (["measure"], None, [None], MEASURE_CHARACTERIZATION_PARAMETERS),
        (["measure_fidelity"], None, [None], MEASURE_CHARACTERIZATION_PARAMETERS),
        (["cz", "move", "flux_pulse"], None, ["coupler", "qubit"], FLUX_PULSE_GATE_CHARACTERIZATION_PARAMETERS),
        (["rz_physical"], None, ["qubit"], FLUX_PULSE_GATE_CHARACTERIZATION_PARAMETERS),
        (["cz", "move", "flux_pulse"], None, ["qubit"], [FLUX_GATE_DETUNING]),
        (["cz"], None, [None], [CPHASE_DERIVATIVE, CPHASE_PI_DISTANCE]),
        (["move"], None, ["coupler", "qubit"], [RZ_ANGLE]),
        (["cz", "move"], ["crf_acstarkcrf", "slepian_acstarkcrf"], [None], [AC_STARK_QUBIT_TO_SWEEP]),
        (["lru"], ["f0g1"], [None], [F0G1_PULSE_STARK_SHIFT_POLYNOMIAL_COEFFICIENTS]),
    ]
    add_characterization_gate_properties_settings(
        settings,
        _DEFAULT_CHARACTERIZATION_GATE_PROPERTIES_RECIPES,  # type: ignore
    )
    # create the model characterization nodes
    _DEFAULT_CHARACTERIZATION_MODEL_RECIPES = [
        (["q", "c", "r"], [None], SINGLE_QUBIT_MODEL_CHARACTERIZATION_PARAMETERS),
        (["q", "c", "r"], ["readout"], SINGLE_QUBIT_READOUT_MODEL_CHARACTERIZATION_PARAMETERS),
        (["r"], [None], [COMPONENT_FREQUENCY]),
        (["cq"], [None], SINGLE_COUPLER_MODEL_CHARACTERIZATION_PARAMETERS),
    ]
    add_characterization_model_settings(
        settings,
        components,
        chip_topology,
        _DEFAULT_CHARACTERIZATION_MODEL_RECIPES,  # type: ignore
    )
    # create the custom characterization node (empty by default)
    settings.add_for_path([SettingNode(name="other")], path="characterization")


def add_characterization_gate_properties_settings(
    settings: SettingNode,
    recipes: list[tuple[list[str], list[str] | None, list[str], list[Parameter]]],
) -> None:
    """Generate & add the gate properties characterization setting nodes for given gates/implementations/sub-nodes.

    The added nodes are controlled by the provided ``recipes`` list. Each ``recipes`` list element is a tuple of the
    form ``(<list of gate names>, <limit to implementations>, <subnodes to create under locus>, <list of parameters>)``.
    Example: ``(["measure", "prx"], ["constant", "drag_crf"], ["stuff"], my_parameters)`` would add Settings
    corresponding to``my_parameters`` for all the loci of "measure" and "prx" gates, but only for the implementations
    ``["constant", "drag_crf"]``, and the Settings will be added under a sub-node "stuff" under each locus. ``None``
    as the limiting implementations means implementations are not limited, the nodes will be added for all
    implementations.

    Args:
        settings: the settings tree.
        recipes: determines which nodes to add.

    """
    # aggregate the recipes per gate/impls for better performance
    recipes_for_gate: dict = defaultdict(list)
    for recipe in recipes:
        for gate in recipe[0]:
            recipes_for_gate[(gate, tuple(recipe[1]) if recipe[1] is not None else None)].append(recipe[2:4])

    for gate_and_impls, rcps in recipes_for_gate.items():
        gate, impls = gate_and_impls
        locus_paths = settings.get_locus_node_paths_for(gate, implementations=impls)
        for locus_path in locus_paths:
            for prefixes, params in rcps:
                for prefix in prefixes:
                    path = f"characterization.gate_properties.{locus_path}" + (f".{prefix}" if prefix else "")
                    settings.add_for_path(params, path)


def add_characterization_model_settings(
    settings: SettingNode,
    components: list[str],
    chip_topology: ChipTopology,
    recipes: list[tuple[list[str | None], list[str], list[Parameter]]],
) -> None:
    """Generate & add the gate properties characterization model nodes for given components.

    The added nodes are controlled by the provided ``recipes`` list. Each ``recipes`` list element is a tuple of the
    form ``(<component types>, <subnodes to create under locus>, <list of parameters>)``. Here, `<component types>`
    is a list containing a combination of the keys `"q"` (qubits), `"c"` (couplers), `"r"` (computational resonators),
    and `"p"` (probe lines). Example: the recipe ``(["q", "r"], ["foo", "bar"], my_params)`` would add Settings
    corresponding to ``my_params`` under each of the provided ``components`` that are either qubits or resonators, and
    the Settings will be added (duplicated) under sub-nodes "foo" and "bar".

    Args:
        settings: the settings tree.
        components: the settings components.
        chip_topology: the chip topology.
        recipes: determines which nodes to add.

    """
    for component_types, prefixes, parameters in recipes:
        all_components = []
        if "q" in component_types:
            all_components.extend(list(chip_topology.qubits_sorted))
        if "c" in component_types:
            all_components.extend(list(chip_topology.couplers_sorted))
        if "r" in component_types:
            all_components.extend(list(chip_topology.computational_resonators_sorted))
            all_components.extend(
                [
                    qubit
                    for qubit in chip_topology.qubits_sorted
                    if qubit in settings.controllers.subtrees and "drive" not in settings.controllers[qubit].subtrees
                ]
            )
        if "p" in component_types:
            all_components.extend(list(chip_topology.probe_lines_sorted))
        filtered_components = [c for c in components if c in all_components]
        if "cq" in component_types:
            filtered_components.extend(
                [
                    f"{coupler}.{qubit}"
                    for coupler, qubits in chip_topology.coupler_to_components.items()
                    for qubit in qubits
                    if coupler in components
                ]
            )
        for component in filtered_components:
            for prefix in prefixes:
                path = f"characterization.model.{component}" + (f".{prefix}" if prefix else "")
                settings.add_for_path(parameters, path)


def add_pulse_settings(settings: SettingNode, ops_table: dict[str, Any], chip_topology: ChipTopology) -> None:
    """Add the gate parameters and gate definition nodes to the settings tree.

    The required nodes are deduced from ``ops_table`` (that contains the quantum operation definitions)
    and the chip topology, which contains the locus mapping graphs (any custom locus mappings required must be
    added there). The symmetricity of the gate implementations is taken into account. The gate parameter nodes
    are added under the node ``"gates"`` in the settings tree, and the gate definition nodes under the
    "gate_definitions" node.

    Args:
        settings: The settings tree under which to add the gate calibration data.
        ops_table: Quantum operation definitions to use. Gates & implementations must be included here
            in order for them to be added to the settings tree.
            Derived from ``experiment.yml::experiment.gate_definitions`` and the default iqm-pulse operations,
            then filtered by ``experiment.yml::experiment.gates_used``.
        chip_topology: The chip topology.

    """
    # only build gate parameter nodes for loci whose components have controller nodes in the tree
    components_in_settings = frozenset(
        component for component in chip_topology.all_components if component in settings.controllers.children
    )
    settings.subtrees["gates"] = _create_gate_setting_nodes(
        ops_table, chip_topology, limit_locus_by=components_in_settings
    )  # for performance, the paths & names are generated manually for gate nodes
    settings["gate_definitions"] = _create_gate_definition_nodes(ops_table)


def _create_gate_definition_nodes(
    ops_table: dict[str, Any],
    override_default_implementations_by: dict[str, str] | None = None,
) -> SettingNode:
    """Build the ``gate_definitions`` node for the given QuantumOps.

    Args:
        ops_table: Quantum operations to include.
        override_default_implementations_by: Optional mapping from operation names to the names of
            their default implementations. Any entry overrides the corresponding hardcoded default.

    Returns:
        The ``gate_definitions`` node.

    """
    override_default_implementations_by = override_default_implementations_by or {}
    operations_dict: dict[str, SettingNode] = {}
    for operation_name, operation in ops_table.items():
        op_implementations: dict[str, SettingNode] = {}
        for impl_name in operation.implementations:
            # implementation properties
            impl_class = operation.implementations[impl_name]
            impl_settings: dict[str, Setting] = {}
            impl_settings_map = [
                (Parameter("class_name", "Class name", data_type=DataType.STRING), impl_class.__name__),
                (Parameter("symmetric", "Is symmetric", data_type=DataType.BOOLEAN), impl_class.symmetric),
                (
                    Parameter("needs_calibration", "Needs calibration", data_type=DataType.BOOLEAN),
                    impl_class.needs_calibration(),
                ),
            ]
            for param, value in impl_settings_map:
                impl_settings[param.name] = Setting(param, value, read_only=True, source=GATE_DEFAULTS_SOURCE)

            # find if this implementation is the default for some specific loci
            specified_loci: list[str] = []
            for specified_locus, specified_default in operation.defaults_for_locus.items():
                if impl_name == specified_default:
                    specified_loci.append("__".join(specified_locus))
            impl_settings["override_default_for_loci"] = Setting(
                Parameter(
                    "override_default_for_loci",
                    "Override default for loci",
                    data_type=DataType.STRING,
                    collection_type=CollectionType.LIST,
                ),
                specified_loci,  # type: ignore[arg-type]
                source=QUANTUM_OP_DEFAULTS_SOURCE,
            )
            op_implementations[impl_name] = SettingNode(name=impl_name, settings=impl_settings)

        if op_implementations:
            # op has implementations, set up operation properties
            op_settings: dict[str, Setting] = {}
            default_override = override_default_implementations_by.get(operation_name, None)
            op_settings["default_implementation"] = Setting(
                Parameter(
                    "default_implementation",
                    "Default gate implementation",
                    data_type=DataType.STRING,
                ),
                default_override or operation.default_implementation,
                source=QUANTUM_OP_DEFAULTS_SOURCE,
            )
            op_settings_map = [
                (Parameter("symmetric", "Is symmetric", data_type=DataType.BOOLEAN), operation.symmetric),
                (Parameter("arity", "Arity", data_type=DataType.INT), operation.arity),
                (Parameter("params", "Parameters", data_type=DataType.ANYTHING), tuple(operation.params.keys())),
                (Parameter("factorizable", "Factorizable", data_type=DataType.BOOLEAN), operation.factorizable),
            ]
            for param, value in op_settings_map:
                op_settings[param.name] = Setting(param, value, read_only=True, source=GATE_DEFAULTS_SOURCE)

            operations_dict[operation_name] = SettingNode(
                name=operation_name, settings=op_settings, subtrees=op_implementations
            )
    return SettingNode(name="gate_definitions", subtrees=operations_dict)


def _validate_gate_definitions_against_settings(
    gate_definitions: dict[str, QuantumOp], settings: SettingNode
) -> dict[str, QuantumOp]:
    """Validate the current gate_definitions against the settings recorded with a previous experiment run.

    The data from a previous experiment run might have been collected in a different configuration. This might mean
    the automatically created `gate_definitions` do not correspond to ones used with that run. Check both settings
    and the `gate_definitions`, trim the latter and raise warnings if anything is missing.

    Note: default implementations and defaults for locus in the `gate_definitions' are not updates, since they can
    always be resolved from the settings.

    Args:
        gate_definitions: Mapping of gate names to their QuantumOps
        settings: Setting tree used to run the experiment

    Returns:
        Updated gate definitions

    """
    # for unit testing
    if not hasattr(settings, "gate_definitions"):
        return gate_definitions

    default_gates = build_quantum_ops({})

    missing_gates = set(settings.gate_definitions.children.keys()) - set(gate_definitions.keys())
    for gate_name in missing_gates.copy():
        if default_gate_definition := default_gates.get(gate_name):
            gate_definitions[gate_name] = copy.deepcopy(default_gate_definition)
            missing_gates.remove(gate_name)
    gates_to_remove = set(gate_definitions.keys()) - set(settings.gate_definitions.children.keys())
    missing_implementations = set()
    for gate_name in gates_to_remove:
        del gate_definitions[gate_name]

    for gate_name, quantum_op in gate_definitions.items():
        implementation_names = set(
            child_name
            for child_name, child in settings.gate_definitions[gate_name].children.items()
            if isinstance(child, SettingNode)
        )
        missing_gate_implementations = implementation_names - set(quantum_op.implementations.keys())
        if gate_name in default_gates:
            for impl_name in missing_gate_implementations.copy():
                if default_gate_implementation := default_gates[gate_name].implementations.get(impl_name):
                    gate_definitions[gate_name].implementations[impl_name] = default_gate_implementation
                    missing_gate_implementations.remove(impl_name)

        implementations_to_remove = set(quantum_op.implementations.keys()) - implementation_names
        missing_implementations.update(set(f"{gate_name}.{impl_name}" for impl_name in missing_gate_implementations))
        for impl_name in implementations_to_remove:
            del gate_definitions[gate_name].implementations[impl_name]

    if missing_gates or missing_implementations:
        logger.warning(
            "The following gates: %s and/or implementations: %s are present in the settings used to run"
            " the experiment, but missing from the QuantumOp table. This usually means the gates are"
            " missing from the current experiment configuration, or were removed using the 'gates_used' attribute."
            " If the analysis breaks, update the configuration and load the data again.",
            missing_gates,
            missing_implementations,
        )

    return gate_definitions


def _create_gate_setting_nodes(
    ops_table: dict[str, QuantumOp],
    chip_topology: ChipTopology,
    *,
    path: tuple[str, ...] = (),
    limit_gates_by: tuple[str, ...] = (),
    forbidden_impls: tuple[str, ...] = (),
    limit_locus_by: Iterable[str] | None = None,
    custom_calibration: bool = False,
) -> SettingNode:
    """Build a gate calibration settings node.

    Args:
        ops_table: QuantumOps (with their GateImplementations) to include in the settings.
        chip_topology: QPU topology.
        path: Settings path prefix for the generated node. Used for CompositeGates that have recursive gate calibration
            nodes.
        limit_gates_by: Only include these gates from ``ops_table``. If empty, do no filtering.
        forbidden_impls: Do not include these implementations. If empty, do no filtering.
        limit_locus_by: Only include gate loci which are subsets of these QPU components. If None, do no filtering.
        custom_calibration: True iff this gate settings node is the custom member calibration of a CompositeGate.
            It may not have any default values from the gate implementations, since
            they would always override the global calibration values of the member gates.

    Returns:
        Gate calibration settings node.

    """
    # TODO most of this function should be in iqm.pulse, here we should just convert the nested dict
    # it returns into a SettingNode.
    gates_settings: dict[str, SettingNode] = {}
    for operation_name, operation in ops_table.items():
        if limit_gates_by and operation_name not in limit_gates_by:
            continue
        o_settings: dict[str, SettingNode] = {}
        for impl_name, impl_class in operation.implementations.items():
            if impl_name in forbidden_impls:
                continue
            # TODO get the possible loci directly from impl_class
            locus_mapping_name = impl_class.get_locus_mapping_name(operation_name, impl_name)
            oi_settings: dict[str, SettingNode] = {}
            loci = chip_topology.get_loci(name=locus_mapping_name, default_mapping_dimension=operation.arity)
            limited_loci = _get_loci(loci, impl_class.symmetric, limit_locus_by)
            for locus in limited_loci:
                # TODO the locus normalization should be done by iqm-pulse so the rules are all in one place
                sorted_locus = sort_components(locus) if impl_class.symmetric else locus
                locus_str = locus_to_locus_str(sorted_locus) if sorted_locus else GLOBAL_LOCUS_STRING_REPRESENTATION
                locus_path = path + ("gates", operation_name, impl_name, locus_str)

                # Create a SettingNode with the default values of the GateImplementation
                oil_settings = impl_class.get_parameters(
                    locus=locus, path=locus_path, use_defaults=not custom_calibration
                )
                # TODO iqm-pulse should not use Settings or Parameters, they should be created here.
                oil_settings.set_source(GATE_DEFAULTS_SOURCE)

                if issubclass(impl_class, CompositeGate) and (customizable_members := impl_class.customizable_gates):
                    # add custom calibration for member gates
                    # TODO this code should be in CompositeGate.get_parameters
                    node_name = ".".join(locus_path)
                    oil_settings_members = SettingNode.fast_construct(
                        name=node_name,
                        path=node_name,
                        gates=_create_gate_setting_nodes(
                            ops_table,
                            chip_topology,
                            path=locus_path,
                            limit_gates_by=customizable_members,
                            # a composite impl cannot be registered under itself, that would lead to infinite recursion
                            forbidden_impls=forbidden_impls + (impl_name,),
                            limit_locus_by=locus,
                            custom_calibration=True,
                        ),
                    )
                    oil_settings = SettingNode.merge(
                        oil_settings,
                        oil_settings_members,
                        merge_nones=True,
                        align_name=False,
                        deep_copy=False,
                    )
                oi_settings[locus_str] = oil_settings

            o_settings[impl_name] = SettingNode.fast_construct(
                name=".".join(path + ("gates", operation_name, impl_name)),
                subtrees=oi_settings,
                generate_paths=False,
            )

        if o_settings:
            gates_settings[operation_name] = SettingNode.fast_construct(
                name=".".join(path + ("gates", operation_name)),
                subtrees=o_settings,
                generate_paths=False,
            )
    return SettingNode.fast_construct(
        name=".".join(path + ("gates",)),
        subtrees=gates_settings,
        generate_paths=False,
        path="gates",
    )


def _get_loci(
    possible_loci: list[tuple[str, ...] | frozenset[str]],
    is_symmetric: bool,
    parent_locus: Iterable[str] | None = None,
) -> list[tuple[str, ...] | frozenset[str]]:
    """Allowed loci for a particular implementation of a QuantumOp.

    Process loci from chip topology's locus mapping to be:

    1. in the correct form (``frozenset`` vs. ``tuple``)  based on the gate implementation's symmetricity.
    2. limited by a possible parent gate locus in the case this gate node is a part of a composite gate.

    Args:
        possible_loci: Possible loci from a locus mapping.
        is_symmetric: True iff the implementation is symmetric.
        parent_locus: Only return loci which are subsets of this locus. If None, do no filtering.

    Returns:
        Allowed loci.

    """
    loci: list[tuple[str, ...] | frozenset[str]]
    if not possible_loci or len(possible_loci[0]) < 2:
        loci = possible_loci
    else:
        loci = []
        for locus in possible_loci:
            if is_symmetric:
                if isinstance(locus, tuple):
                    loci.append(frozenset(locus))
                else:
                    loci.append(locus)
            elif isinstance(locus, frozenset):
                loci.extend(list(permutations(locus)))  # type: ignore
            else:
                loci.append(locus)
    # optionally limit the loci by parent gate's locus
    if parent_locus is None:
        return loci
    parent_locus_set = set(parent_locus)
    return [locus for locus in loci if set(locus).issubset(parent_locus_set)]


# TODO: any meaningful setting defaults should probably be added into the gate implementations themselves?
def set_default_pulse_setting_values(
    pulse_settings: SettingNode, path_to_value: dict[str, Any], max_depth: int | None = None
) -> None:
    """Set default values for pulse settings.

    Args:
        pulse_settings: The pulse settings tree
        path_to_value: settings tree paths mapped to the values which should be set to them. Will find the first
            occurrence of the given path under the settings tree. `"*"`-char acts as a wildcard that will apply the
            values to all settings under a specified path.
        max_depth: optional maximum depth to which stop the iteration.

    """

    def _set_for(
        node: SettingNode,
        path: list[str],
        value_to_set: Any,
        depth: int,
        beginning_found: bool = False,
    ) -> None:
        for child in node.children:
            if max_depth is None or depth <= max_depth:
                if isinstance(node[child], SettingNode):
                    if path[0] == child:
                        _set_for(
                            node=node[child],  # type: ignore[arg-type]
                            path=path[1:],
                            value_to_set=value_to_set,
                            depth=depth + 1,
                            beginning_found=True,
                        )
                    elif not beginning_found:
                        _set_for(node=node[child], path=path, value_to_set=value_to_set, depth=depth + 1)  # type: ignore[arg-type]
                elif len(path) == 1 and (path[0] == child or path[0] == "*"):
                    node[child] = value_to_set
                    node[child]._source = GATE_DEFAULTS_SOURCE

    for path, value in path_to_value.items():
        _set_for(node=pulse_settings, path=path.split("."), value_to_set=value, depth=0)


def gates_data_from(gate_calib_data: SettingNode) -> dict[str, Any]:
    """Convert the ``gates`` branch of the settings tree into gate calibration data expected by :mod:`iqm.pulse`.

    Args:
        gate_calib_data: The ``gates`` branch of the settings tree.

    Returns:
        The gate calibration data in :mod:`iqm.pulse` format.

    """
    # TODO limit the use of this function in exa-core and avoid it entirely in exa-experiments and gbc-graphs.
    # Experiments should not use iqm.pulse internals or access the converted cal data. It should only be passed to
    # (and accessed through) ScheduleBuilder.

    def _convert_settings(node: SettingNode) -> dict[str, Any]:
        """Recursively pick out the values of the settings into a nested dict, ignoring other attributes."""
        converted = {}
        for child in node.children:
            if isinstance(node[child], SettingNode):
                converted[child] = _convert_settings(node[child])  # type: ignore[arg-type]
            else:
                converted[child] = node[child].value  # type: ignore[assignment]
        return converted

    op_calib_data = _convert_settings(gate_calib_data)
    return fix_loci(op_calib_data)


def fix_loci(op_calib_data: dict[str, Any]) -> dict[str, Any]:
    """Modify the calibration data dict by turning the locus str representations into tuples
    and simplify CompositeGate calibration nodes.

    Args:
        op_calib_data: Nested dict of gate calibration data built out of the ``gates`` branch of the settings tree

    Returns:
        The gate calibration data in :mod:`iqm.pulse` format.

    """
    fixed_data: dict[str, dict] = {}
    for op_name, o_data in op_calib_data.items():
        fixed_o_data: dict[str, dict] = {}
        for impl_name, oi_data in o_data.items():
            fixed_oi_data: dict[LocusTuple, dict] = {}
            for locus_str, oil_data in oi_data.items():
                if locus_str == GLOBAL_LOCUS_STRING_REPRESENTATION:
                    locus: LocusTuple = tuple()
                else:
                    locus = locus_str_to_locus(locus_str)
                if "gates" in oil_data:
                    # collapse the "gates" node
                    composite_node = oil_data.pop("gates")
                    fixed_oi_data[locus] = fix_loci(composite_node)
                    fixed_oi_data[locus] |= oil_data
                else:
                    fixed_oi_data[locus] = oil_data
            fixed_o_data[impl_name] = fixed_oi_data
        fixed_data[op_name] = fixed_o_data
    return fixed_data


def add_move_detuning_to_implementations(
    settings: SettingNode, observation_handler: ObservationHandlerBase, computational_resonators: list[str]
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
