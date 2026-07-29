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
"""Convert quantum circuits into instruction schedules.

This is the core module of ``CPC``. It contains the functionality to define a compiler, whose job is to
convert quantum circuits and calibration data into configuration settings and instruction schedules that
can be executed by the IQM server on quantum hardware.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
import itertools
import logging
import pathlib
from typing import TYPE_CHECKING, Any, TypeAlias
import warnings

# TODO see if we can avoid pulla depending on ipython since SC depends on pulla (or make it an extra dependency)
from IPython.core.display import HTML
from IPython.core.display_functions import display
import jinja2

from exa.common.control.sweep.sweep import Sweep
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import NdSweep
import iqm.cpc.compiler._utils.settings as utils_settings
from iqm.cpc.compiler._utils.stages import (
    CIRCUIT_SWEEP_PARAMETER,
    generate_sweep_spot,
    get_default_channel_properties,
    get_full_spot_dict,
    get_stage_spot_dict,
    split_to_hard_and_soft_sweeps,
    update_stage_spot_dict,
    validate_inputs,
)
import iqm.cpc.compiler._utils.topology as utils_topology
from iqm.cpc.compiler.compilation_stage import (
    DEFAULT_STAGE_ARGS,
    CompilationStage,
    PullaInputType,
    StagesList,
    format_stages,
    resolve_circuit_function_and_stages,
)
from iqm.cpc.core.config import ComponentGrouping, get_conventional_control_mapping
from iqm.cpc.core.observation.observation_handler_base import ObservationHandlerBase
from iqm.cpc.core.observation.observation_loading_rules import RuleType
from iqm.pulse.builder import ScheduleBuilder, build_quantum_ops
from iqm.pulse.gate_implementation import (
    GateImplementation,
    Locus,
)
from iqm.pulse.gates import register_implementation, register_operation
from iqm.pulse.quantum_ops import QuantumOp

if TYPE_CHECKING:
    from exa.common.qcm_data.chip_topology import ChipTopology

STATION_CONFIGURATION_SOURCE = {"type": "configuration_source", "configurator": "station_configuration"}
EXPERIMENT_DEFAULTS_SOURCE = {"type": "configuration_source", "configurator": "experiment_defaults"}

cpc_logger = logging.getLogger("cpc")

Components: TypeAlias = list[str] | list[tuple[str, ...]] | list[list[tuple[str, ...]]]
"""Same as :class:`ComponentGrouping` but using a normal :class:`list`."""


@dataclass
class CompilerOptions:
    """Options for Compiler."""

    idempotent_stages: bool = False
    """Set to ``True`` for idempotent stages (the context and the circuits are deepcopied between each stage pass)."""
    update_settings: bool = False
    """If True, the settings tree will be updated with the sweep spot values."""


@dataclass
class CompilerStages:
    """Combines all compilation stages under a single object for easy access."""

    circuit_stages: StagesList
    """Circuit-level (and above) stages."""
    pulse_stages: StagesList
    """Pulse-level stages."""
    final_stages: StagesList
    """Final stages."""
    pp_stages: StagesList
    """Post-processing stages."""


class Compiler:
    """Compile quantum circuits into jobs that can be submitted to a quantum computer.

    Args:
        dut_label: DUT label of the chip being used.
        loading_rules: Observation loading rules.
        chip_topology: Connectivity of the quantum chip.
        software_version_set_id: An integer ID representing the software versions in the current python environment.
        component_mapping: Mapping of logical QPU component names to physical QPU component names.
            ``None`` means the identity mapping.
        station_control_settings: Settings representing the device and station controllers.
        controller_mapping: Dictionary that maps physical QPU component names to their device controller names.
            The dictionary is of the form: ``{<component_name>: {<operation_name>: <controller name>}}``,
            where operation is one of the following: "drive", "readout", "flux"
            (not all components have all operations supported).
        gate_definitions: Names of quantum operations mapped to their definitions, see :class:`.QuantumOp`.
        observation_handler: Observation handler.
        circuit_stages: Compilation stages to use in processing gate-level (and above) circuits. ``None`` means none.
            Note that meaningful circuit compilation requires at least some stages. These stages are sweepable in the
            compilation loop via their stage settings.
        pulse_stages: Compilation stages to use in processing `TimeBox`-level (and below) circuits. ``None`` means none.
            Note that meaningful circuit compilation requires at least some stages. These stages are sweepable in the
            compilation loop via their stage settings.
        final_stages: Compilation stages to use in creating the final run request. These stages are not sweepable.
        pp_stages: Post-processing stages to use.
        compiler_options: General options to define the compiler behaviour.
        name: The name of this compiler.

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dut_label: str,
        loading_rules: Sequence[RuleType],
        chip_topology: ChipTopology,
        software_version_set_id: int,
        component_mapping: dict[str, str] | None,
        station_control_settings: SettingNode,
        controller_mapping: dict[str, dict[str, str]] | None = None,
        gate_definitions: dict[str, QuantumOp] | None = None,
        observation_handler: ObservationHandlerBase | None = None,
        circuit_stages: list[CompilationStage] | None = None,
        pulse_stages: list[CompilationStage] | None = None,
        final_stages: list[CompilationStage] | None = None,
        pp_stages: list[CompilationStage] | None = None,
        compiler_options: CompilerOptions | None = None,
        name: str = "IQM Compiler",
    ) -> None:
        self.dut_label = dut_label
        self.name = name
        self.chip_topology = chip_topology
        self.station_control_settings = station_control_settings
        self.controller_mapping = (
            deepcopy(controller_mapping)
            if controller_mapping
            else get_conventional_control_mapping(chip_topology, station_control_settings)
        )
        self.gate_definitions = deepcopy(gate_definitions) if gate_definitions else build_quantum_ops({})
        self.component_mapping = component_mapping
        self._default_loading_rules = list(loading_rules)
        self.observation_handler = observation_handler or ObservationHandlerBase([])
        self.circuit_stages = StagesList(circuit_stages or [])
        self.pulse_stages = StagesList(pulse_stages or [])
        self.final_stages = StagesList(final_stages or [])
        self.pp_stages = StagesList(pp_stages or [])
        self.compiler_options = compiler_options if compiler_options else CompilerOptions()
        self.stages = CompilerStages(self.circuit_stages, self.pulse_stages, self.final_stages, self.pp_stages)

        self._software_version_set_id = software_version_set_id

        if self.chip_topology.all_components:
            self._available_channels = utils_topology.get_component_to_available_channels_mapping(
                self.chip_topology,
                self.station_control_settings,
                self.controller_mapping,
            )
            utils_topology.set_single_component_mapping_in_init(self.chip_topology, self._available_channels)
            utils_topology.set_fast_cz_mapping_in_init(self.chip_topology, self._available_channels)
            utils_topology.set_qubits_through_resonator_mapping_in_init(chip_topology)
            # register any custom locus mappings under the default name for the implementation
            # TODO: ChipTopology might be the wrong place for locus mappings, consider moving them in gate_definitions
            for op_name, quantum_op in self.gate_definitions.items():
                for impl_name, impl_class in quantum_op.implementations.items():
                    if (
                        mapping := impl_class.get_custom_locus_mapping(self.chip_topology, self._available_channels)
                    ) is not None:
                        self.chip_topology.set_locus_mapping(f"{op_name}.{impl_name}", mapping)
        else:
            self._available_channels = {}

    def get_settings(  # noqa: PLR0913
        self,
        circuits: PullaInputType | None = None,
        timeboxes: PullaInputType | None = None,
        circuit_stages: list[CompilationStage] | None = None,
        pulse_stages: list[CompilationStage] | None = None,
        final_stages: list[CompilationStage] | None = None,
        qubits: list[str] | None = None,
        couplers: list[str] | None = None,
        computational_resonators: list[str] | None = None,
        create_characterization_nodes: bool = True,
    ) -> SettingNode:
        """Build a settings tree for the Compiler.

        The user can modify the returned settings tree (containing the default settings) as they
        wish, and then pass it to :meth:`compile`. One typical thing to set is the number of
        **shots** or repetitions of the executed circuit, which is easiest done using
        :meth:`.SettingNode.set_shots`.

        Args:
            circuits: Circuit-level input to be compiled.
            timeboxes: TimeBox-level input to be compiled. Either ``circuits`` or ``timeboxes`` must be provided.
            circuit_stages: Circuit-level (and above) compilation stages. If not provided, ``self.circuit_stages``
                will be used.
            pulse_stages: TimeBox-level (and below) compilation stages. If not provided, ``self.pulse_stages`` will be
                used.
            final_stages: Station-control job finalization stages. If not provided, ``self.final_stages`` will be used.
            qubits: Names of the qubits to get the settings for.
            couplers: Names of the couplers to get the settings for.
            computational_resonators: Names of the computational resonators to get the settings for.
            create_characterization_nodes: If ``True``, characterization nodes will be added the to the settings.
                Characterization nodes are not required in the standard compilation and leaving them out can give
                some performance gains in these cases.

        Returns:
            The settings tree.

        """
        if qubits is None:
            qubits = list(self.chip_topology.qubits_sorted)
        if computational_resonators is None:
            computational_resonators = list(self.chip_topology.computational_resonators_sorted)
        if couplers is None:
            couplers = list(self.chip_topology.get_connecting_couplers(qubits + computational_resonators))
        if circuit_stages is None:
            circuit_stages = copy(self.circuit_stages)  # type:ignore[assignment]
        if pulse_stages is None:
            pulse_stages = copy(self.pulse_stages)  # type:ignore[assignment]
        if final_stages is None:
            final_stages = copy(self.final_stages)  # type:ignore[assignment]

        data, timebox_input = validate_inputs(circuits, timeboxes)
        if timebox_input:
            stages = resolve_circuit_function_and_stages(pulse_stages, data) + final_stages  # type:ignore[operator, arg-type]
        else:
            stages = resolve_circuit_function_and_stages(circuit_stages, data) + pulse_stages + final_stages  # type:ignore[operator, arg-type]

        settings, probe_lines = utils_settings.get_controller_settings(
            qubits + couplers + computational_resonators,
            self.chip_topology,
            self.controller_mapping,
            self.station_control_settings,
        )

        utils_settings.add_pulse_settings(
            settings,
            self.gate_definitions,
            self.chip_topology,
        )
        utils_settings.set_default_pulse_setting_values(settings["gates"], {"rz.*": 0.0}, max_depth=4)  # type: ignore[arg-type]
        if create_characterization_nodes:
            utils_settings.add_default_charaterization_settings(
                settings,
                qubits + couplers + computational_resonators + probe_lines,
                self.chip_topology,
            )
        utils_settings.add_stage_setting_options(settings, stages, DEFAULT_STAGE_ARGS)
        self._update_settings_from_observations(settings)
        utils_settings.add_move_detuning_to_implementations(
            settings, self.observation_handler, computational_resonators
        )

        utils_settings.update_settings_single_shot_threshold(qubits, couplers, settings, self.gate_definitions)
        settings._mark_index_dirty()
        return settings

    def _update_settings_from_observations(self, settings: SettingNode) -> None:
        """Update settings from observations using ``self.observation_handler``."""
        self.observation_handler.populate_from(settings, load_rules=self._default_loading_rules)
        self.observation_handler.load_observations()
        overrides = self.observation_handler.value_dict(return_full_observation=True)
        for path, obs in overrides.items():
            utils_settings.update_settings_from_observations(settings, path.split("."), obs)

    def compiler_context(self, components: ComponentGrouping | None, settings: SettingNode, **kwargs) -> dict[str, Any]:
        """Return initial compiler context dictionary.

        Used automatically by :meth:`compile`.
        """
        return {
            "dut_label": self.dut_label,
            "components": components,
            "settings": settings,
            "component_mapping": self.component_mapping,
            "chip_topology": self.chip_topology,
            "software_version_set_id": self._software_version_set_id,
            "timebox_input": False,
        } | kwargs

    def compile(
        self,
        circuits: PullaInputType | None = None,
        timeboxes: PullaInputType | None = None,
        components: Components | None = None,
        settings: SettingNode | None = None,
        sweeps: NdSweep | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[PullaInputType, dict[str, Any]]:
        """Run all compiler stages.

        Initial context will be derived using :meth:`compiler_context` unless a custom
        context dictionary is provided.

        Args:
            circuits: Circuit-level input to be compiled.
            timeboxes: TimeBox-level input to be compiled. Either ``circuits`` or ``timeboxes`` must be provided. If
                this argument is provided, the stages in ``self.circuit_stages`` will not be run.
            components: Apply the circuits on these active components. If logical component names are used,
                ``self.component_mapping`` must contain logic for mapping them into physical ones. If ``components`` is
                ``None``, the active components used in the circuits can be automatically resolved in a compiler pass.
            settings: The settings tree to use. If None, the default settings tree will be generated.
                See :meth:`get_settings`.
            sweeps: The sweeps to perform in the job. If None, the job will consist of one trivial sweep spot.
            context: Custom initial compiler context dictionary. Contents are updated into the default context given
                by :meth:`.compiler_context`.

        Returns:
            Compiled `data, final context.

        """
        components = ComponentGrouping(components) if components else None
        settings = settings if settings else self.get_settings(circuits=circuits, timeboxes=timeboxes)
        if components is not None:
            components = utils_settings.limit_components_by_settings(components, settings, self.chip_topology)
        compiler_context = self.compiler_context(components, settings) | (context or {})
        executable, timebox_input = validate_inputs(circuits, timeboxes)
        compiler_context["timebox_input"] = timebox_input
        stages = self.circuit_stages + self.pulse_stages if not timebox_input else self.pulse_stages
        cpc_logger.info("Running compilation stages...")
        circuits, compiler_context = self.run_stages(
            executable,
            compiler_context,
            stages=stages,
            sweeps=sweeps,
        )
        return self.finalize(circuits, compiler_context, self.final_stages)

    def run_stages(  # noqa: PLR0912,PLR0915
        self,
        data: PullaInputType,
        context: dict[str, Any],
        stages: list[CompilationStage] | None = None,
        sweeps: NdSweep | None = None,
    ) -> tuple[Iterable[Any], dict[str, Any]]:
        """Run the circuit- and pulses-level stages on the given data.

        Circuit- and pulse-level stages can be swept as per the provided sweeps. The provided hard sweeps determine
        the dimensions of the final Playlist such that one segment (circuit) is generated per hard sweep spot.

        Args:
            data: The data to compile.
            context: Custom initial compiler context dictionary.
            sweeps: Sweeps to run on ``data``.
            stages: compilation stages to be run on ``data``. If not provided,
                ``self.circuit stages + self.pulse_stages`` will be used.
                Each stage may make modifications to ``context`` before it is passed to the next stage.
            sweeps: Sweeps to run on ``data``. Can contain both hard- and soft sweeps.

        Returns:
            Processed data, final context.

        """
        if stages is None:
            stages = self.circuit_stages + self.pulse_stages  # type:ignore[assignment]
        stages = resolve_circuit_function_and_stages(stages, data)
        stages = format_stages(stages, idempotent=self.compiler_options.idempotent_stages)
        sweeps = sweeps.copy() if sweeps else []
        settings = context["settings"]
        context["inputted_components"] = copy(context["components"])

        # create builder
        channel_properties, qubit_to_channel = get_default_channel_properties(settings, self.chip_topology)
        gate_definitions = utils_settings.assign_default_gate_implementations_from(
            settings.gate_definitions, gate_definitions=deepcopy(self.gate_definitions)
        )
        builder = ScheduleBuilder(
            gate_definitions,
            utils_settings.gates_data_from(settings.gates),
            self.chip_topology,
            channel_properties,
            component_channels=qubit_to_channel,
        )
        context["builder"] = builder

        # split sweeps to hard and soft sweeps
        sweeps = [sweep if isinstance(sweep, tuple) else (sweep,) for sweep in sweeps]
        soft_sweeps, hard_sweeps = split_to_hard_and_soft_sweeps(sweeps, settings)  # type:ignore[arg-type]
        stage_spot_dict = get_stage_spot_dict(settings.stages)
        stage_parameter_names = [s.name.split(".")[-1] for s in settings.stages.all_settings]
        # the actual compilation loop
        sweep_spots = itertools.product(
            *[generate_sweep_spot(tuple_of_sweeps) for tuple_of_sweeps in reversed(hard_sweeps)]
        )
        sweep_parameters = [[s.parameter for s in sweeps_tuple] for sweeps_tuple in reversed(hard_sweeps)]
        spot_data = []
        spot_idx = 0
        spot_contexts: dict[int, dict[str, Any]] = {}
        context["spot_contexts"] = spot_contexts
        gate_impl_caches: dict = {}
        for spot_idx, spot in enumerate(sweep_spots):
            data = []
            spot_context = copy(context)
            spot_context["sweep_spot"] = spot

            # update the spot dicts (stage options dict & the pulse spot dict)
            update_stage_spot_dict(
                stage_spot_dict,
                spot,
                sweep_parameters,
                stage_parameter_names,
                settings,
            )

            full_spot_dict, param_values = get_full_spot_dict(spot, sweep_parameters, settings)
            spot_context["sweep_spot_dict"] = full_spot_dict

            gate_impl_cache = gate_impl_caches.get(param_values)
            if "gates" in full_spot_dict:
                builder.inject_calibration(utils_settings.fix_loci(full_spot_dict["gates"]), cache=gate_impl_cache)  # type: ignore[arg-type]
                if param_values not in gate_impl_caches:
                    gate_impl_caches[param_values] = builder.get_cache(
                        ops=list(utils_settings.fix_loci(full_spot_dict["gates"]))
                    )  # type: ignore[arg-type]

            if "gate_definitions" in full_spot_dict:
                builder.inject_gate_definitions(full_spot_dict["gate_definitions"])

            spot_context["builder"] = builder

            if self.compiler_options.update_settings:
                settings.stages.set_from_dict(stage_spot_dict)
                settings.set_from_dict(full_spot_dict)
                spot_context["settings"] = settings

            # run the stages
            for stage in stages:  # type:ignore[union-attr, operator]
                cpc_logger.info('Running stage "%s"...', stage.name)
                data, spot_context = stage.run(data, spot_context, stage_spot_dict)

            spot_data.append(data)
            context["spot_contexts"][spot_idx] = spot_context
            # store the context components to the final top-level context as these cannot be yet determined effectively
            # in a standalone stage
            # TODO: implement performant component resolving from TimeBoxes so we don't need this
            if spot_context.get("components") is not None and context.get("components") is None:
                context["components"] = spot_context["components"]

        # gather sweep data to context
        context["soft_sweeps"] = soft_sweeps
        implicit_circuit_sweep_len = len(spot_data[0])  # TODO: handle ragged case

        # reorder the implicit circuit sweep as the outermost dim and combine the sweep spot results
        final_data = [spot[i] for i in range(implicit_circuit_sweep_len) for spot in spot_data]
        if len(final_data) != implicit_circuit_sweep_len * (spot_idx + 1):
            raise RuntimeError("Ragged circuit sweep dimensions are not yet supported.")
        if implicit_circuit_sweep_len > 1:
            hard_sweeps.append(
                (Sweep(parameter=CIRCUIT_SWEEP_PARAMETER, data=list(range(implicit_circuit_sweep_len))),)
            )
        context["hard_sweeps"] = hard_sweeps

        return final_data, context

    def finalize(
        self, data: PullaInputType, context: dict[str, Any], stages: list[CompilationStage] | None = None
    ) -> tuple[PullaInputType, dict[str, Any]]:
        """Run ``self.final_stages, i.e. finalize the payload to be sent for execution.``

        The final stages are not sweepable but their settings can be changed.

        Args:
            data: The circuit data for the final stages.
            context: The compiler context.
            stages: The final stages to be run on ``data``. If not provided, uses ``self.final_stages``.

        Returns:
            The finalized payload to be sent for execution and the compiler context.

        """
        stages = stages if stages is not None else self.final_stages  # type:ignore[assignment]
        settings = context["settings"]
        stage_spot_dict = get_stage_spot_dict(settings.stages)
        final_stages = format_stages(stages, idempotent=self.compiler_options.idempotent_stages)  # type:ignore[arg-type]
        for stage in final_stages:
            data, context = stage.run(data, context, stage_spot_dict)
        return data, context

    def post_process(
        self, data: PullaInputType, context: dict[str, Any], stages: list[CompilationStage] | None = None
    ) -> tuple[PullaInputType, dict[str, Any]]:
        """Run ``self.pp_stages``.

        Post-processing stages are not sweepable and their settings cannot be changed, but one may still pass stateful
        effects via the context.

        Args:
            context: The compiler context.
            data: The circuit data for the final stages.
            stages: The final stages to be run on ``data``. If not provided, uses ``self.final_stages``.

        Returns:
            Human-readable result data and the compiler context.

        """
        stages = stages if stages is not None else self.pp_stages  # type:ignore[assignment]
        cpc_logger.info("Running postprocessing stages...")
        pp_stages = format_stages(stages, idempotent=self.compiler_options.idempotent_stages)  # type:ignore[arg-type]
        for stage in pp_stages:
            data, context = stage.run(data, context, {})
        return data, context

    def set_default_implementation(self, gate_name: str, implementation_name: str) -> None:
        """Set the default implementation of a gate.

        Args:
            gate_name: Name of the gate.
            implementation_name: Name of the implementation to set as the default.

        """
        self.gate_definitions[gate_name].set_default_implementation(implementation_name)

    def set_default_implementation_for_loci(
        self, gate_name: str, implementation_name: str, loci: Iterable[Locus]
    ) -> None:
        """Set the default implementation for a gate for a specific loci.

        Args:
            gate_name: Name of the gate.
            implementation_name: Name of the implementation to set as the default for ``loci``.
            loci: Loci of the gate for which to set ``implementation_name`` as the default.

        """
        for locus in loci:
            self.gate_definitions[gate_name].set_default_implementation_for_locus(implementation_name, locus)

    def add_implementation(
        self,
        op_name: str,
        impl_name: str,
        impl_class: type[GateImplementation],
        *,
        set_as_default: bool = False,
        overwrite: bool = False,
        quantum_op: QuantumOp | None = None,
    ) -> None:
        """Adds a new implementation for a quantum operation (gate).

        Refreshes the compiler after adding a new implementation.

        Args:
            op_name: The name of the quantum operation for which to register a new implementation.
            impl_name: The "human-readable" name with which the new implementation will be found e.g. in settings.
            impl_class: The class of the new implementation to be added.
            set_as_default: Whether to set the new implementation as the default implementation for the operation.
            overwrite: If True, replaces any existing implementation of the same name for the operation.
            quantum_op: The quantum operation this gate represents. If a QuantumOp is given, it is used as is.
                If None is given and the same gate has been registered before, the previously registered properties are
                used. Existing operations cannot be replaced or modified.

        """
        if quantum_op is not None:
            register_operation(self.gate_definitions, quantum_op)
        register_implementation(
            operations=self.gate_definitions,
            op_name=op_name,
            impl_name=impl_name,
            impl_class=impl_class,
            set_as_default=set_as_default,
            overwrite=overwrite,
        )

    def show_stages(self, pass_name: str | None = None) -> Any:
        """Displays an interactive HTML representation of the compiler stages."""
        tmpl_path = pathlib.Path(__file__).parent
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader(tmpl_path))

        search_query = None

        if pass_name:
            query_lower = pass_name.lower()
            # Aggregate all stages to check for matches
            all_stages = self.circuit_stages + self.pulse_stages + self.final_stages + self.pp_stages
            # Verify match exists to trigger 'search mode'
            if any(query_lower in p.__name__.lower() for s in all_stages for p in s.passes):
                search_query = query_lower
            else:
                warnings.warn(f"No pass found matching '{pass_name}'. Showing all stages.", UserWarning)

        html_output = jenv.get_template("compiler_stages.html.jinja2").render(compiler=self, search_query=search_query)

        # Return HTML object so it renders automatically in Notebooks
        return display(HTML(html_output))
