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
"""IQM Experiment base class."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import signature
import logging
import time
from typing import Any, Literal, Self, final
import uuid

from IPython.display import HTML
from iqm.iqm_server_client.iqm_server_executor import IQMServerExecutor
import xarray

from exa.common.control import SweepValues
from exa.common.data import CollectionType, DataType, Parameter, Setting, SettingNode
from exa.common.errors import (
    ConflictError,
    EmptyComponentListError,
    InvalidOperationError,
    IQMError,
    NotFoundError,
    ValidationError,
)
from exa.common.qcm_data import ChipLayout, ChipTopology
from exa.common.qcm_data.chad_model import CHAD
from exa.common.sweep import NdSweep, Sweeps, convert_sweeps_to_list_of_tuples
from exa.core.analyze_result import AnalyzeResult, generic_plot_definition
from exa.core.experiment_configuration import (
    set_calibration_data_for_gate_implementation,
    set_gate_implementation_as_default,
)
from exa.core.experiment_definitions import (
    AnalyzeFunction,
    CircuitGenerationFunction,
    ComponentGrouping,
    ComponentGroupingMode,
    Components,
    FullTGF,
    ParallelGroup,
    QPUType,
    ReadoutMode,
    ReadoutProgrammingMode,
    SweepGenerationFunction,
)
from exa.core.experiment_options import ExperimentOptions
from exa.core.observation import (
    FromExperimentConfiguration,
    ObservationHandler,
    get_global_default_observation_loading_rules,
)
from exa.core.provider import ExaRuntime
from exa.core.run_result import RunResult
from exa.core.util.experiment_class import get_class_by_name
from iqm.cpc.compiler._utils.settings import (
    COMPILER_DEFAULTS_SOURCE,
    _validate_gate_definitions_against_settings,
    assign_default_gate_implementations_from,
)
from iqm.cpc.compiler.compiler import CompilationStage, Compiler
from iqm.cpc.compiler.post_process import _STANDARD_POST_PROCESSING_STAGES
from iqm.cpc.compiler.standard_stages import (
    _STANDARD_CIRCUIT_STAGES_EXPERIMENT,
    _STANDARD_FINAL_STAGES,
    _STANDARD_PULSE_STAGES_EXPERIMENT,
)
from iqm.cpc.core.config import ExperimentConfiguration, get_conventional_control_mapping
from iqm.cpc.core.parallel_utils import generate_full_tgf
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.gate_implementation import GateImplementation, Locus
from iqm.pulse.gates import register_implementation, register_operation
from iqm.pulse.gates.reset import RESET_FEEDBACK_KEY
from iqm.pulse.playlist.visualisation.base import inspect_playlist
from iqm.pulse.quantum_ops import QuantumOp
from iqm.pulse.timebox import TimeBox
from iqm.station_control.client.job_trackers import RunJobTracker, _run_data_to_timeline
from iqm.station_control.interface.executor_interface import ExecutorInterface
from iqm.station_control.interface.models import RunDefinition, SweepDefinition
from iqm.station_control.interface.models.jobs_new import JobData, JobStatus
from iqm.station_control.interface.storage_interface import StorageInterface

from .stages import add_terminal_measurement, post_process_target_variables
from .utils import (
    parallel_utils,
    setting_utils,
    sweep_utils,
)
from .utils._core import finalize_settings, get_sweep_parameters, merge_sweeps

logger = logging.getLogger(__name__)


def _empty_box_circuit(components: list[str], builder: ScheduleBuilder) -> list[TimeBox]:
    return [TimeBox.composite([])]


def _empty_sweeps(components: ComponentGrouping, chip_topology: ChipTopology, settings: SettingNode) -> Sweeps:
    return []


def _dummy_analyze(components: ComponentGrouping, run_result: RunResult, **kwargs) -> AnalyzeResult:
    plots = {param: generic_plot_definition(param) for param in run_result.run_definition.default_data_parameters}
    analyze_result = AnalyzeResult(run_result=run_result, plots=plots)
    return analyze_result


def _format_run_summary(
    qpu_runtime: float | None,
    run_time: float,
    create_time: float | None,
) -> list[str]:
    """Compose the run time summary for the :meth:`Experiment.submit_run` info log.

    Args:
        qpu_runtime: Predicted QPU run time (in seconds), or ``None`` if the
            prediction could not be computed.
        run_time:  Wall-clock time (in seconds) spent executing the job, postprocessing and fetching results.
        create_time: Wall-clock time (in seconds) spent in ``Experiment.create``, or ``None`` if not available.

    Returns:
        Runtime summary.

    """
    parts: list[str] = []
    # TODO SW-2885: subtract the time spent in the queue (not useful in profiling experiments)
    # SC does not provide direct queuing time, once EXA jobs have access to the timeline we
    # should sum up all the ended-started deltas to get the actual runtime and report everything
    # else as "queuing + overheads"
    queuing_time = None
    if queuing_time is not None:
        run_time -= queuing_time
        parts.append(f"queuing time = {queuing_time:.2f}s")

    if qpu_runtime is not None:
        pct = qpu_runtime / run_time * 100.0 if run_time > 0 else 0.0
        parts.append(f"runtime (quantum/total) = {qpu_runtime:.2f}s/{run_time:.2f}s ({pct:.1f} %)")
    else:
        parts.append(f"runtime (total) = {run_time:.2f}s")

    if create_time is not None:
        parts.append(f"create time = {create_time:.2f}s")

    return parts


class InputType(Enum):
    """Return type of the Experiment's Circuit Generation Function."""

    TIMEBOX = "TimeBox"
    """The Circuit Generation Function returns ``list[TimeBox]``, i.e. circuits in the time-ordered
    IQM intermediate representation."""
    CIRCUIT = "Circuit"
    """The Circuit Generation Function returns circuits that are not yet resolved into TimeBoxes. They
    The actual abstraction level of the circuits is otherwise arbitrary (it can even be above the circuit-level),
    but they will need to resolved into TimeBoxes and the fully time-resolved."""


@dataclass
class ExperimentDefinition:
    """Combines the :class:`.RunDefinition` produced via compilation with the Compiler context."""

    run_definition: RunDefinition
    """The RunDefinition produced via compilation."""
    context: dict[str, Any]
    """The Compiler context (containing arbitrary stateful effects accumulated during the compilation)."""

    def __getattr__(self, name: str) -> Any:
        """Get attributes of the member fields from the top level.

        In the case of name collisions, the attributes of ``run_definition`` take precedence.
        """
        if name in ["run_definition", "context"]:
            return super().__getattribute__(name)
        if hasattr(self.run_definition, name):
            return getattr(self.run_definition, name)
        if name in self.context:
            return self.context[name]
        raise AttributeError(name)


class Experiment:
    """Base class for multi-qubit experiments.

    This class is designed to run arbitrary control sequences on multiple qubits.
    Sequences can be provided either as time-ordered :class:`Timeboxes <.TimeBox>`,
    or in the circuit-level, represented as quantum operations (i.e. gates),
    or even above the gate-level if the necessary compilation stages to
    go from that abstraction to the circuit-level is included.

    :meth:`__init__` should not be overridden in the subclasses. Instead, one should override
    the non-final methods below and/or :attr:`default_options` in order to achieve the desired behaviour.

    Args:
        label: Human-readable label for this experiment.
        circuit_generation_function: The circuit generation function.
            Define when using Experiment without inheritance.
            The function should take at least 2 arguments: ``components`` and ``builder`` and return a list of
            TimeBoxes. Any other arguments are treated as extra circuit parameters.
            See :meth:`get_default_circuit_generation_function` for more details on the signature.
            By default, returns an empty TimeBox, so no pulses are played.

    """

    name = "experiment"
    """Unique identifier for the experiment class. Used to differentiate between different datasets in the database."""

    circuit_stages: list[CompilationStage] = deepcopy(_STANDARD_CIRCUIT_STAGES_EXPERIMENT)
    """Circuit-level (above TimeBox) sweepable compilation stages."""

    pulse_stages: list[CompilationStage] = deepcopy(_STANDARD_PULSE_STAGES_EXPERIMENT)
    """TimeBox and below sweepable compilation stages."""

    final_stages: list[CompilationStage] = deepcopy(_STANDARD_FINAL_STAGES)
    """The final stages for creating the SC RunDefinition."""

    pp_stages: list[CompilationStage] = deepcopy(_STANDARD_POST_PROCESSING_STAGES)
    """The return data post-processing stages."""

    default_circuit_generation_function: CircuitGenerationFunction = staticmethod(_empty_box_circuit)
    """The circuit generation function is a ``Callable`` that returns either a :class:`.TimeBox` or a ``list[TimeBox]``
    object. For more information see :meth:`get_default_circuit_generation_function`"""

    default_sweep_generation_function: SweepGenerationFunction = staticmethod(_empty_sweeps)
    """The sweep generation function is a ``Callable`` that returns the type ``Sweeps``.
    For more information see :meth:`get_default_sweep_generation_function`"""

    default_analyze_function: AnalyzeFunction = staticmethod(_dummy_analyze)
    """The sweep generation function is a ``Callable`` that returns an :class:`.AnalyzeResult` object. For more
    information see :meth:`get_default_analyze_function`"""

    default_options: ExperimentOptions | None = None
    """Default experiment options. Inheriting classes may override this attribute to suit their use case."""

    default_label = "Multi Qubit Experiment"
    """The default label for the experiment. Inheriting classes may override this attribute."""

    allowed_qpu_types: Iterable[QPUType] = [QPUType.CRYSTAL, QPUType.STAR]
    """Defines if the experiment supports only star or normal QPUs. By default all QPU types are supported.
    Inheriting classes can override this attribute."""

    allowed_component_grouping_modes: list[str] = [
        ComponentGroupingMode.LIST.value,
        ComponentGroupingMode.GROUP.value,
        ComponentGroupingMode.COLOUR_GROUP.value,
    ]
    """Only for backwards compatibility, does not actually do anything anymore."""  # TODO remove

    parallel_group: ParallelGroup | str = ParallelGroup.NONE
    """Unit of parallelization for the experiment. Determines the
    logic for processing and validation of the experiment's run components (see :meth:`process_run_components`).
    Certain commonly encountered parallel groups are defined in :class:`.ParallelGroup`, but a custom
    string may also be used and the :meth:`process_run_components` may be overridden to process these appropriately."""

    input_type: InputType = InputType.TIMEBOX
    """Input type of the Experiment class. See :class:`.InputType`."""

    @final
    def __init__(
        self,
        label: str | None = None,
        circuit_generation_function: CircuitGenerationFunction | None = None,
    ) -> None:
        xarray.set_options(keep_attrs=True)

        # constructor for the normal use case: run new experiments, get station properties from ExaRuntime.executor
        self._old_data = False
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.label = label or self.default_label
        self.options: ExperimentOptions = (
            self.default_options.model_copy(deep=True) if self.default_options else ExperimentOptions()
        )
        self.run_results: list[RunResult] = []

        # chip design record
        self._chip_design_record: dict | None = ExaRuntime.executor().get_chip_design_record()
        chip_topology = ChipTopology.from_chip_design_record(self._chip_design_record)

        # experiment configuration
        self.configuration = deepcopy(ExaRuntime.experiment_configuration())

        # always get the DUT label from the station (can't trust self.configuration since user may edit it)
        dut_label = ExaRuntime.executor().get_dut_label()
        if self.configuration.dut_label:
            if self.configuration.dut_label != dut_label:
                raise ConflictError(
                    f"DUT label in experiment configuration {self.configuration.dut_label} differs from"
                    f"the one from the station: {dut_label}"
                )
        else:
            self.configuration.dut_label = dut_label
        self._dut_label = dut_label

        try:
            self._chip_layout = ChipLayout.from_chip_design_record(self._chip_design_record)
        except (ValidationError, TypeError) as err:
            self.logger.warning(
                "Couldn't fetch chip layout for DUT label '%s' from the executor, "
                "initializing a dummy chip layout. Original error: %s",
                dut_label,
                err,
            )
            self._chip_layout = ChipLayout({}, {}, {})

        try:
            sw_set_id = ExaRuntime.get_or_create_software_version_set()
            station_control_settings = ExaRuntime.executor().get_settings()
        except (RuntimeError, IQMError) as err:
            self.logger.warning(
                "Couldn't fetch settings or SW version set id for DUT label '%s' from the station control, "
                "initializing Experiment with dummy properties: %s",
                dut_label,
                err,
            )
            sw_set_id = 0
            station_control_settings = SettingNode("root")

        pulse_stages = deepcopy(self.pulse_stages)
        pp_stages = deepcopy(self.pp_stages)
        construct_data_vars_stage = next((s for s in pp_stages if s.name == "construct_data_variables"), None)
        if construct_data_vars_stage is not None:
            construct_data_vars_stage.add_passes(post_process_target_variables)
        load_rules = get_global_default_observation_loading_rules() + [FromExperimentConfiguration(self.configuration)]
        self.compiler = Compiler(
            dut_label=self._dut_label,
            chip_topology=chip_topology,
            software_version_set_id=sw_set_id,
            station_control_settings=station_control_settings,
            observation_handler=ObservationHandler(
                [],
                dut_label=self._dut_label,
                default_storage=ExaRuntime.storage(),
                load_rules=load_rules,
            ),
            component_mapping=None,
            controller_mapping=self.configuration.components,
            gate_definitions=self.configuration.gate_definitions,
            circuit_stages=[] if self.input_type == InputType.TIMEBOX else deepcopy(self.circuit_stages),
            pulse_stages=pulse_stages,
            final_stages=deepcopy(self.final_stages),
            pp_stages=pp_stages,
        )
        self.compiler.stages.pulse_stages.timebox_stage.add_passes(add_terminal_measurement, index=0)
        self.circuit_generation_function = circuit_generation_function
        self._previous_context: dict[str, Any] = {}
        self.builder: ScheduleBuilder | None = None
        self.setup_additional_init_properties()

    @final
    @classmethod
    def from_dut_label(
        cls,
        *,
        dut_label: str,
        label: str,
    ) -> Self:
        """Construct an Experiment instance for loading and re-analyzing old run data.

        Gets the station properties from :attr:`.ExaRuntime.storage`.
        Cannot be used for executing new runs.

        Args:
            dut_label: Constructed Experiment instance will refer to this QPU/station instead of the the
                one you're connected to.
            label: Human-readable label for this experiment.

        Returns:
            Experiment instance for loading and re-analyzing old data.

        """
        xarray.set_options(keep_attrs=True)
        self = cls.__new__(cls)  #  We have no use case neutral __init__, so we can't call it here.
        self._old_data = True

        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.label = label
        self.options = self.default_options.model_copy(deep=True) if self.default_options else ExperimentOptions()
        self.run_results = []

        # get experiment configuration and chip design record
        try:
            self._chip_design_record = ExaRuntime.storage().get_chip_design_record(dut_label)
            chip_topology = ChipTopology.from_chip_design_record(self._chip_design_record)
        except IQMError as err:
            self.logger.warning(
                "Couldn't fetch chip design record for DUT label '%s' from the station control, "
                "initializing a dummy chip topology. Original error: %s",
                dut_label,
                err,
            )
            self._chip_design_record = None
            chip_topology = ChipTopology((), (), {}, {})

        # minimal default configuration for this QPU type
        self.configuration = ExperimentConfiguration(
            {
                "dut_label": dut_label,
                "components": get_conventional_control_mapping(chip_topology),
            }
        )
        self._dut_label = dut_label

        try:
            self._chip_layout = (
                ChipLayout({}, {}, {})
                if self._chip_design_record is None
                else ChipLayout.from_chip_design_record(self._chip_design_record)
            )
        except (ValidationError, TypeError) as err:
            self.logger.warning(
                "Couldn't fetch chip layout for DUT label '%s' from the executor, "
                "initializing a dummy chip layout. Original error: %s",
                dut_label,
                err,
            )
            self._chip_layout = ChipLayout({}, {}, {})

        # minimal Compiler that can only postprocess results
        pp_stages = deepcopy(self.pp_stages)
        construct_data_vars_stage = next((s for s in pp_stages if s.name == "construct_data_variables"), None)
        if construct_data_vars_stage is not None:
            construct_data_vars_stage.add_passes(post_process_target_variables)
        # Most of the Compiler parameters are not needed here since Experiment.analyze does not use them,
        # and RunJobTracker.result_exa only calls Compiler.post_process.
        self.compiler = Compiler(
            dut_label=self._dut_label,
            chip_topology=chip_topology,
            software_version_set_id=0,
            station_control_settings=SettingNode("root"),
            component_mapping=None,
            controller_mapping=None,
            gate_definitions=None,
            observation_handler=ObservationHandler([], dut_label=self._dut_label, default_storage=ExaRuntime.storage()),
            circuit_stages=None,
            pulse_stages=None,
            final_stages=None,
            pp_stages=pp_stages,
        )
        self.circuit_generation_function = None
        self._previous_context = {}
        self.builder = None
        self.setup_additional_init_properties()
        return self

    @property
    def executor(self) -> ExecutorInterface:
        """The instance implementing the :class:`ExecutorInterface` that this experiment will use when running.

        Returns:
            The active executor instance.

        """
        return ExaRuntime.executor()

    @property
    def storage(self) -> StorageInterface:
        """The instance implementing the :class:`StorageInterface` that this experiment will use when running.

        Returns:
            The active storage instance.

        """
        return ExaRuntime.storage()

    @property
    def chad(self) -> CHAD:
        """The CHAD data this experiment will use when running."""
        # TODO should be deprecated, use self.chip_topology instead
        if self._chip_design_record is None:
            raise NotFoundError("CHAD data not available.")
        return CHAD(**self._chip_design_record)

    @property
    def chip_topology(self) -> ChipTopology:
        return self.compiler.chip_topology

    @property
    def chip_layout(self) -> ChipLayout:
        return self._chip_layout

    @property
    def coupler_mapping(self) -> dict[str, list[str]]:
        # TODO should be deprecated, use self.chip_topology.coupler_to_components instead
        return {c: list(qs) for c, qs in self.chip_topology.coupler_to_components.items()}

    @property
    def observation_handler(self) -> ObservationHandler:
        return self.compiler.observation_handler

    @property
    def gate_definitions(self) -> dict[str, QuantumOp]:
        return self.compiler.gate_definitions

    @property
    def gates(self) -> dict[str, Any]:
        return self.configuration.gates

    @staticmethod
    def has_required_gates() -> bool:
        """Check whether the station provides all gates required by the circuit generation function.

        Subclasses should override this method to verify availability of any non-default gates they depend on.

        Note:
            Currently overridden only by experiment classes that require the MOVE operation.

        """
        # TODO Ideally, required gates would be inferred automatically from the circuit generation
        # function, removing the need to override this method in each experiment subclass.
        return True

    def get_default_circuit_generation_function(self) -> CircuitGenerationFunction:
        """Circuit generation function for this experiment.

        By default, the function stored (as a static method) in the attribute
        :attr:`default_circuit_generation_function` is returned, but in case more complex logic
        (including any number of ``self`` calls)
        is needed for determining the function, the default method can be fully overridden in inheriting classes.

        The circuit generation function (CGF) is a ``Callable`` that produces the control schedule
        the experiment wants to execute. There are two kinds of CGFs, which are identified by inspecting
        their signature:

        1. Subcircuit CGF:
           Has ``group: tuple[str, ...]`` and ``builder: ScheduleBuilder`` as its first two parameters.
           Returns a :class:`.TimeBox` that contains the control schedule for the parallelisable
           unit of QPU component names given in ``group``.
           :attr:`parallel_group` determines how the different groups of simultaneously parallelizable
           units (also called color groups since determining them is a graph coloring problem)
           are derived from the components given to :meth:`create`.
           All the subcircuit TimeBoxes corresponding to the same parallel group are automatically
           wrapped into the full circuit Timebox, which runs them in parallel.

        2. Full circuit CGF:
           Has ``components: ComponentGrouping`` and ``builder: ScheduleBuilder`` as its first two parameters.
           Returns a ``list[TimeBox]`` that represents the full circuit control schedule.

        In addition to the above required parameters, a CGF may have any number of additional args
        which may have default values. The default values will be added as Settings in the settings tree
        *unless* they are defined as default circuit settings (see :meth:`get_default_circuit_settings`).

        Note that ``couplers`` are not given as default arguments to the CGF. They are accessible via
        the ``builder`` argument (mostly implicitly, as the user does not have to usually adress the couplers;
        the builder will do that under the hood). If the user explicitly includes the couplers in the run components,
        they will be accessible in the ``components`` argument.

        The CGF gets the values for its extra parameters from the following sources in this order of precedence:

        1. Sweeps that are defined for a parameter with the same name as the argument. These can come from either
           :meth:`get_default_sweeps` or given by the user in the :meth:`run` argument ``sweeps``.

        2. Circuit settings values given by the user after if they have defined a parameter of the same name in
           :meth:`get_default_circuit_settings` (or by inserting a parameter manually to :attr:`circuit_settings`).
           These values can be inserted in several places: manually in the notebook/runtime, in :meth:`get_settings`
           by overriding :meth:`update_settings` or in the actual setting definitions in
           :meth:`get_default_circuit_settings` (as the default value of the Setting).

        3. The default values provided for the parameters in the definition of the CGF that
           do not have a circuit setting in :attr:`circuit_settings` (these will still be added to the settings tree
           but without any labels or units).

        Returns:
            Circuit generation function.

        """
        return self.default_circuit_generation_function

    def get_default_circuit_settings(self) -> list[Setting]:
        """Override to define default circuit parameters.

        Enables specifying default values and units for circuit generation function parameters. This method should
        return a list of Settings that will be added in the settings tree (including their units).
        Then if their name coincides with an argument of the circuit_generation_function, their value will be used
        to generate a circuit (you can add circuit settings which do not coincide with an argument of the circuit
        generation function).

        Returns:
            The default circuit settings. These will be added in the Settings tree alongside with their units.

        """
        return []

    def setup_additional_init_properties(self) -> None:
        """Override to e.g. change the observation loading rules or register new gate implementations
        for this experiment instance.

        Here you may add new components to ``self.observation_handler``, or new observation to be loaded for a given
        component and change the default observations rules associated with any component and observation. Note that
        any active global observation rules will override whatever defaults you add here (including the defaults added
        to new observations you yourself added in this method).

        You can also register here a specific gate implementation needed only in this experiment.
        """

    @final
    def _get_default_sweeps(self, components: ComponentGrouping, settings: SettingNode) -> NdSweep:
        """Default sweeps for the experiment.

        Calls :meth:`get_default_sweeps`, and converts them to NdSweep.

        Args:
            components: Qubits and couplers for which the sweeps are defined.
            settings: Full settings object for the experiment.

        Returns:
            Default sweeps as a full cartesian sweep description.

        """
        return convert_sweeps_to_list_of_tuples(self.get_default_sweeps(components, settings))

    def get_default_sweeps(self, components: ComponentGrouping, settings: SettingNode) -> Sweeps:
        """Default sweep definition for the experiment.

        Calls :meth:`get_default_sweep_generation_function`, and then calls the returned function
        to generate the sweeps, applying automatic parallelization logic for group-wise sweep
        generation functions.

        Sweeps defined as the input argument of the :meth:`run` method take precedence.
        The default sweeps are assumed to be done in the innermost loop.

        The default sweeps can be defined for any parameter in settings tree (including pulse
        parameters and circuit parameters), or for a parameter whose name coincides with a
        ``circuit_generation_function`` argument.

        .. hint::

           For a parallel Rabi experiment, this method could return
           ``Sweep(parameter=<parameter for the I amplitude of a PRX gate>, data=[...])``
           for all the qubits.

        .. note::

           Overriding this method is allowed for backwards compatibility, but should be done anymore.
           Define :attr:`default_sweep_generation_function` instead.

        Args:
            components: Qubits and couplers for which the sweeps are defined.
            settings: Full settings object for the experiment.

        Returns:
            Default sweeps.

        """
        # current method
        default_sweep_function = self.get_default_sweep_generation_function()
        sweep_function_sig = list(signature(default_sweep_function).parameters.keys())
        if sweep_function_sig[0:3] == ["components", "chip_topology", "settings"]:
            return default_sweep_function(components, self.chip_topology, settings)
        if sweep_function_sig[0:3] == ["group", "chip_topology", "settings"]:
            return parallel_utils.get_color_grouped_sweeps(
                components, self.chip_topology, settings, default_sweep_function
            )

        raise ValueError(
            "The sweep function should have arguments [components, chip_topology, settings]"
            "for a full sweep function, or the arguments [group, chip_topology, settings] for "
            f"a group sweep function. Your sweep function has the arguments {sweep_function_sig}."
        )

    # TODO should be @final, the experiments that redefine this just repeat the logic here.
    def get_default_sweep_generation_function(self) -> SweepGenerationFunction:
        """Return the sweep generation function.

        Returns the static method :attr:`default_sweep_generation_function`, with all its
        non-standard parameters bound to the values of similarly named attributes in :attr`options`,
        if they exist.

        :attr:`default_sweep_generation_function` can be either a full function or a group-wise function,
        acting on the defined parallelisable unit of the experiment (see :attr:`parallel_group`),
        in which case it is automatically wrapped into the full sweep function.
        In each case it has three standard positional parameters:

        * Group-wise functions have the standard parameters
          ``[group: tuple[str, ...], chip_topology: ChipTopology, settings: SettingNode]``, with
          ``group`` containing the QPU component names of the parallelisable group.
        * Full sweep functions have the standard parameters
          ``[components: ComponentGrouping, chip_topology: ChipTopology, settings: SettingNode]``,
          where ``components`` contains the full component grouping.

        .. note::

           If more complex logic (including any number of ``self`` calls) is needed for determining the
           function, this method can be fully overridden in inheriting classes, but this
           should be avoided.

        """
        sweep_function = self.default_sweep_generation_function
        non_standard_arg_names = list(signature(sweep_function).parameters.keys())[3:]
        kwargs = {n: getattr(self.options, n) for n in non_standard_arg_names if hasattr(self.options, n)}
        return partial(sweep_function, **kwargs)

    def update_settings(self, qubits: list[str], couplers: list[str], settings: SettingNode) -> None:
        """Additional updates for the settings.

        Inheriting classes should override this function instead of overriding the whole get_settings.
        Here, you can for example:

        - Add/remove operations or settings or setting nodes.
        - Override values in the settings tree (including any values for the circuit parameters).
        - Add more validation for options.

        What you should not try to do here:

        - Change :meth:`get_settings` behaviour related to `readout_mode`.

        Args:
            qubits: qubits for the settings.
            couplers: couplers for the settings.
            settings: the settings tree.

        """

    def process_run_components(
        self,
        components: ComponentGrouping,
        settings: SettingNode | None = None,
    ) -> tuple[ComponentGrouping, list[str] | dict[str, list[str]]]:
        """Process ``components`` and :attr:`.ExperimentOptions.subscribed_components` inputs given by the user
        into parallelly runnable component groups.

        The behaviour of this method is determined by :attr:`parallel_group` such that commonly encountered parallel
        groups are processed automatically. In addition, this method can be overridden in the inheriting classes for
        a specific behaviour needed in that class and not achieved by one of the default :class:`.ParallelGroup` cases.

        Note that this method should never return ``None`` or empty dict which are valid values for both run
        argument ``components`` and :attr:`options.subscribed_components`. These cases must be dealt in one way or
        another so that a non-empty :class:`.ComponentGrouping` and a non-empty
        ``subscribed_components`` are returned.

        Args:
            components: The components given as the run argument by the user.
            settings: Settings to set before sweeping. Defaults to `None` if settings are not needed to process run
                components.

        Returns:
            Parallelization for ``components``, subscribed_components.

        """
        if self.parallel_group == ParallelGroup.NONE:
            if not components:
                components = ComponentGrouping(list(self.chip_topology.qubits_sorted))
            return components, self.options.subscribed_components or components.flatten()
        if self.parallel_group == ParallelGroup.SINGLES:
            return parallel_utils.process_singular_run_components(components, self.chip_topology, self.options)
        if self.parallel_group == ParallelGroup.PROBE_LINES:
            return parallel_utils.process_probe_lines(components, self.chip_topology, self.options)
        if self.parallel_group == ParallelGroup.CONNECTED_PAIRS:
            return parallel_utils.process_components_to_pairs(components, self.chip_topology, self.options)
        if self.parallel_group == ParallelGroup.QUBIT_RESONATOR_PAIRS:
            return parallel_utils.process_components_to_pairs(
                components, self.chip_topology, self.options, allow_both=False
            )

        # TODO: Add star triplets coloring
        return parallel_utils.process_groups_to_colors(components, self.chip_topology, self.options)

    def update_compilation_context(
        self, context: dict[str, Any], components: ComponentGrouping, settings: SettingNode
    ) -> None:
        """Additional updates for the compilation context.

        Inheriting classes should override this function instead of subclassing
        :class:`~iqm.cpc.compiler.compiler.Compiler` just to override what is in the context.

        What should be done here:

        - Add additional context objects needed for compilation, eg. a Backend.

        What you should not try to do here:

        - Override any existing default context object, eg. components, builder.

        The context dictionary at input contains any additional keyword arguments added by the user when calling
        :meth:`create`, and by default it is empty. The modified dictionary is then used to update the Compiler's
        context.

        Args:
            context: Additional context provided by the user.
            components: Component groups given to the compiler.
            settings: Full settings tree to use as reference.

        """

    def make_final_mutations(self, run: RunDefinition) -> None:
        """Make any experiment-specific changes to the final run definition.

        The :class:`.RunDefinition` contains the complete information of the run, including both the request to be
        sent to Station Control, and experiment-specific details.

        This method can be overridden in inheriting experiments to make any kind of last-minute mutations to the
        definition before it is executed. The base implementation does nothing. The method will be called as the
        very last step of :meth:`create`.

        For example, this can be used to change metadata, such as the target data variables, which is needed by
        analysis.

        This method should be used only as the last resort if the desired logic cannot be expressed using the other
        methods provided by Experiment.

        Args:
            run: the RunDefinition instance to mutate.

        """

    def get_default_analyze_function(self) -> AnalyzeFunction:
        """Returns default analyze function.

        The analyze function can be either a full analyze function, in which case it should have the arguments
        ``[components, run_result, **kwargs]``, and it should perform the whole analysis for the experiment.

        You can also provide a group analyze function, which can have any combination of the following arguments,
        which will be passed in automatically:

        * ``group``: Mandatory. (:class:`.ComponentGrouping`) The parallel group for the iteration.
        * ``analyze_result``: (:class:`.AnalyzeResult`) Contains the dataset and all metadata.
           Can be mutated iteratively to generate analysis.
        * ``run_result``: (:class:`.RunResult`) Contains the dataset and all metadata.
        * ``chip_topology``: (:class:`.ChipTopology`) The chip topology data structure.
        * ``settings``: (:class:`.SettingNode`) The settings tree used in the run. Also contained in ``result``.
        * ``data_color``: (:class:`xarray.Dataset`) The data for the color group of this component group.
        * ``circuit_index``: (int) Index of the color group.

        A group analyze function can return either

        * a new AnalyzeResult object, the contents of which will be merged with results from other iterations, or
        * None. This is sufficient if ``result`` is passed as an argument and mutated by the function.

        The default behaviour of this method just returns the function stored in :attr:`.default_analyze_function`,
        but it can be overridden if more complex behaviour is needed.
        """
        return self.default_analyze_function

    def analyze(  # type: ignore[override]
        self,
        components: ComponentGrouping | None = None,
        run_index: int = -1,
        analyze_function: AnalyzeFunction | None = None,
        **kwargs,
    ) -> AnalyzeResult:
        """Analyzes an experiment run and returns the result.

        Based on the used analyze function (either the one returned by :meth:`.get_default_analyze_function` or
        the argument `analyze_function`), it will either just apply the function to the run result
        specified by ``run_index`` (full analyze function) or loop through the parallelly runnable partitions of the
        QPU (colours) and further the parallel groups within, building the full analyze result iteratively
        (group analyze function). The former and latter cases are automatically detected by inspection the signature
        of the provided analyze function.

        .. note::

            Overriding this method is allowed for backwards compatibility, but will be restricted in the future.

        Args:
            components: limit the analysis to these components.
            run_index: The run index.
            analyze_function: function for performing the analysis.
                See :meth:`.get_default_analyze_function` for more info.
            kwargs: Additional arguments for ``analyze_function``.

        Returns:
            The analyze result.

        """
        run_result = self.run_results[run_index]
        if not analyze_function:
            analyze_function = self.get_default_analyze_function()
        analyze_sig = list(signature(analyze_function).parameters.keys())
        chip_topology = deepcopy(self.chip_topology)
        chip_properties = self.chip_properties
        chip_layout = self.chip_layout
        settings = run_result.run_definition.sweep_definition.settings
        gate_definitions = _validate_gate_definitions_against_settings(deepcopy(self.gate_definitions), settings)
        if hasattr(settings, "gate_definitions"):
            gate_definitions = assign_default_gate_implementations_from(
                settings.gate_definitions, gate_definitions=gate_definitions, skip_missing=True
            )
        else:
            gate_definitions = {}
        if analyze_sig[0:2] == ["components", "run_result"]:
            auto_args = {
                "chip_topology": chip_topology,
                "chip_layout": chip_layout,
                "chip_properties": chip_properties,
                "gate_definitions": gate_definitions,
            }
            for not_wanted in set(auto_args.keys()) - set(analyze_sig):
                del auto_args[not_wanted]
            return analyze_function(components, run_result, **auto_args, **kwargs)  # type: ignore[return-value]
        if "group" not in analyze_sig:
            raise ValueError(
                "If the analyze function is for the whole dataset, it should have the arguments "
                "'components', 'run_result', '**kwargs' in the same order. "
                "Alternatively, if the analyze function analyzes a parallel group, the arguments should be some "
                "combination of: 'group' (mandatory), 'chip_topology', 'chip_layout', 'chip_properties', "
                "'gate_definitions', 'data_color', 'settings', 'circuit_index', 'run_result', 'analyze_result'. "
                f"The analyze function {analyze_function} has the arguments {analyze_sig}."
            )
        return parallel_utils.color_grouped_analysis(
            run_result=run_result,
            group_analyze_function=analyze_function,
            chip_topology=chip_topology,
            chip_layout=chip_layout,
            chip_properties=chip_properties,
            gate_definitions=gate_definitions,
            components=components,
            **kwargs,
        )

    # Anything below this line is _not_ meant to be overridden by any subclasses inheriting from Experiment.
    # If a feature is not supported by overriding the above functions (or the implementation would be too cumbersome),
    # please contact the EXA team for adding support for the feature.

    @final
    def get_settings(
        self,
        qubits: list[str] | None = None,
        couplers: list[str] | None = None,
        computational_resonators: list[str] | None = None,
    ) -> SettingNode:
        """Retrieves and modifies settings for Experiment.

        Args:
            qubits: The qubits to get the settings for. If None, all qubits found in the CHAD
                will be used instead.
            couplers: The couplers to get the settings for. If None, all couplers connecting the `qubits`
                defined above will be used.
            computational_resonators: Names of the computational resonators. If None, all resonators found in the CHAD
                will be used.

        Returns:
            The settings tree.

        """
        setting_utils.validate_options(self.options)
        self._validate_qpu_type()
        if qubits is None:
            qubits = list(self.chip_topology.qubits_sorted)
        if computational_resonators is None:
            computational_resonators = list(self.chip_topology.computational_resonators_sorted)
        if couplers is None:
            couplers = list(self.chip_topology.get_connecting_couplers(qubits + computational_resonators))
        kwargs: dict[str, Any] = {
            "qubits": qubits,
            "couplers": couplers,
            "computational_resonators": computational_resonators,
        }
        if self.input_type == InputType.TIMEBOX:
            kwargs["timeboxes"] = self._resolve_circuit_generation_function()
        else:
            kwargs["circuits"] = self._resolve_circuit_generation_function()
        settings = self.compiler.get_settings(**kwargs)
        self._apply_multi_qubit_options(settings)
        for setting in self.get_default_circuit_settings():
            settings.stages.circuit_generation.circuit[setting.name] = setting
        settings.stages.set_source(COMPILER_DEFAULTS_SOURCE, ignore_nones=True, ignore_populated=True)
        if self.options.exclude_operations_for:
            for operation, components in self.options.exclude_operations_for.items():
                if operation != "readout":  # excluding readout handled in circuit generation
                    settings = self.remove_operations(settings, components, [operation])
        self.update_settings(qubits, couplers, settings)
        return settings

    @final
    def _apply_multi_qubit_options(self, settings: SettingNode) -> None:  # noqa: PLR0912
        """Apply ``self.options`` to settings.

        Args:
            settings: The settings tree.

        """
        settings.controllers.options.playlist_repeats = self.options.repetitions
        settings.controllers.options.playlist_repeats._source = COMPILER_DEFAULTS_SOURCE
        if self.options.readout_mode == ReadoutMode.COUNTER.value:
            settings.stages["subscribed_groups"] = SettingNode("subscribed_groups")
        if isinstance(self.options.subscribed_components, dict):
            # TODO: Setting/Parameter needs dict support
            if self.options.readout_mode == ReadoutMode.COUNTER.value:
                for group, components in self.options.subscribed_components.items():
                    settings[f"stages.subscribed_groups.{group}"] = Setting(
                        Parameter(group, data_type=DataType.STRING, collection_type=CollectionType.LIST),
                        components,
                    )
            else:
                raise ValidationError("Grouped readout is only supported in the ReadoutMode.COUNTER.")
        elif self.options.subscribed_components:
            settings.stages.additionally_subscribed_components = self.options.subscribed_components

        # Single shot
        if self.options.readout_mode in [ReadoutMode.SINGLE_SHOT.value, ReadoutMode.SINGLE_SHOT_THRESHOLD.value]:
            settings.controllers.options.averaging_bins = self.options.repetitions
        else:
            settings.controllers.options.averaging_bins = 1
        settings.controllers.options.averaging_bins._source = COMPILER_DEFAULTS_SOURCE
        complex_modes = [ReadoutMode.COMPLEX.value, ReadoutMode.SINGLE_SHOT.value]
        acquisition_type = "complex" if self.options.readout_mode in complex_modes else "threshold"

        for measure_gate_name in [
            gate_name
            for gate_name in ["measure", "measure_fidelity", "measure_012"]
            if gate_name in settings.gate_definitions.subtrees
        ]:
            # for component in self.components:
            for impl_name in self.gate_definitions[measure_gate_name].implementations:
                for component in settings.gates[measure_gate_name][impl_name].children:
                    if "acquisition_type" in settings.gates[measure_gate_name][impl_name][component].settings:
                        settings.gates[measure_gate_name][impl_name][component].acquisition_type = acquisition_type
                        settings.gates[measure_gate_name][impl_name][
                            component
                        ].acquisition_type._source = COMPILER_DEFAULTS_SOURCE
        if "create_run_definition" in settings.stages.job_creation.subtrees:
            settings.stages.job_creation.create_run_definition.data_size_safety_switch = (
                self.options.data_size_safety_switch
            )
            settings.stages.job_creation.create_run_definition.force_ragged_data = self.options.force_ragged_data
        if "prepend_reset" in settings.stages.timebox_stage.subtrees:
            settings.stages.timebox_stage.prepend_reset.active_reset_cycles = self.options.active_reset_cycles
            if self.options.readout_programming_mode == ReadoutProgrammingMode.RESET_AT_END.value:
                settings.stages.timebox_stage.prepend_reset.active_reset_cycles = None
        if (
            "prepend_heralding" in settings.stages.timebox_stage.subtrees
            and self.options.readout_programming_mode == ReadoutProgrammingMode.HERALDING.value
        ):
            settings.stages.timebox_stage.prepend_heralding.add_heralding = True
            settings.stages.timebox_stage.prepend_heralding.measure_operation = self.options.used_measure_operations[
                "heralding"
            ]
        if "add_terminal_measurement" in settings.stages.timebox_stage.subtrees:
            settings.stages.timebox_stage.add_terminal_measurement.measure_operation = (
                self.options.used_measure_operations["terminal"]
            )
            if self.options.readout_programming_mode == ReadoutProgrammingMode.CUSTOM.value:
                settings.stages.timebox_stage.add_terminal_measurement.skip_terminal_measurement = True
            if self.options.readout_programming_mode == ReadoutProgrammingMode.RESET_AT_END:
                settings.stages.timebox_stage.add_terminal_measurement.feedback_key = RESET_FEEDBACK_KEY

    @final
    def create(  # noqa: PLR0915
        self,
        components: Components | None = None,
        default_sweep_ranges: sweep_utils.DefaultSweepOptions | None = None,
        sweeps: Sweeps | None = None,
        settings: SettingNode | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExperimentDefinition:
        """Create a complete :class:`.RunDefinition` for a new run.

        This method formulates the complete definition of the run from user inputs.
        It is used when an experiment is run, and defines the arguments accepted by :meth:`run`.
        `create` can be called manually to inspect the content of the RunDefinition.

        Args:
            components: The names of the "computational" components for the experiment. Typically, this means
                the qubits that are involved in the playlist generation, but in specific experiments this could be
                anything else as well (e.g. the couplers or some combination of the couplers and qubits).
                The components can be given as a flat list, a grouped list (list of tuples) or as an already
                partitioned list (i.e. "colour groups") of lists of tuples.
            default_sweep_ranges: New ranges for the default sweeps. These values replace the ``sweep.options`` defined
                in the sweeps in :meth:`get_default_sweeps`.
            sweeps: Additional user-defined sweeps.
            settings: All settings to set before sweeping.
                By default, uses the return value of :meth:`get_settings` when called without arguments.
            context: Optional custom Compiler context.

        Returns:
            A definition for a new run and the Compiler context.

        """
        create_t0 = time.monotonic()
        if settings:
            settings = settings.model_copy()

        settings = settings or self.get_settings()
        components, subscribed_components = self._validate_and_process_run_components(components, settings=settings)
        components = setting_utils.limit_components_by_settings(components, settings, self.chip_topology)
        finalize_settings(settings, self.options, self.chip_topology, subscribed_components)
        context = context or {}
        # get rid of the user convenience type for sweeps
        nd_sweeps: NdSweep = convert_sweeps_to_list_of_tuples(sweeps) if sweeps else []
        # Add default sweeps
        default_sweeps = self._get_default_sweeps(components, settings)
        if default_sweep_ranges:
            default_sweeps = sweep_utils.update_options_for_default_sweeps(default_sweeps, default_sweep_ranges)
        nd_sweeps = merge_sweeps(nd_sweeps, default_sweeps)

        self.update_compilation_context(context, components, settings)

        compile_kwargs: dict[str, Any] = {
            "components": components,
            "settings": settings,
            "sweeps": nd_sweeps,
            "context": context,
        }
        if self.input_type == InputType.TIMEBOX:
            compile_kwargs["timeboxes"] = self._resolve_circuit_generation_function()
        else:
            compile_kwargs["circuits"] = self._resolve_circuit_generation_function()
        run_definition, context = self.compiler.compile(**compile_kwargs)

        self._default_run_mutations(run_definition, components, settings)
        self.make_final_mutations(run_definition)
        # bw compatibility for storing builder in self
        self.builder = context["builder"]
        run_definition.additional_run_properties["create_time"] = time.monotonic() - create_t0
        return ExperimentDefinition(run_definition, context)

    @final
    def run(  # noqa: PLR0915
        self,
        components: Components | None = None,
        default_sweep_ranges: sweep_utils.DefaultSweepOptions | None = None,
        sweeps: Sweeps | None = None,
        settings: SettingNode | None = None,
        context: dict[str, Any] | None = None,
    ) -> Self:
        """Run the experiment to get results.

        The run result is appended to :attr:`run_results`.

        Args:
            components: The names of the "computational" components for the experiment. Typically, this means
                the qubits that are involved in the playlist generation, but in specific experiments this could be
                anything else as well (e.g. the couplers or some combination of the couplers and qubits).
                The components can be given as a flat list, a grouped list (list of tuples) or as an already
                partitioned list (i.e. "colour groups") of lists of tuples.
            default_sweep_ranges: New ranges for the default sweeps. These values replace the `sweep.options` defined
                in the sweeps in :meth:`get_default_sweeps`.
            sweeps: Additional user-defined sweeps.
            settings: All settings to set before sweeping.
                By default, uses the return value of :meth:`get_settings` when called without arguments.
            context: Optional custom Compiler context.

        Returns:
            self.

        """
        exp_definition = self.create(
            components=components,
            default_sweep_ranges=default_sweep_ranges,
            sweeps=sweeps,
            settings=settings,
            context=context,
        )
        self.submit_run(exp_definition)
        return self

    @final
    def submit_run(self, exp_definition: ExperimentDefinition) -> None:
        """Submit an experiment for execution on Station Control, wait for the results and post-process them.

        :attr:`.ExperimentDefinition.run_definition.additional_run_properties` is recorded in the database.

        .. note::

           :class:`Experiment` does not currently support stateful effects recorded in the Compiler
           context to be used when post-processing the return data. The reason is that Experiment
           runs need to be loadable from the DB such that the post-processing is applicable. The
           Compiler context is not generally serializable, so relying on data stored in it during
           the compilation (before execution) would break the loadability. Therefore, an empty
           context is used for the post-processing (the post-processing stages can still be stateful
           and utilize the context, but they cannot rely on information stored during the
           pre-execution compilation).

        Args:
            exp_definition: Experiment definition (the run definition and the compiler context).

        """
        if self._old_data:
            raise InvalidOperationError(
                f"This Experiment instance contains old run data for dut_label {self._dut_label}. "
                "If you wish to execute new runs, create a new instance."
            )

        run_definition = exp_definition.run_definition
        # lower bound estimate for the quantum runtime
        try:
            qpu_runtime_estimate: float | None = run_definition.sweep_definition.qpu_runtime
        except (NotFoundError, ValidationError):
            qpu_runtime_estimate = None

        # submit the job to executor
        executor = ExaRuntime.executor()
        run_t0 = time.monotonic()
        job_tracker = executor.submit_job(run_definition)
        # submit_run sets run_id and sweep_id
        self.logger.info(
            "%s started, job_id=%s, run_id=%s, sweep_id=%s%s",
            self.label,
            job_tracker.job_id,
            run_definition.run_id,
            run_definition.sweep_definition.sweep_id,
            "" if qpu_runtime_estimate is None else f", quantum runtime={qpu_runtime_estimate:.2f}s",
        )
        try:
            status = job_tracker.wait_for_completion()
        except KeyboardInterrupt:
            # cancel job on KeyboardInterrupt
            job_tracker.cancel()
            self.logger.info("Received KeyboardInterrupt, canceling job %s", job_tracker.job_id)
            return

        # job may be completed, failed, canceled or still in progress (due to hitting the wait timeout),
        # wait_for_completion logged a message explaining this
        if status != JobStatus.COMPLETED:
            raise NotFoundError(f"The job {job_tracker.job_id} did not complete successfully, status is {status}.")

        # HACK to accommodate StationControlClient-based execution (and MonolithExecutor-based integration tests):
        # Build a timeline using the timestamps found in RunData, obtained from StationControlStorage.
        # IQMServerExecutor provides the timeline automatically.
        if not isinstance(executor, IQMServerExecutor):
            run_data = ExaRuntime.storage().get_run(run_id=run_definition.run_id)
            job_tracker.job_data.timeline = _run_data_to_timeline(run_data)

        # fetch the results
        run_result = job_tracker.result_exa(compiler=self.compiler)
        if run_result is None:
            raise NotFoundError("Could not fetch the run results, should not happen.")
        self.run_results.append(run_result)

        # Wall-clock seconds spent executing the job, postprocessing and fetching results
        run_time = time.monotonic() - run_t0
        # log a summary of the run
        parts = [f"{self.label} finished"] + _format_run_summary(
            qpu_runtime_estimate,
            run_time,
            create_time
            if isinstance(create_time := run_definition.additional_run_properties.get("create_time"), (int, float))
            else None,
        )
        self.logger.info(", ".join(parts))

    @final
    def register_gate_implementation(
        self,
        gate_name: str,
        impl_name: str,
        impl_class: type[GateImplementation],
        set_as_default: bool = False,
        quantum_op: QuantumOp | None = None,
    ) -> None:
        """Register new gate implementation for this experiment instance.

        Args:
            gate_name: The gate name for which to register a new implementation.
            impl_name: The "human-readable" name with which the new gate implementation will be found e.g. in settings.
            impl_class: The class of the new gate implementation to be added.
            set_as_default: Whether to set the new implementation as the default implementation for the gate.
            quantum_op: The quantum operation ``impl_class`` implements, which is separately registered if given.
                If None, ``gate_name`` must already have been registered.

        """
        if quantum_op is not None:
            register_operation(self.gate_definitions, quantum_op)

        register_implementation(
            self.gate_definitions,
            gate_name,
            impl_name,
            impl_class,
            set_as_default=set_as_default,
        )
        # construct and cache the locus mapping for the gate implementation
        impl_class.get_locus_mapping(self.compiler.station_properties)

    @final
    def set_calibration_data_for_gate_implementation(
        self, gate_name: str, impl_name: str, calib_data: dict[Locus, dict[str, Any]]
    ) -> None:
        """Set calibration data for a gate implementation in this experiment instance.

        The result is as if this data was provided in the experiment config.

        Args:
            gate_name: The gate name for which to register a new implementation.
            impl_name: The "human-readable" name with which the new gate implementation will be found e.g. in settings.
            calib_data: Dict containing the calibration data. It should have the following structure:
                ``{<locus tuple1>: {<param1_name>: <param1_value>, <param2_name>: <param2_value>, ...}}``

        """
        set_calibration_data_for_gate_implementation(self.configuration.gates, gate_name, impl_name, calib_data)

    @final
    def set_gate_implementation_as_default(
        self,
        gate_name: str,
        impl_name: str,
    ) -> None:
        """Set this gate implementation as default for this experiment instance.

        Args:
            gate_name: The gate name for which to register a new implementation.
            impl_name: The "human-readable" name with which the new gate implementation will be found e.g. in settings.

        """
        set_gate_implementation_as_default(self.gate_definitions, gate_name, impl_name)

    @final
    def inspect_segment(
        self,
        components: Components | None = None,
        default_sweep_ranges: sweep_utils.DefaultSweepOptions | None = None,
        sweeps: Sweeps | None = None,
        settings: SettingNode | None = None,
        *,
        run_definition: RunDefinition | None = None,
        segments: Sequence[int] = (0,),
        render_output: bool = True,
    ) -> HTML | str:
        """Visualise the segments of a playlist.

        If ``run_definition`` is given, visualizes its playlist. Otherwise generates a
        run definition instance using the first four parameters.

        Args:
            components: The components for which to generate a run definition.
            default_sweep_ranges: New ranges for the default sweeps, which replace the ranges
                returned defined by :meth:`get_default_sweeps`.
            sweeps: Additional user-defined sweeps.
            settings: Settings to set before sweeping.
                By default, uses the return value of :meth:`get_settings` when called without arguments.
            run_definition: Run definition containing the playlist to inspect. If not given, uses
                :meth:`create` to create one based on the first four parameters.
            segments: Segment indices to inspect.
            render_output: Iff True return an HTML object that can be rendered to Jupyter notebook. Else returns HTML
                as a string.

        Return:
            HTML string or the string wrapped in to an HTML object.

        """
        if not run_definition:
            run_definition = self.create(components, default_sweep_ranges, sweeps, settings).run_definition
        playlist = run_definition.sweep_definition.playlist
        if not playlist:
            raise RuntimeError(f"No playlist defined for {run_definition}")
        html_text = inspect_playlist(playlist, segments)
        if render_output:
            return HTML(html_text)
        return html_text

    @property
    def last_run(self) -> RunResult:
        """Get the results from the latest run.

        Raises:
            ValueError: No runs have been executed

        Returns:
            Run result which contains the post-processed dataset for the most recent run.

        """
        return self.run_results[-1]

    @final
    def _default_run_mutations(
        self,
        run_definition: RunDefinition,
        components: ComponentGrouping,
        settings: SettingNode,
    ) -> None:
        """Default run result mutations."""
        run_definition.username = self.configuration.user
        run_definition.experiment_name = self.name
        run_definition.experiment_label = self.label
        run_definition.options = self.options.to_dict()
        run_definition.default_sweep_parameters = [p.name for p in self.default_sweep_parameters(components, settings)]

    @final
    def chip_properties(
        self,
        component_type: Literal["qubit", "tunable_coupler", "computational_resonator", "probe_line"],
        property_type: Literal["design", "fabrication"] = "design",
    ) -> dict[str, dict[str, Any]]:
        """Get a selected properties of chip components.

        The properties are extracted from station control's chip design record.
        If the record does not contain some information that matches the filters, no error is raised,
        the dictionary will simply be empty on that part.

        Args:
            component_type: Return properties only for these kinds of components.
            property_type: Return only these kinds of properties.

        Returns:
            A mapping from canonical component names to (undocumented) dictionaries of properties.

        """
        if self._chip_design_record is None:
            raise NotFoundError(
                "Chip design record not available. See the warnings emitted when creating the experiment instance."
            )
        properties = {}
        for component in self._chip_design_record["content"]["components"].get(component_type, []):
            properties[component["name"]] = component.get("properties", {}).get(property_type, {})
        return properties

    @final
    def default_sweep_parameters(
        self, components: ComponentGrouping, settings: SettingNode | None = None
    ) -> list[Parameter]:
        """Returns default sweep parameters.

        Args:
            components: Qubits and couplers for which the sweeps are defined.
            settings: All settings to set before sweeping.

        Returns:
            List of sweep parameters

        """
        settings_new = settings or self.get_settings()
        return get_sweep_parameters(
            self._get_default_sweeps(components, settings_new),
            only_first_in_tuple=True,
        )

    @final
    def _validate_and_process_run_components(
        self,
        components: Components | None,
        settings: SettingNode | None = None,
    ) -> tuple[ComponentGrouping, list[str] | dict[str, list[str]]]:
        """Validate and process the run arg ``components`` and ``self.options.subscribed_components``.

        Args:
            components: The components given as the run argument by the user.
            settings: Settings to set before sweeping. Defaults to `None` if settings are not needed to process run
                components.

        Returns:
            Validated and processed run components and subscribed components.

        Raises:
            ValueError: if the components are not of the correct form (either malformed or of otherwise valid form
                but not supported by the experiment) or if :meth:`process_run_components` returns ``None`` or empty
                lists/dicts or the returned ``subscribed_components`` is ``None``.

        """
        if components == []:
            raise EmptyComponentListError("Components are not allowed to be []. Please provide components or None.")

        components_grouping = ComponentGrouping(components)
        processed_components, subscribed_components = self.process_run_components(
            components_grouping, settings=settings
        )
        if not processed_components:
            raise ValueError(
                "The function `self.process_run_components` must not return None or empty components but a valid"
                "ComponentGrouping object."
            )
        if subscribed_components is None:
            raise ValueError("The function `self.process_run_components` must not return `None` subscribed components.")

        if isinstance(subscribed_components, dict):
            subscribed_components_copy = deepcopy(subscribed_components)
            for group_name, subscribed_group in subscribed_components.items():
                if len(subscribed_group) == 0:
                    del subscribed_components_copy[group_name]
                else:
                    subscribed_components_copy[group_name] = subscribed_group
            return processed_components, subscribed_components_copy
        return processed_components, subscribed_components

    @final
    def _resolve_circuit_generation_function(self) -> FullTGF:
        """Return the user given circuit function or the default function.

        The user given function can be either the sub-circuit function or the full circuit function which is detected
        automatically from the signature. In case of a sub-circuit, it will be wrapped to create the full function.
        """
        # we handle the typing here using runtime checks, hence the type: ignores
        circuit_generation_function = self.circuit_generation_function or self.get_default_circuit_generation_function()
        sig = list(signature(circuit_generation_function).parameters.keys())
        if sig[0] == "components":
            return circuit_generation_function  # type: ignore[return-value]
        if sig[0] == "group":
            if self.input_type == InputType.CIRCUIT:
                raise ValueError("Circuit-level Experiments do not yet support automatic circuit parallelization.")
            return generate_full_tgf(circuit_generation_function)  # type: ignore[arg-type]
        # TODO should raise an error if sig[0] is none of the above
        return circuit_generation_function  # type: ignore[return-value]

    @final
    def _validate_qpu_type(self) -> None:
        """Validate qpu type."""
        comp_resonators = self.chip_topology.computational_resonators
        if len(comp_resonators) > 0 and QPUType.STAR not in self.allowed_qpu_types:
            raise ValueError("This experiment can only be run on a QPU that has no computational resonators")
        if len(comp_resonators) == 0 and QPUType.CRYSTAL not in self.allowed_qpu_types:
            raise ValueError("This experiment can only be run on an star QPU.")

    @final
    def print_default_sweeps(
        self, components: ComponentGrouping | None = None, settings: SettingNode | None = None
    ) -> None:
        """Prints the default sweeps from :meth:`get_default_sweeps` in a human-readable format.

        Args:
            components: the run components.
            settings: the settings tree.

        """
        _, sweep_spec = self._get_default_sweep_spec(components, settings)
        print(sweep_spec)

    @final
    def get_default_sweeps_as_dict(
        self, components: ComponentGrouping | None = None, settings: SettingNode | None = None
    ) -> dict[str, SweepValues]:
        """Returns the default sweeps from :meth:`get_default_sweeps` as a dict that can be fed to :meth:`run`.

        The sweeps are returned in a flat dict with the parameter names as keys and the sweep data as values.

        Args:
            components: the run components.
            settings: the settings tree.

        Returns:
            The default sweeps as a dict.

        """
        sweep_spec, _ = self._get_default_sweep_spec(components, settings)
        return sweep_spec

    @final
    def _get_default_sweep_spec(
        self, components: ComponentGrouping | None = None, settings: SettingNode | None = None
    ) -> tuple[dict[str, SweepValues], str]:
        """Get default sweep specs as a string representation and a dict."""
        components, _ = self._validate_and_process_run_components(components, settings=settings)
        components = setting_utils.limit_components_by_settings(components, settings, self.chip_topology)
        settings = settings or self.get_settings()
        parallel_sweeps = self._get_default_sweeps(components, settings)
        sweep_specs = []
        sweep_specs_dict = {}
        for index, parallel_sweep in enumerate(parallel_sweeps):
            sweep_spec = f"{index + 1}. "
            indentation = ""
            if len(parallel_sweep) > 1:
                sweep_spec += "parallel sweep:\n"
                indentation = "    "
            for sweep in parallel_sweep:
                sweep_spec += f"{indentation}{sweep.parameter.name}: {sweep.data}\n"
                sweep_specs_dict[sweep.parameter.name] = sweep.data
            sweep_specs.append(sweep_spec)
        return sweep_specs_dict, "\n".join(sweep_specs)

    @final
    def _assign_default_gate_implementations_from(
        self,
        pulse_settings: SettingNode,
        gate_definitions: dict[str, QuantumOp] | None = None,
        skip_missing: bool = False,
    ) -> dict[str, QuantumOp]:
        """Added for backwards compatibility."""
        gate_definitions = gate_definitions or self.gate_definitions
        return assign_default_gate_implementations_from(pulse_settings, gate_definitions, skip_missing=skip_missing)

    @final
    @staticmethod
    def _get_experiment_with_result(
        experiment_name: str,
        experiment_label: str,
        dut_label: str,
        job_tracker: RunJobTracker,
        error_msg: str,
    ) -> Experiment:
        """An instance of the named Experiment class, pre-filled with the postprocessed results of an old job."""
        if experiment_class := get_class_by_name(Experiment, experiment_name):
            experiment = experiment_class.from_dut_label(
                dut_label=dut_label,
                label=experiment_label,
            )
            # FIXME: stateful post process does not work (if it requires compilation objects in the context)
            compiler: Compiler | None = experiment.compiler
        else:
            logger.error(
                "Cannot find an experiment class with name '%s'. "
                "Returning standard post processed data with the base class %s, without experiment-specific analysis.",
                experiment_name,
                Experiment.__qualname__,
            )

            try:
                experiment = Experiment.from_dut_label(
                    dut_label=dut_label,
                    label=experiment_label,
                )
                compiler = experiment.compiler
            except Exception:
                logger.exception(
                    "Post-processing failed. A stub class instance will be used to convey raw result data.",
                )
                experiment = ErrorExperiment(
                    label=experiment_label,
                    error_details=f"The experiment class with name {experiment_name} was not found.",
                )
                compiler = None

        run_result = job_tracker.result_exa(compiler=compiler)
        if run_result is None:
            raise NotFoundError(error_msg)
        experiment.run_results.append(run_result)
        return experiment

    @final
    @staticmethod
    def get_by_run_id(run_id: uuid.UUID) -> Experiment:
        """Get an instance of the Experiment that produced ``run_id``, holding its run result.

        This method is for accessing legacy data. To access runs sumbitted via IQM Server, use :meth:`get_by_job_id`.

        If the ``name`` of the experiment in the database does not match any currently imported Experiment class,
        will try to load and post-process the data using the base class :class:`.Experiment`.
        In this case, the experiment-specific analysis is not available.
        If the post-processing also fails, a stub Experiment containing only the raw data is returned.

        .. note::

           The old result is loaded from an EXA database, so you must initialize EXA with a connection to the
           database that holds your run. In most on-premises installations there is just one shared
           database that is automatically accessed through
           :envvar:`STATION_CONTROL_SERVICE_URL` or :envvar:`IQM_SERVER_URL`.

        .. warning::

           The returned Experiment instance should only be used to view and analyze old run data,
           not execute new runs, since it refers to the station that produced the old run, and
           not necessarily the station you're connected to. If you want to execute new runs, create another
           instance of the correct Experiment subclass and use it instead.

        Args:
            run_id: ID of a run that exists in the database.

        Returns:
             An instance of the :class:`.Experiment` subclass that produced ``run_id``.
             The dataset is accessible through the :attr:`last_run` attribute.

        """
        executor = ExaRuntime.executor()
        storage = ExaRuntime.storage()
        run_data = storage.get_run(run_id=run_id)

        # Re-construct the payload
        run_definition = RunDefinition(
            run_id=run_data.run_id,
            username=run_data.username,
            experiment_name=run_data.experiment_name,
            experiment_label=run_data.experiment_label,
            options=run_data.options.copy(),
            software_version_set_id=run_data.software_version_set_id,
            additional_run_properties=run_data.additional_run_properties.copy(),
            hard_sweeps=run_data.hard_sweeps.copy(),
            components=list(run_data.components),
            default_data_parameters=list(run_data.default_data_parameters),
            default_sweep_parameters=list(run_data.default_sweep_parameters),
            sweep_definition=SweepDefinition(
                sweep_id=run_data.sweep_data.sweep_id,
                dut_label=run_data.sweep_data.dut_label,
                settings=run_data.sweep_data.settings,
                sweeps=run_data.sweep_data.sweeps,
                return_parameters=run_data.sweep_data.return_parameters,
                playlist=None,  # RunData does not contain the playlist
            ),
        )
        # We can't access the real job with run_id only, since run_id != job_id.
        # TODO (Marko): Must be fixed, but for now we can only assume "job_id" == "sweep_id" in this case.
        # StationControlExecutor specifically queries run/sweep results using "sweep_id" from the server,
        # but parameter name in the interface must be "job_id" (required by IQM Server implementation).
        sweep_id = run_data.sweep_data.sweep_id
        # Get the sweep results (raw data) using a semi-fake RunJobTracker,
        run_job_tracker = RunJobTracker(
            job_data=JobData(
                id=sweep_id,
                type="run",
                status=JobStatus.COMPLETED,  # assume run is already completed
                timeline=_run_data_to_timeline(run_data),
            ),
            _executor=executor,
            _payload=run_definition,
        )
        # Construct the Experiment, retrieve and postprocess run results
        experiment = Experiment._get_experiment_with_result(
            run_data.experiment_name,
            run_data.experiment_label,
            run_data.sweep_data.dut_label,
            run_job_tracker,
            f"Run {run_id} (sweep {sweep_id}) results are not available, should not happen!",
        )
        return experiment

    @final
    @staticmethod
    def get_by_job_id(job_id: uuid.UUID) -> Experiment:
        """Get an instance of the Experiment that produced ``job_id``, holding its run result.

        If the ``name`` of the experiment in the database does not match any currently imported Experiment class,
        will try to load and post-process the data using the base class :class:`.Experiment`.
        In this case, the experiment-specific analysis is not available.
        If the post processing also fails, a stub Experiment containing only the raw data is returned.

        .. note::

           The old result is loaded from an EXA database, so you must initialize EXA with a connection to the
           database that holds your run. In most on-premises installations there is just one shared
           database that is automatically accessed through :envvar:`IQM_SERVER_URL`.

        .. warning::

           The returned Experiment instance should only be used to view and analyze old run data,
           not execute new runs, since it refers to the station that executed the old run, and
           not necessarily the station you're connected to. If you want to execute new runs, create another
           instance of the correct Experiment subclass and use it instead.

        Args:
            job_id: ID of a run job that exists in the database.

        Returns:
             An instance of the :class:`.Experiment` that produced ``job_id``.
             The dataset is accessible through the :attr:`last_run` attribute.

        """
        executor = ExaRuntime.executor()
        job_data = executor.get_job(job_id)
        if job_data.type != "run":
            raise ValueError("Job {job_id} is not an experiment run, its type is {job_data.type}.")
        if job_data.status != JobStatus.COMPLETED:
            raise ValueError("Job {job_id} is not completed, its status is {job_data.status}.")

        # Get job payload
        run_job_tracker = RunJobTracker(job_data=job_data, _executor=executor)
        run_definition = run_job_tracker.payload()
        # Construct the Experiment, retrieve and postprocess run results
        experiment = Experiment._get_experiment_with_result(
            run_definition.experiment_name,
            run_definition.experiment_label,
            run_definition.sweep_definition.dut_label,
            run_job_tracker,
            f"Job {job_id} results are not available, should not happen!",
        )
        return experiment

    @final
    @staticmethod
    def remove_operations(
        settings: SettingNode, component_or_group: str | Sequence[str], operations: list[str]
    ) -> SettingNode:
        """Removes the given operations from the settings tree under the given component or group.

        Note:
            The removed operations are not validated in any way. It is possible that the experiment produces
            unwanted results or does not even work properly when run with the resulting settings.

        Args:
            settings: Setting tree that one wants to modify.
            component_or_group: Single component, or component group, or a list of components or component groups
                from which to remove the operations.
            operations: Operations to remove.

        Returns:
            A new settings tree with ``operations`` removed from ``component_or_group``.

        """
        new_settings = settings.model_copy(deep=True)
        comp_or_group_list = [component_or_group] if isinstance(component_or_group, str) else component_or_group
        for comp_or_group in comp_or_group_list:
            for operation in operations:
                if operation not in new_settings.controllers[comp_or_group].children:
                    raise ValueError(
                        f'Could not remove operation "{operation}" from "{comp_or_group}" as it does not exist there'
                    )
                del new_settings.controllers[comp_or_group][operation]
        return new_settings


def get_experiment(run_id: uuid.UUID) -> Experiment:
    """Get a new instance of the class:`.Experiment` that produced ``run_id``."""
    return Experiment.get_by_run_id(run_id)


def error_analyze(experiment: Experiment, run_index: int = -1) -> AnalyzeResult:
    """Fallback for analyzing the result of an experiment run.

    Called when no specific analysis function is available for the experiment results,
    provides some generic plots and the raw experiment results.

    Args:
        experiment: an experiment instance.
        run_index: the run index of the experiment.

    Returns:
          An :class:`.AnalyzeResult` instance, containing the dataset, plots, observations, and more.

    """
    run_result = experiment.run_results[run_index]
    plots = {param: generic_plot_definition(param) for param in run_result.run_definition.default_data_parameters}
    analyze_result = AnalyzeResult(run_result=run_result, plots=plots)
    return analyze_result


def _throw_error_circuit(components: ComponentGrouping, builder: ScheduleBuilder) -> list[TimeBox]:
    raise NotImplementedError("ErrorExperiment is only used for loading data.")


class ErrorExperiment(Experiment):
    """Minimal implementation of Experiment for conveying sweep results and metadata to analyze results in case there is
    an error using the correct class.

    Args:
        label: Experiment label used as pretty identifier for display purposes.
        error_details: Error details to append to the :class:`.AnalyzeResult` returned from this class'
            :meth:`.analyze`.

    """

    name = "error_experiment"

    def __init__(self, label: str = "Error experiment label", error_details: str = ""):  # type: ignore[misc]
        Experiment.__init__(self, label, circuit_generation_function=_throw_error_circuit)
        self.error_details = error_details

    def analyze(self, run_index: int = -1, **kwargs) -> AnalyzeResult:  # type: ignore[override]
        analyze_result = error_analyze(self, run_index=run_index)
        analyze_result.error = (
            "A stub class instance was used to convey sweep results and metadata to analyze results with the base "
            "class method `Experiment.analyze()`. Only result data and default plots are available for this "
            "experiment, without experiment-specific post-processing. "
            f"Error details: {self.error_details}"
        )
        return analyze_result
