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
"""Convert quantum circuits into instruction schedules.

This is the core module of ``CPC``. It contains the functionality to define a compiler, whose job is to
convert quantum circuits and calibration data into configuration settings and instruction schedules that
can be executed by the IQM server on quantum hardware.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from exa.common.data.setting_node import SettingNode
from iqm.cpc.compiler.errors import CalibrationError, InsufficientContextError
from iqm.cpc.interface.compiler import (
    CircuitBoundaryMode,
    CircuitExecutionOptions,
    DDMode,
    HeraldingMode,
    MeasurementMode,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
)
from iqm.pulla.interface import CalibrationSet
from iqm.pulla.utils import (
    _update_channel_props_from_calibration,
    build_settings,
    calset_to_cal_data_tree,
    initialize_schedule_builder,
)
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.gate_implementation import GateImplementation, Locus
from iqm.pulse.gates import register_implementation

# from iqm.pulse.gates.move import apply_move_gate_phase_corrections, validate_move_instructions
from iqm.pulse.quantum_ops import QuantumOp

if TYPE_CHECKING:
    from exa.common.qcm_data.chip_topology import ChipTopology
    from iqm.pulse.playlist.channel import ChannelProperties


tracer = trace.get_tracer(__name__)
cpc_logger = logging.getLogger("cpc")

STANDARD_CIRCUIT_EXECUTION_OPTIONS_DICT = {
    "measurement_mode": MeasurementMode.ALL,
    "heralding_mode": HeraldingMode.NONE,
    "dd_mode": DDMode.DISABLED,
    "dd_strategy": None,
    "circuit_boundary_mode": CircuitBoundaryMode.ALL,
    "move_gate_frame_tracking": MoveGateFrameTrackingMode.FULL,
    "move_gate_validation": MoveGateValidationMode.STRICT,
    "active_reset_cycles": None,
}

STANDARD_CIRCUIT_EXECUTION_OPTIONS = CircuitExecutionOptions(**STANDARD_CIRCUIT_EXECUTION_OPTIONS_DICT)  # type: ignore


PassFunction = Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]]
"""A function that takes the data and context as arguments and returns the modified data and context.
The context is a dictionary that can contain any information that needs to be passed between the passes."""


def pass_function_idempotent(function: PassFunction) -> PassFunction:
    """Wrap a pass function to make it idempotent."""

    @functools.wraps(function)
    def pass_with_idempotency(data_: Any, context_: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        data = deepcopy(data_)
        context = deepcopy(context_)
        return function(data, context)

    return pass_with_idempotency


def compiler_pass(function) -> PassFunction:
    """Convenience wrapper to create a valid compiler pass.

    When the wrapped function is called, the compilation data (e.g. circuits) is passed as the first argument.
    If ``function`` has any other arguments, the wrapper takes their values from the ``context`` dict.
    If no matching key is found for a required argument, an error is raised.

    ``function` must return either a tuple of ``(data, ctx)`` where ``data`` is the
    compilation result and ``ctx`` is a dict with any new context data, or only ``data``.
    The contents of ``ctx`` will be merged to the input context.
    Note the difference to a plain, unwrapped CompilationPass: not returning ``ctx`` is valid.
    """
    sig = inspect.signature(function)
    if not sig.parameters:
        raise ValueError(f"Callable {function} wrapped with 'compiler_pass' should have at least one input argument.")
    required_keys = [key for key, param in sig.parameters.items() if param.default is inspect.Parameter.empty][1:]

    @functools.wraps(function)
    def pass_with_converted_args(data_: Any, context: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        kwargs = {}
        for required_key in required_keys:
            if required_key not in context:
                raise InsufficientContextError(f'Missing context data: "{required_key}".')
            kwargs[required_key] = context[required_key]
        result = function(data_, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            context = context | result[1]
            return result[0], context
        return result, context

    return pass_with_converted_args


class CompilationStage:
    """Sequence of compiler passes that are applied to the data.

    The data and context are returned after all passes have been applied.
    A pass is a function that takes the data and context as arguments and
    returns the modified data and context. The context is a dictionary that can contain any information that needs to be
    passed between the passes.
    """

    def __init__(self, name: str):
        self.name: str = name
        self.passes: list[Callable] = []

    def ready(self) -> bool:
        """Check if the stage is ready to run. A stage is ready if it has at least one pass defined."""
        return len(self.passes) > 0

    def add_passes(self, *pass_functions: PassFunction) -> None:
        """Add multiple passes to the stage.

        Args:
            pass_functions: One or more pass functions to be added to the stage.

        """
        for pass_function in pass_functions:
            self.passes.append(pass_function)

    def run(self, data: Any, context: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Run all the passes in the stage on the data and context. The data and context are returned after all passes have
        been applied.

        Args:
            data: The data to be processed.
            context: A dictionary containing any additional information that needs to be passed between the passes.

        Returns:
            The processed data and context.

        """  # noqa: E501
        for pass_function in self.passes:
            try:
                data, context = pass_function(data, context)
            except Exception as exc:
                raise RuntimeError(f'Error in stage "{self.name}" pass "{pass_function.__name__}": {exc}') from exc

        return data, context


class Compiler:
    """Stateful object that contains a calibration set, a schedule builder, and a set
    of compilation stages.

    The compiler's state does not include the data to be compiled.

    Args:
        calibration_set: Calibration data.
        chip_topology: Physical layout and connectivity of the quantum chip.
        channel_properties: Channel properties.
        component_channels: Mapping between components and their control channels.
        component_mapping: Custom mapping of components. Defaults to None.
        options: Circuit execution options.
            Defaults to STANDARD_CIRCUIT_EXECUTION_OPTIONS.
        stages: List of compilation stages. Defaults to None.
            Note that in the absence of stages, the compiler will not be ready to compile circuits.
        strict: If True, raises CalibrationError on calibration validation failures.
            If False, only logs warnings. Defaults to False.

    Raises:
        CalibrationError: When strict=True and calibration validation fails during compiler initialization.

    """

    def __init__(
        self,
        *,
        calibration_set: CalibrationSet,
        chip_topology: ChipTopology,
        channel_properties: dict[str, ChannelProperties],
        component_channels: dict[str, dict[str, str]],
        component_mapping: dict[str, str] | None = None,
        options: CircuitExecutionOptions = STANDARD_CIRCUIT_EXECUTION_OPTIONS,
        stages: list[CompilationStage] | None = None,
        strict: bool = False,  # consider extending to e.g. errors: Literal["raise", "warning", "ignore"] = "warning"
    ):
        self._calibration_set = calibration_set
        self.component_mapping = component_mapping
        self.options = options
        self.stages = stages or []

        self.builder: ScheduleBuilder = initialize_schedule_builder(
            calibration_set, chip_topology, channel_properties, component_channels
        )
        try:
            self.builder.validate_calibration()
        except ValueError as exc:
            if strict:
                raise CalibrationError(f"{exc}") from exc
            cpc_logger.warning("Calibration validation failed: %s", exc)

    def _refresh(self) -> None:
        """Refresh the compiler by re-creating the ScheduleBuilder and validating the calibration.
        Must be called automatically by any method that modifies the calibration set, or the op_table
        """
        _updated_channel_properties = _update_channel_props_from_calibration(
            self.builder.channels, self.builder.component_channels, self._calibration_set
        )
        self.builder = ScheduleBuilder(
            self.builder.op_table,
            calset_to_cal_data_tree(self._calibration_set),
            self.builder.chip_topology,
            _updated_channel_properties,
            self.builder.component_channels,
        )
        try:
            self.builder.validate_calibration()
        except ValueError as exc:
            raise CalibrationError(f"{exc}") from exc

    def get_calibration(self) -> CalibrationSet:
        """Returns a copy of the current local calibration set."""
        return deepcopy(self._calibration_set)

    def set_calibration(self, calibration: CalibrationSet) -> None:
        """Sets the current calibration set to a given calibration set, then refreshes the compiler.

        Args:
            calibration: The calibration set to be set as the current calibration set.

        """
        self._calibration_set = calibration
        self._refresh()

    @property
    def gates(self) -> dict[str, QuantumOp]:
        """Registered quantum gates."""
        return self.builder.op_table

    def set_default_implementation(self, gate_name: str, implementation_name: str) -> None:
        """Set the default implementation of a gate.

        Args:
            gate_name: Name of the gate.
            implementation_name: Name of the implementation to set as the default.

        """
        self.builder.op_table[gate_name].set_default_implementation(implementation_name)
        self._refresh()

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
            self.builder.op_table[gate_name].set_default_implementation_for_locus(implementation_name, locus)
        self._refresh()

    def amend_calibration_for_gate_implementation(
        self, gate_name: str, impl_name: str, locus: Locus, params: dict[str, Any]
    ) -> None:
        """Update the current local calibration set with calibration values for a specific gate/implementation/locus.

        The calibration values are given as a dictionary
        of parameter names and their values. This method refreshes the compiler after amending the calibration set.

        Args:
            gate_name: Name of the gate to which the calibration values are applied.
            impl_name: Name of the implementation of the gate to which the calibration values are applied.
            locus: Locus of the gate to which the calibration values are applied.
            params: Updated parameter names and their values.

        """
        if gate_name not in self.builder.op_table:
            raise ValueError(f"{gate_name} is not a registered gate.")
        if impl_name not in self.builder.op_table[gate_name].implementations:
            raise ValueError(f"{impl_name} is not a registered gate implementation of {gate_name}.")

        locus_str = "__".join(locus)
        for param, value in params.items():
            path = f"gates.{gate_name}.{impl_name}.{locus_str}.{param}"
            self._calibration_set[path] = value

        self._refresh()

    def add_implementation(
        self,
        op_name: str,
        impl_name: str,
        implementation: type[GateImplementation],
        *,
        set_as_default: bool = False,
        overwrite: bool = False,
        quantum_op_specs: QuantumOp | dict | None = None,
    ) -> None:
        """Adds a new implementation of a quantum operation (gate).

        Refreshes the compiler after adding a new implementation.

        Args:
            op_name: The gate name for which to register a new implementation.
            impl_name: The "human-readable" name with which the new gate implementation will be found e.g. in settings.
            implementation: The python class of the new gate implementation to be added.
            set_as_default: Whether to set the new implementation as the default implementation for the gate.
            overwrite: If True, allows replacing any existing implementation of the same name.
            quantum_op_specs: The quantum operation this gate represents. If a QuantumOp is given, it is used as is.
                If None is given and the same gate has been registered before, the previously registered properties are
                used.
                Otherwise, the given dict values are given to the constructor of
                :class:`~iqm.pulse.quantum_ops.QuantumOp`.
                For any missing constructor values, some defaults suitable for a 1-QB gate are used.

        """
        register_implementation(
            operations=self.builder.op_table,
            gate_name=op_name,
            impl_name=impl_name,
            impl_class=implementation,
            set_as_default=set_as_default,
            overwrite=overwrite,
            quantum_op_specs=quantum_op_specs,
        )
        self._refresh()

    def ready(self) -> bool:
        """Check if the compiler is ready to compile circuits. The compiler is ready if at least one stage is defined, and
        all the stages are non-empty.
        """  # noqa: E501
        if len(self.stages) == 0:
            return False
        for stage in self.stages:
            if not stage.ready():
                return False
        return True

    def print_all_implementations_trees(self) -> None:
        """Prints all implementations of all currently known quantum operations (gates), including parameters."""
        for op in self.builder.op_table.values():
            print(f"Operation: {op.name}")
            self.print_implementations_trees(op)
            print("-----------------------------------\n")

    def print_implementations_trees(self, op: QuantumOp) -> None:
        """Prints all implementation of a particular quantum operation (gate).

        Args:
            op: Quantum operation (gate) to print implementations of.

        """
        for impl_name, _ in op.implementations.items():
            self.builder.get_implementation_class(op.name, impl_name).get_parameters(
                [], path=[f"Operation: {op.name}, implementation: {impl_name}"]
            ).print_tree()

    def show_stages(self, full: bool = False) -> None:
        """Print the stages and passes defined in the compiler.

        Args:
            full: Iff True, also print the docstring of each pass function.

        """
        if len(self.stages) == 0:
            print("No stages defined.")
            return

        for index, stage in enumerate(self.stages):
            print(f"Stage {index}: {stage.name}")
            if not stage.ready():
                print("    No passes defined.")
            for pass_index, pass_fn in enumerate(stage.passes):
                print(f"    {pass_index}: {pass_fn.__name__}")
                if full and pass_fn.__doc__:
                    print(f"        {pass_fn.__doc__}")
            print()

    def compiler_context(self) -> dict[str, Any]:
        """Return initial compiler context dictionary. Used automatically by :meth:`Compiler.compile`."""
        return {
            "calibration_set": self._calibration_set,
            "builder": self.builder,
            "component_mapping": self.component_mapping,
            "options": self.options,
            "channel_properties": self.builder.channels,
            "chip_topology": self.builder.chip_topology,
        }

    def compile(
        self, data: Iterable[Any], context: dict[str, Any] | None = None
    ) -> tuple[Iterable[Any], dict[str, Any]]:
        """Run all compiler stages. Initial context will be derived from :meth:`Compiler.compiler_context` unless a custom
        context dictionary is provided.

        Args:
            data: An iterable of circuits to be compiled.
            context: Custom initial compiler context dictionary.

        """  # noqa: E501
        if not self.ready():
            raise RuntimeError("Compiler is not ready: no stages defined, or some stages have zero passes.")

        # Dictionary of context to be passed through all the passes
        if context is None:
            context = self.compiler_context()

        # Run the stages
        for stage in self.stages:
            cpc_logger.info('Running stage "%s"...', stage.name)
            data, context = stage.run(data, context)

        cpc_logger.info("Compilation finished.")
        return data, context

    def build_settings(self, context: dict[str, Any], shots: int) -> tuple[SettingNode, dict[str, Any]]:
        """Build the settings for the execution. Updates context["circuit_metrics"] with schedule_duration and
        min_execution_time.

        Args:
            context: A dictionary containing the necessary data for building the settings.
            shots: The number of shots to be executed.

        Returns:
            settings: A dictionary containing the settings for the execution.
            context: The updated context.

        """
        try:
            schedules = context["schedules"]
            builder = context["builder"]
            calibration_set = context["calibration_set"]
            circuit_metrics = context["circuit_metrics"]
            options = context["options"]
            custom_settings = context.get("custom_settings")
        except Exception as exc:
            raise InsufficientContextError(f"Missing context data for building settings: {exc}") from exc

        settings = build_settings(
            shots=shots,
            calibration_set=calibration_set,
            builder=builder,
            circuit_metrics=circuit_metrics,
            options=options,
        )
        # if custom_settings are given, use them to override similarly named generated settings
        if custom_settings is not None:
            settings = SettingNode.merge(custom_settings, settings)

        # fill in the schedule durations to the metrics
        end_delay = calibration_set["controllers.options.end_delay"]
        # Assumes all channels have the same sampling rate
        channel = next(iter(builder.channels.values()))

        for metrics, schedule in zip(circuit_metrics, schedules):
            metrics.schedule_duration = channel.duration_to_seconds(schedule.duration)
            # lower bound on the actual execution time: schedule duration + reset
            # does not include the heralding measurement
            metrics.min_execution_time = shots * (metrics.schedule_duration + end_delay)

        return settings, context
