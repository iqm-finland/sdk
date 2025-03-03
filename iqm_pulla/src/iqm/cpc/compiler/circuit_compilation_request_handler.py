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

This is the entry-point for integrations, e.g. for a server-side component performing circuit-to-pulse compilation.
"""

import logging
from typing import Any

from opentelemetry import context as telemetry_context
from opentelemetry import propagate

from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.cpc.compiler.compiler import Compiler, cpc_logger, tracer
from iqm.cpc.compiler.standard_stages import get_standard_stages
from iqm.cpc.interface.compiler import CircuitBatch, CircuitCompilationResult, CircuitExecutionOptions
from iqm.pulla.interface import CalibrationSet
from iqm.pulse.playlist.channel import ChannelProperties


def handle_circuit_compilation_request(  # noqa: PLR0913
    job_id: str,
    circuits: CircuitBatch,
    shots: int,
    calibration_set: CalibrationSet,
    chip_topology: ChipTopology,
    channel_properties: dict[str, ChannelProperties],
    component_channels: dict[str, dict[str, str]],
    options: CircuitExecutionOptions,
    custom_settings: SettingNode | None,
    qubit_mapping: dict[str, str] | None,
    trace_context: dict[str, Any] | None,
) -> CircuitCompilationResult:
    """Compile a batch of quantum circuits into a form that can be executed by Station Control.

    Args:
        job_id: ID of the job requesting compilation, used in logging
        circuits: quantum circuits to compile into schedules
        shots: number of times to repeat the execution of each circuit
        calibration_set: calibration data for the station the circuits are executed on
        chip_topology: topology of the QPU the circuits are executed on
        channel_properties: properties of control channels on the station
        component_channels: QPU component to function to channel mapping
        options: various discrete options for circuit execution that affect compilation
        custom_settings: additional Station Control settings to override generated settings
        qubit_mapping: Mapping of logical qubit names to physical qubit names. ``None`` means the identity mapping.
        trace_context: telemetry context to be propagated during circuit compilation

    Returns:
        circuit compilation result

    """
    if trace_context:
        telemetry_context.attach(propagate.extract(trace_context))
    cpc_logger.debug("Compiling circuits for job %s", job_id)

    with tracer.start_as_current_span("prepare_compilation"):
        if cpc_logger.isEnabledFor(logging.DEBUG):
            cpc_logger.debug(
                "calibration set contents:\n%s",
                "".join(f"  {k:60s}:  {v}\n" for k, v in calibration_set.items()),
            )
        compiler = Compiler(
            calibration_set=calibration_set,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=qubit_mapping,
            options=options,
            stages=get_standard_stages(idempotent=False),
            strict=True,
        )

    with tracer.start_as_current_span("compile_circuits", attributes={"len(circuits)": len(circuits)}):
        playlist, compilation_context = compiler.compile(circuits)

    with tracer.start_as_current_span("build_settings"):
        compilation_context["custom_settings"] = custom_settings
        settings, compilation_context = compiler.build_settings(compilation_context, shots)

    with tracer.start_as_current_span("prepare_compilation_results"):
        return CircuitCompilationResult(
            playlist=playlist,
            readout_mappings=compilation_context["readout_mappings"],
            settings=settings,
            circuit_metrics=compilation_context["circuit_metrics"],
        )
