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
"""Utilities for working with Qiskit objects."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from copy import deepcopy
from datetime import date
from typing import TYPE_CHECKING, Any

from iqm.iqm_client import ExistingMoveHandlingOptions
from iqm.iqm_server_client.models import JobStatus
from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import QuantumCircuit
from qiskit.result import Counts, Result

from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.cpc.compiler.compiler import (
    CompilationStage,
    Compiler,
    CompilerOptions,
)
from iqm.cpc.compiler.post_process import (
    _STANDARD_CIRCUIT_POST_PROCESSING_STAGES,
    _STANDARD_POST_PROCESSING_STAGES,
)
from iqm.cpc.compiler.standard_stages import (
    _STANDARD_CIRCUIT_STAGES,
    _STANDARD_FINAL_STAGES,
    _STANDARD_PULSE_STAGES,
)
from iqm.cpc.core.config import ComponentGrouping, ComponentGroupingMode
from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash, RuleType
from iqm.cpc.interface.circuit_execution import Circuit
from iqm.pulla.interface import HERALDING_KEY
from iqm.pulse.quantum_ops import QuantumOp

if TYPE_CHECKING:
    from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMBackendBase

    from iqm.cpc.compiler.compiler import CompilationStage, Compiler
    from iqm.pulla.pulla import Pulla, PullaJob


def qiskit_circuits_to_pulla(
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
    qubit_idx_to_name: dict[int, str],
    custom_gates: Collection[str] = (),
) -> list[Circuit]:
    """Convert Qiskit quantum circuits into IQM Pulse quantum circuits.

    Args:
        qiskit_circuits: One or many Qiskit quantum circuits to convert.
        qubit_idx_to_name: Mapping from Qiskit qubit indices to the names of the corresponding
            qubit names.
        custom_gates: Names of custom gates that should be treated as additional native gates
            by qiskit-iqm, i.e. they should be passed as-is to Pulla.

    Returns:
        Equivalent IQM Pulse circuit(s).

    """
    if isinstance(qiskit_circuits, QuantumCircuit):
        qiskit_circuits = [qiskit_circuits]

    return [
        Circuit(
            name=qiskit_circuit.name,
            instructions=tuple(
                serialize_instructions(
                    qiskit_circuit,
                    qubit_idx_to_name,
                    custom_gates,
                ),
            ),
        )
        for qiskit_circuit in qiskit_circuits
    ]


def qiskit_to_pulla(
    pulla: Pulla,
    backend: IQMBackend,
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
) -> tuple[list[Circuit], Compiler]:
    """Convert transpiled Qiskit quantum circuits to IQM Pulse quantum circuits.

    Also provides the Compiler object for compiling them, with the correct
    calibration set and component mapping initialized.

    Args:
        pulla: Quantum computer pulse level access object.
        backend: qiskit-iqm backend used to transpile the circuits. Determines
            the calibration set to be used by the returned compiler.
        qiskit_circuits: One or many transpiled Qiskit QuantumCircuits to convert.

    Returns:
        Equivalent IQM Pulse circuit(s), compiler for compiling them.

    """
    # TODO backend is connected to Cocos, which must be connected to the same Station Control as pulla.
    # The pieces here still don't fit perfectly together.

    # build a qiskit-iqm RunRequest, then prepare to compile and execute it using Pulla
    run_request = backend.create_run_request(qiskit_circuits, shots=1)
    if run_request.calibration_set_id is None:
        raise ValueError("RunRequest created by IQMBackend has no calibration set id.")

    # create a compiler containing all the required station information
    compiler = pulla.get_standard_compiler(exa_style_pp=False)
    compiler.component_mapping = run_request.qubit_mapping
    # We can be certain run_request contains only Circuit objects, because we created it
    # right in this method with qiskit.QuantumCircuit objects
    circuits: list[Circuit] = [c for c in run_request.circuits if isinstance(c, Circuit)]
    return circuits, compiler


class QiskitCompiler(Compiler):
    """Pulla Compiler which contains the Qiskit backend (:class:`.IQMBackendBase`) and extra circuit stages for
    parallelizing and transpiling Qiskit circuits and finally converting them to IQM circuits.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dut_label: str,
        loading_rules: list[RuleType],
        chip_topology: ChipTopology,
        software_version_set_id: int,
        station_control_settings: SettingNode,
        component_mapping: dict[str, str] | None,
        controller_mapping: dict[str, dict[str, str]] | None = None,
        gate_definitions: dict[str, QuantumOp] | None = None,
        circuit_stages: list[CompilationStage] | None = None,
        pulse_stages: list[CompilationStage] | None = None,
        final_stages: list[CompilationStage] | None = None,
        pp_stages: list[CompilationStage] | None = None,
        compiler_options: CompilerOptions | None = None,
        name: str = "IQM Compiler",
        backend: IQMBackendBase | None = None,
    ) -> None:
        super().__init__(
            dut_label=dut_label,
            loading_rules=loading_rules,
            chip_topology=chip_topology,
            software_version_set_id=software_version_set_id,
            station_control_settings=station_control_settings,
            component_mapping=component_mapping,
            controller_mapping=controller_mapping,
            gate_definitions=gate_definitions,
            circuit_stages=circuit_stages,
            pulse_stages=pulse_stages,
            final_stages=final_stages,
            pp_stages=pp_stages,
            compiler_options=compiler_options,
            name=name,
        )
        self.backend = backend

    def compiler_context(self, components: ComponentGrouping | None, settings: SettingNode, **kwargs) -> dict[str, Any]:
        """Adds the Qiskit backend to the Compiler context."""
        context = super().compiler_context(components, settings, **kwargs)
        context["backend"] = self.backend
        return context


def get_qiskit_compiler(
    pulla: Pulla,
    backend: IQMBackendBase,
    loading_rules: list[RuleType] | None = None,
    *,
    exa_style_pp: bool = True,
    controller_mapping: dict[str, dict[str, str]] | None = None,
    gate_definitions: dict[str, QuantumOp] | None = None,
    options: CompilerOptions | None = None,
) -> QiskitCompiler:
    """Get the Qiskit-specific Pulla Compiler.

    Args:
        pulla: Pulla instance
        backend: Qiskit backend.
        loading_rules: Observation loading rules. If ``None``, will use the current default calibration set.
        exa_style_pp: Whether to do EXA-style dataset post-processing by default.
        controller_mapping: Dictionary that maps physical QPU component names to their device controller names.
            The dictionary is of the form: ``{<component_name>: {<operation_name>: <controller name>}}``,
            where operation is one of the following: "drive", "readout", "flux"
            (not all components have all operations supported).
        gate_definitions: Names of quantum operations mapped to their definitions, see :class:`.QuantumOp`.
        options: General options to define the compiler behaviour.

    Returns:
        The Qiskit-specific Pulla Compiler.

    """
    pp_stages = (
        deepcopy(_STANDARD_POST_PROCESSING_STAGES)
        if exa_style_pp
        else deepcopy(_STANDARD_CIRCUIT_POST_PROCESSING_STAGES)
    )
    loading_rules = loading_rules if loading_rules is not None else [LatestFromStash(pulla.get_calibration_stash())]
    return QiskitCompiler(
        dut_label=pulla.get_chip_label(),
        loading_rules=loading_rules,
        chip_topology=pulla._chip_topology,
        software_version_set_id=pulla._software_version_set_id,
        station_control_settings=pulla._station_control_settings.model_copy(),
        component_mapping=None,
        controller_mapping=controller_mapping,
        gate_definitions=gate_definitions,
        circuit_stages=[qiskit_transpilation_stage, qiskit_to_iqm_stage] + deepcopy(_STANDARD_CIRCUIT_STAGES),
        pulse_stages=deepcopy(_STANDARD_PULSE_STAGES),
        final_stages=deepcopy(_STANDARD_FINAL_STAGES),
        pp_stages=pp_stages,
        compiler_options=options,
        backend=backend,
    )


def sweep_job_to_qiskit(
    job: PullaJob,
    *,
    shots: int,
) -> Result:
    """Convert a completed Pulla job to a Qiskit Result.

    Args:
        job: The completed job to convert.
        shots: Number of shots requested.
        execution_options: Circuit execution options used to produce the result.

    Returns:
        The equivalent Qiskit Result.

    """
    circuit_execution_results = job.result()
    if circuit_execution_results is None:
        raise ValueError(
            f'Cannot format Qiskit result without result measurements. Job status is "{job.status.upper()}"'
        )

    if circuit_execution_results.circuit_measurement_results is None:
        raise ValueError("Cannot format station control result without result.")

    used_heralding = sum(HERALDING_KEY in key for key in circuit_execution_results.sweep_results.keys()) > 0

    # Convert the measurement results from a batch of circuits into the Qiskit format.
    batch_results: list[tuple[str, list[str]]] = [
        # TODO: Proper naming instead of "index"
        (
            f"{index}",
            IQMJob._iqm_format_measurement_results(
                circuit_measurements, requested_shots=shots, expect_exact_shots=used_heralding
            ),
        )
        for index, circuit_measurements in enumerate(circuit_execution_results.circuit_measurement_results)
    ]

    result_dict = {
        "backend_name": "IQMPullaBackend",
        "backend_version": "",
        "qobj_id": "",
        "job_id": str(job.job_id),
        "success": job.status == JobStatus.COMPLETED,
        "date": date.today().isoformat(),
        "status": str(job.status),
        "timeline": job.data.timeline.copy(),
        "results": [
            {
                "shots": len(measurement_results),
                "success": True,
                "data": {
                    "memory": measurement_results,
                    "counts": Counts(Counter(measurement_results)),
                    "metadata": {},
                },
                "header": {"name": name},
            }
            for name, measurement_results in batch_results
        ],
    }
    return Result.from_dict(result_dict)


# QISKIT COMPILER PASSES AND STAGES


def parallelize_and_transpile(  # noqa: PLR0913
    circuits: list[QuantumCircuit],
    components: ComponentGrouping | None,
    context: dict[str, Any],
    perform_move_routing: bool = True,
    optimize_single_qubits: bool = True,
    ignore_barriers_in_1qb_optimization: bool = False,
    remove_final_rzs: bool = True,
    existing_moves_handling: str | None = None,
    optimization_level: int = 0,  # below qiskit native transpile kwargs
    seed_transpiler: int | None = None,
    num_processes: int | None = None,
) -> list[list[QuantumCircuit]]:
    """Transpile Qiskit circuits and parallelize them if colour grouped components were inputted.

    Args:
        circuits: List of Qiskit QuantumCircuit objects to transpile and potentially parallelize.
        components: List of (physical) components on which to transpile (route) the circuits. If a flat list of
            components is provided, the IQMTarget will be built only on that subset of the full QPU. If colour
            grouped components (i.e. of the form ``list[list[tuple(str, ...)]]``) is provided, the circuits will
            be parallelized such that each colour group becomes its own circuit, and the circuit will be broadcasted
            to parallel groups within a colour group, i.e. executed parallelly. If ``None`` is provided, the default
            target for the full QPU will be used.
        context: The Compiler context.
        perform_move_routing: Whether to perform MOVE gate routing.
        optimize_single_qubits: Whether to optimize single qubit gates away.
        ignore_barriers_in_1qb_optimization: Whether to ignore barriers when optimizing single qubit gates.
        remove_final_rzs: Whether to remove the final z rotations.
        existing_moves_handling: How to handle existing MOVE gates in the circuit, required if the circuit contains
            MOVE gates.
        optimization_level: The optimization level of the Qiskit transpiler.
        seed_transpiler: The seed of the Qiskit transpiler.
        num_processes: The number of parallel processes to use.

    Returns:
        Transpiled and possibly parallelized circuits. The circuit(s) in each inner list are executed in parallel.
        If there is no parallelization, each inner list has just one item.

    """
    qiskit_kwargs: dict[str, Any] = {
        "backend": context["backend"],
        "perform_move_routing": perform_move_routing,
        "optimize_single_qubits": optimize_single_qubits,
        "ignore_barriers": ignore_barriers_in_1qb_optimization,
        "remove_final_rzs": remove_final_rzs,
        "existing_moves_handling": ExistingMoveHandlingOptions(existing_moves_handling)
        if existing_moves_handling
        else None,
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
        "num_processes": num_processes,
    }
    transpiled_circuits: list[list[QuantumCircuit]] = []
    if components is not None and components.grouping_mode == ComponentGroupingMode.COLOUR_GROUP:
        # parallelize the circuit(s)
        if not (len(circuits) == 1 or {len(par_circs) for par_circs in components} == {len(circuits)}):
            raise RuntimeError(
                "Parallelization only available for a single circuit parallelized over multiple"
                " colour groups or a separate circuit for each parallel group (i.e. the same number"
                " of circuits and parallel groups in every colour group."
            )
        for colour in components:
            parallel_circuits: list[QuantumCircuit] = []
            circuits_to_broadcast = [circuits[0]] * len(colour) if len(circuits) == 1 else circuits
            for group, circuit in zip(colour, circuits_to_broadcast):
                qiskit_kwargs["restrict_to_qubits"] = list(group)
                qiskit_kwargs["initial_layout"] = [idx for idx, _ in enumerate(group) if idx < circuit.num_qubits]
                parallel_circuits.append(transpile_to_IQM(circuit, **qiskit_kwargs))
            transpiled_circuits.append(parallel_circuits)
    else:
        for circuit in circuits:
            if components is not None:
                qiskit_kwargs["initial_layout"] = [idx for idx, _ in enumerate(components) if idx < circuit.num_qubits]
                qiskit_kwargs["restrict_to_qubits"] = components.flatten()
            transpiled_circuits.append([transpile_to_IQM(circuit, **qiskit_kwargs)])
    return transpiled_circuits


def qiskit_circuits_to_iqm_circuits(
    circuits: list[list[QuantumCircuit]], components: ComponentGrouping | None, context: dict[str, Any]
) -> list[Circuit]:
    """Convert Qiskit QuantumCircuits to IQM circuits.

    Args:
        circuits: Qiskit QuantumCircuit objects to compile. The circuits in each inner list are executed in parallel.
        components: Physical components on which to compile the circuits. If ``None``, will use the default IQMTarget
            in the Qiskit backend, otherwise restricts to these components.
        context: The Compiler context.

    Returns:
        Converted IQM circuits.

    """
    if components is not None:
        iqm_circuits: list[Circuit] = []
        colour_groups = (
            components
            if components.grouping_mode == ComponentGroupingMode.COLOUR_GROUP
            else [[tuple(components)]] * len(circuits)
        )
        for colour_group, parallel_circuits in zip(colour_groups, circuits):
            iqm_instructions = []
            for parallel_circuit, parallel_group in zip(parallel_circuits, colour_group):
                qubit_idx_to_name = dict(enumerate(parallel_group))
                iqm_instructions.extend(
                    list(qiskit_circuits_to_pulla(parallel_circuit, qubit_idx_to_name)[0].instructions)
                )
            iqm_circuits.append(
                Circuit(
                    name=f"Parallel Circuit on {colour_group}",
                    instructions=tuple(iqm_instructions),
                )
            )
        return iqm_circuits
    idx_mapping = context["backend"].target.iqm_idx_to_component
    return qiskit_circuits_to_pulla([group[0] for group in circuits], idx_mapping)


qiskit_transpilation_stage = CompilationStage(
    name="qiskit_transpilation", info="Transpile and route Qiskit circuits to the correct architecture."
)
qiskit_transpilation_stage.add_passes(parallelize_and_transpile)
qiskit_to_iqm_stage = CompilationStage(
    name="qiskit_to_iqm", info="Convert Qiskit circuits into the internal circuit representation."
)
qiskit_to_iqm_stage.add_passes(qiskit_circuits_to_iqm_circuits)
