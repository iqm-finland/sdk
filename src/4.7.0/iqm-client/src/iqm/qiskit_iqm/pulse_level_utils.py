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
"""Utilities for running pulse-level jobs using Qiskit classes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from datetime import date
from typing import Any

from iqm.iqm_client.transpile import ExistingMoveHandlingOptions
from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMBackend
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import QuantumCircuit
from qiskit.result import Counts, Result

from exa.common.data.setting_node import SettingNode
from exa.common.errors.iqm_error import CircuitExecutionError, NotFoundError
from iqm.cpc.compiler.compiler import CompilationStage, Compiler, StagesList
from iqm.cpc.compiler.standard_stages import HERALDING_KEY
from iqm.cpc.core.config import ComponentGrouping, ComponentGroupingMode
from iqm.cpc.core.observation.observation_loading_rules import RuleType
from iqm.pulse import Circuit, CircuitOperation
from iqm.pulse.quantum_ops import QuantumOp
from iqm.station_control.client.job_trackers import RunJobTracker
from iqm.station_control.interface.models.jobs_new import JobStatus


def qiskit_to_iqm(
    backend: IQMBackend,
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
    *,
    loading_rules: list[RuleType] | None = None,
) -> tuple[list[Circuit], Compiler]:
    """Convert transpiled Qiskit quantum circuits to IQM format, for compiling to pulse level.

    Also provides the Compiler object for compiling them, with the correct
    calibration set and component mapping initialized.

    Args:
        backend: qiskit-iqm backend used to transpile the circuits. Determines
            the calibration set to be used by the returned compiler.
        qiskit_circuits: One or many transpiled Qiskit QuantumCircuits to convert.
        loading_rules: Optional list of rules to use for loading the circuit. If not provided, will load the default
            calibration set from the backend.

    Returns:
        Equivalent IQM circuit(s), compiler for compiling them.

    """
    # build a qiskit-iqm CircuitJobDefinition, then prepare to compile and execute it
    # TODO can we skip this?
    circuit_job_definition = backend.create_circuit_job_definition(qiskit_circuits, shots=1)
    if circuit_job_definition.calibration_set_id is None:
        raise ValueError("CircuitJobDefinition created by IQMBackend has no calibration set id.")

    # create a compiler containing all the required station information
    compiler = backend.client.get_standard_compiler(exa_style_pp=False, loading_rules=loading_rules)
    compiler.component_mapping = circuit_job_definition.qubit_mapping
    # We can be certain circuit_job_definition contains only Circuit objects, because we created it
    # right in this method with qiskit.QuantumCircuit objects
    circuits: list[Circuit] = [c for c in circuit_job_definition.circuits if isinstance(c, Circuit)]
    return circuits, compiler


def sweep_job_to_qiskit(
    job_tracker: RunJobTracker,
    *,
    shots: int,
) -> Result:
    """Query the results of a completed pulse-level job and convert it to a Qiskit Result.

    Args:
        job_tracker: Tracker for the completed job.
        shots: Number of shots that was requested. Only used for validating the result.

    Returns:
        The equivalent Qiskit Result.

    Raises:
        NotFoundError: Job has not finished yet.

    """
    result_dict = {
        "backend_name": "IQMBackend",
        "backend_version": "",
        "qobj_id": "",
        "job_id": str(job_tracker.job_id),
        "success": False,
        "date": date.today().isoformat(),
        "results": [],
        # the ones below go into result._metadata
        "timeline": job_tracker.job_data.timeline.copy(),
    }
    status = job_tracker.update()

    # try to return results if job completed successfully
    # job is successful iff it is JobStatus.COMPLETED and we got result data
    if status == JobStatus.COMPLETED:
        if (circuit_execution_results := job_tracker.result()) is None:
            raise CircuitExecutionError(f"Job results are not available. Job status is {status}")

        used_heralding = any(
            HERALDING_KEY in readout_label for readout_label in circuit_execution_results.sweep_results
        )
        # Convert the measurement results from a batch of circuits into the Qiskit format.
        batch_results: list[tuple[str, list[str]]] = [
            (
                f"{index}",  # TODO: Proper circuit/schedule names instead of "index", from payload?
                IQMJob._iqm_format_measurement_results(
                    circuit_measurements, requested_shots=shots, expect_exact_shots=not used_heralding
                ),
            )
            for index, circuit_measurements in enumerate(circuit_execution_results.circuit_measurement_results)
        ]
        # fill in the results
        result_dict["success"] = True
        result_dict["results"] = [
            {
                "shots": len(measurement_results),
                "success": True,
                "data": {
                    "memory": measurement_results,
                    "counts": Counts(Counter(measurement_results)),
                    "metadata": {},
                },
                "header": {"name": name},
                "calibration_set_id": job_tracker.job_data.compilation.calibration_set_id
                if job_tracker.job_data.compilation
                else None,
            }
            for name, measurement_results in batch_results
        ]
    elif status not in JobStatus.terminal_statuses():
        raise NotFoundError(f"Job hasn't finished yet, status is {status}.")
    return Result.from_dict(result_dict)


# =======================
# Qiskit compiler
# =======================


def get_qiskit_compiler(
    backend: IQMBackend,
    *,
    loading_rules: list[RuleType] | None = None,
    exa_style_pp: bool = True,
    controller_mapping: dict[str, dict[str, str]] | None = None,
    gate_definitions: dict[str, QuantumOp] | None = None,
) -> Compiler:
    """IQM Compiler for transforming Qiskit circuits into pulse-level jobs.

    Creates a Compiler instance which contains a Qiskit backend and extra initial circuit stages for
    parallelizing and transpiling Qiskit circuits and converting them to IQM circuit format.

    Args:
        backend: Qiskit backend.
        loading_rules: Observation loading rules. If ``None``, will use the current default calibration set.
        exa_style_pp: Whether to do EXA-style dataset post-processing by default.
        controller_mapping: Dictionary that maps physical QPU component names to their device controller names.
            The dictionary is of the form: ``{<component_name>: {<operation_name>: <controller name>}}``,
            where operation is one of the following: "drive", "readout", "flux"
            (not all components have all operations supported).
        gate_definitions: Names of quantum operations mapped to their definitions, see :class:`.QuantumOp`.

    Returns:
        Qiskit-specific IQM Compiler.

    """
    client = backend.client
    compiler = client.get_standard_compiler(
        loading_rules,
        exa_style_pp=exa_style_pp,
        controller_mapping=controller_mapping,
        gate_definitions=gate_definitions,
    )
    compiler.name = "IQM Qiskit compiler"
    # add initial passes
    compiler.circuit_stages = StagesList([qiskit_transpilation_stage, qiskit_to_iqm_stage] + compiler.circuit_stages)

    # monkeypatch the compiler_context method of ``compiler`` to include the backend
    def compiler_context(self: Compiler, components: ComponentGrouping | None, settings: SettingNode) -> dict[str, Any]:
        """Adds the Qiskit backend to the Compiler context."""
        context = Compiler.compiler_context(self, components, settings)  # avoid infinite recursion
        context["backend"] = backend
        return context

    # Mypy does not like us replacing the method, hence the type ignore.
    compiler.compiler_context = compiler_context.__get__(compiler)  # type: ignore[method-assign]
    return compiler


def _qiskit_circuits_to_iqm(
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
    qubit_idx_to_name: dict[int, str],
    custom_gates: Collection[str] = (),
) -> list[Circuit]:
    """Convert Qiskit quantum circuits into IQM quantum circuits.

    Args:
        qiskit_circuits: One or many Qiskit quantum circuits to convert.
        qubit_idx_to_name: Mapping from Qiskit qubit indices to the names of the corresponding
            qubit names.
        custom_gates: Names of custom gates that should be treated as additional native gates
            by qiskit-iqm, i.e. they should be passed as-is to the compiler.

    Returns:
        Equivalent IQM circuit(s).

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
        circuits: Qiskit quantum circuits to transpile and potentially parallelize.
        components: Physical QPU components on which to transpile (route) the circuits. If a flat list of
            components is provided, the :class:`.IQMTarget` will be built only on that subset of the full QPU.
            If colour grouped components are provided, the circuits will be parallelized such that
            each colour group becomes its own circuit, and the circuit will be broadcasted
            to parallel groups within a colour group, i.e. executed parallelly.
            If ``None``, the default target for the full QPU will be used.
        context: Compiler context.
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
        group_counts = {len(par_circs) for par_circs in components}
        n_circuits = len(circuits)
        n_colours = len(components)
        mode_broadcast = n_circuits == 1
        mode_per_group = group_counts == {n_circuits}
        mode_per_colour = (not mode_broadcast) and (not mode_per_group) and n_circuits == n_colours
        if not (mode_broadcast or mode_per_group or mode_per_colour):
            raise RuntimeError(
                "Parallelization only available for a single circuit parallelized over multiple"
                " colour groups, a separate circuit for each parallel group (i.e. the same number"
                " of circuits and parallel groups in every colour group), or one circuit per"
                " colour group (broadcast within each colour)."
            )
        for colour_idx, colour in enumerate(components):
            parallel_circuits: list[QuantumCircuit] = []
            if mode_broadcast:
                circuits_to_use = [circuits[0]] * len(colour)
            elif mode_per_group:
                circuits_to_use = circuits
            else:  # mode_per_colour
                circuits_to_use = [circuits[colour_idx]] * len(colour)
            for group, circuit in zip(colour, circuits_to_use):
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
    circuits: Sequence[Iterable[QuantumCircuit]],
    components: ComponentGrouping | None,
    context: dict[str, Any],
) -> list[Circuit]:
    """Convert Qiskit QuantumCircuits to IQM circuits.

    Args:
        circuits: Qiskit QuantumCircuit objects to compile. The circuits in each inner list are executed in parallel.
        components: Physical components on which to compile the circuits. If ``None``, will use the default
            :class:`.IQMTarget` in the Qiskit backend, otherwise restricts to these components.
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
            iqm_instructions: list[CircuitOperation] = []
            for parallel_circuit, parallel_group in zip(parallel_circuits, colour_group):
                qubit_idx_to_name = dict(enumerate(parallel_group))
                iqm_instructions.extend(_qiskit_circuits_to_iqm(parallel_circuit, qubit_idx_to_name)[0].instructions)
            iqm_circuits.append(
                Circuit(
                    name=f"Parallel Circuit on {colour_group}",
                    instructions=tuple(iqm_instructions),
                )
            )
        return iqm_circuits
    idx_mapping = context["backend"].target.iqm_idx_to_component
    return _qiskit_circuits_to_iqm([next(iter(group)) for group in circuits], idx_mapping)


qiskit_transpilation_stage = CompilationStage(
    name="qiskit_transpilation", info="Transpile and route Qiskit circuits to the correct architecture."
)
qiskit_transpilation_stage.add_passes(parallelize_and_transpile)
qiskit_to_iqm_stage = CompilationStage(
    name="qiskit_to_iqm", info="Convert Qiskit circuits into the internal circuit representation."
)
qiskit_to_iqm_stage.add_passes(qiskit_circuits_to_iqm_circuits)
