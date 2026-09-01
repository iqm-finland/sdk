# Copyright 2026 IQM
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

"""Meta-functions for SABRE routing with parallel tempering and sectionizing."""

from __future__ import annotations

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import (
    CircuitDAG,
    QPUTopology,
    permute_instruction_dag,
    prune_dag,
)
from iqm.qrisp_iqm.passes.routing.core.permutation_tools import invert_permutation, mul_perm
from iqm.qrisp_iqm.passes.routing.core.sabre_workflow import InstructionRef, sabre_gen_swaps
from numba import njit, prange
import numpy as np
import psutil
from qrisp import QuantumCircuit
from qrisp.circuit import Instruction

cpu_count: int = psutil.cpu_count() or 1


@njit(parallel=True, cache=True)
def sabre_gen_swaps_parallel(  # noqa: PLR0913
    topology: QPUTopology,
    instruction_dag: CircuitDAG,
    greediness: int,
    max_instruction: int,
    threads: int,
    depth_array: np.ndarray,
    tempering_range: int = 3,
    congestion_penalty: float = 0.0,
    selection_exponent: float = 0.0,
    seed: int = 0,
) -> tuple[list[InstructionRef], np.ndarray]:
    """Execute sabre_gen_swaps in parallel with different seeds and greediness values.

    Uses parallel tempering across threads, then selects the trial with the
    lowest score. The score is a weighted geometric mean of swap count and
    2-qubit depth::

        score = swaps^(1 - e) * depth^e

    When ``selection_exponent`` is 0 (default) the selection is purely by swap
    count; when 1.0 it is purely by depth.

    Args:
        topology: Available gates.
        instruction_dag: Quantum circuit in DAG format.
        greediness: Controls swap selection randomness. The algorithm will skip the best
            swap candidate with probability 1/greediness. Higher values mean more
            greedy (deterministic) behavior.
        max_instruction: The sabre_gen_swaps will only compile until this threshold is reached.
        threads: The amount of instances that should be executed in parallel (recommended
            to be an integer multiple of the CPU count).
        depth_array: Per-physical-qubit depth array. Carried across sections so that depth
            accounting is continuous. Each parallel trial receives a copy.
        tempering_range: Controls parallel tempering. When > 0, each thread uses a different
            greediness value spread across
            [max(2, greediness - tempering_range), greediness + tempering_range].
            This gives different threads different exploration/exploitation tradeoffs
            (analogous to running replicas at different temperatures in simulated
            annealing). When 0, all threads use the same greediness.
        congestion_penalty: Strength of the per-physical-qubit congestion penalty in swap scoring.
            Forwarded to sabre_gen_swaps. 0.0 disables.
        selection_exponent: Blending weight for the geometric-mean selection metric (0.0–1.0).
            0.0 selects purely by swap count, 1.0 purely by depth.
        seed: Base seed for the random number generation. Each parallel thread
            derives its own seed as ``seed + i*173 + 1``.

    Returns:
        * Compiled instructions from the trial with the lowest score.
          SWAPs have instr_index = -1.
        * Final logical-to-physical qubit mapping of the best solution. The dtype is inherited
          from the topology (int16 on NISQ-scale devices).

    """
    # Determine the amount of qubits
    qubit_amount = topology.dist_matrix.shape[0]
    qdt = topology.dist_matrix.dtype  # inherit narrow qubit dtype from topology

    # This array will hold the score for each trial
    trials_scores = np.zeros(threads, dtype=np.float64)

    # This list will hold the compile instructions for each run
    resulting_instructions = [[(np.int16(x), np.int16(x), np.int16(x)) for x in range(0)] for _ in range(threads)]

    # This list will hold the final permutations for each run
    resulting_permutations = np.zeros((threads, qubit_amount), dtype=qdt)

    depth_arrays = np.broadcast_to(depth_array, (threads, len(depth_array)))

    # Clamp selection_exponent to [0, 1]
    dp = min(1.0, max(0.0, selection_exponent))

    # Parallel tempering: compute greediness range for thread distribution
    g_min = max(np.int64(2), np.int64(greediness) - np.int64(tempering_range))
    g_max = np.int64(greediness) + np.int64(tempering_range)

    # Run through the tasks in parallel
    for i in prange(threads):
        # Each thread gets a different greediness when tempering_range > 0
        if tempering_range > 0 and threads > 1:
            thread_greediness = g_min + (g_max - g_min) * i // (threads - 1)
        else:
            thread_greediness = greediness

        compiled_instructions, c2a = sabre_gen_swaps(
            topology,
            instruction_dag,
            thread_greediness,
            max_instruction,
            seed=seed + i * 173 + 1,
            initial_phys_depth=depth_arrays[i],
            congestion_penalty=congestion_penalty,
        )
        resulting_permutations[i, :] = c2a
        resulting_instructions[i] = compiled_instructions

        # Count inserted SWAPs (instr_index == -1)
        swap_count = np.int64(0)
        for t_idx in range(len(compiled_instructions)):
            if compiled_instructions[t_idx][2] == np.int16(-1):
                swap_count += 1

        # Compute 2-qubit depth for this trial.
        # Copy the incoming depth so update_depth_array doesn't mutate
        # the shared depth_arrays.
        trial_depth = depth_arrays[i].copy().astype(qdt)
        update_depth_array(compiled_instructions, trial_depth)
        depth_val = np.int64(np.max(trial_depth))

        # Geometric mean: swaps^(1-dp) * depth^dp
        # Add 1 to both to avoid 0^x edge cases.
        s = np.float64(swap_count + 1)
        d = np.float64(depth_val + 1)
        trials_scores[i] = s ** (1.0 - dp) * d**dp

    # Determine the best run (lowest score)
    best_index = np.argmin(trials_scores)

    update_depth_array(resulting_instructions[best_index], depth_array)

    return resulting_instructions[best_index], resulting_permutations[best_index]


@njit(cache=True)
def sectionized_sabre(  # noqa: PLR0913
    topology: QPUTopology,
    instruction_dag: CircuitDAG,
    greediness: int,
    threads: int,
    section_lengths: np.ndarray,
    tempering_range: int = 3,
    congestion_penalty: float = 0.0,
    selection_exponent: float = 0.0,
    seed: int = 0,
) -> tuple[list[InstructionRef], np.ndarray]:
    """Execute the SABRE algorithm with "sectionizing" strategy.

    Sectionizing means dividing the QuantumCircuit into several sections,
    which are sequentially compiled.
    Each section is compiled multiple times with different seeds. The best seed
    is picked and the algorithm proceeds to the next section.

    Section boundaries are placed at semantically meaningful positions
    (e.g. after TerminatorNodes in the PermeabilityGraph) rather than at
    uniform intervals. The ``section_lengths`` array specifies how many DAG
    nodes belong to each section.

    The depth_array is carried forward between sections so that per-physical-qubit
    depth tracking remains continuous, enabling the congestion penalty to account
    for congestion accumulated in earlier sections.

    Args:
        topology:
            QPU topology.
        instruction_dag:
            Instruction DAG.
        greediness:
            Controls swap selection randomness.
        threads:
            The amount of instances that should be executed in parallel.
        section_lengths:
            1-D int32 array where each entry is the number of DAG nodes in that
            section. The last entry is treated as "compile everything remaining"
            (i.e. max_instruction=inf). The sum should equal the total DAG size
            but the last section will consume whatever is left regardless.
        tempering_range:
            Controls parallel tempering.
        congestion_penalty:
            Strength of the per-physical-qubit congestion penalty.
        selection_exponent:
            Blending weight for the geometric-mean selection metric (0.0–1.0).
            Forwarded to ``sabre_gen_swaps_parallel``.
        seed:
            Base seed for reproducibility. Each section derives its own seed
            offset so that different sections use non-overlapping seed ranges.

    Returns:
        * Instructions in the compiled QuantumCircuit.
        * Final logical to physical qubit index permutation.

    """
    qubit_amount = topology.dist_matrix.shape[0]
    n_sections = len(section_lengths)

    current_permutation = np.arange(qubit_amount, dtype=topology.dist_matrix.dtype)
    remaining_instruction_dag = instruction_dag
    compiled_instructions = [(np.int16(x), np.int16(x), np.int16(x)) for x in range(0)]
    current_depth_array = np.zeros(qubit_amount, dtype=topology.dist_matrix.dtype)

    for i in range(n_sections - 1):
        sl = section_lengths[i]

        compiled_section, c2a = sabre_gen_swaps_parallel(
            topology,
            remaining_instruction_dag,
            greediness,
            max_instruction=sl,
            threads=threads,
            depth_array=current_depth_array,
            tempering_range=tempering_range,
            congestion_penalty=congestion_penalty,
            selection_exponent=selection_exponent,
            seed=seed + i * 100003,
        )
        remaining_instruction_dag = prune_dag(remaining_instruction_dag, sl)
        remaining_instruction_dag = permute_instruction_dag(remaining_instruction_dag, c2a)
        current_permutation = mul_perm(c2a, current_permutation)
        compiled_instructions = compiled_instructions + compiled_section

    # Last section: compile everything remaining
    compiled_section, c2a = sabre_gen_swaps_parallel(
        topology,
        remaining_instruction_dag,
        greediness,
        max_instruction=np.inf,
        threads=threads,
        depth_array=current_depth_array,
        tempering_range=tempering_range,
        congestion_penalty=congestion_penalty,
        selection_exponent=selection_exponent,
        seed=seed + n_sections * 100003,
    )

    compilation_result = compiled_instructions + compiled_section
    final_permutation = mul_perm(c2a, current_permutation)

    return compilation_result, final_permutation


def instruction_tuples_to_qc(
    compiled_instruction_tuples: list[InstructionRef],
    instruction_list: list[Instruction | None],
    original_qc: QuantumCircuit,
    num_physical_qubits: int | None = None,
) -> QuantumCircuit:
    """Turn a list of compiled instruction tuples back into a QuantumCircuit.

    Uses the instruction list generated from ``convert_qc_to_sparse_dag``
    together with compiled instruction tuples.

    Args:
        compiled_instruction_tuples:
            The list of compiled instruction tuples.
        instruction_list:
            The list of instructions.
        original_qc:
            The QuantumCircuit which is the origin of both of the other arguments.
        num_physical_qubits:
            The number of physical qubits in the target topology. If greater than
            the original circuit's qubit count, ancilla qubits are added so that
            physical qubit indices produced by routing are valid.

    Returns:
        The transpiled QuantumCircuit.

    """
    # Create an empty copy of the QuantumCircuit
    new_qc = original_qc.clearcopy()

    # Extend with ancilla qubits if the topology is larger than the circuit
    if num_physical_qubits is not None:
        while new_qc.num_qubits() < num_physical_qubits:
            new_qc.add_qubit()

    # Maintain a running logical-to-physical qubit permutation (c2a).
    # This is needed to reconstruct multi-qubit no-ops like barriers, whose
    # instruction_data entries only encode a single qubit position but whose
    # Instruction objects reference all N logical qubits.  By tracking the
    # permutation as we process SWAPs we can map every barrier qubit to its
    # correct physical position at the point the barrier is placed.
    n_qubits = original_qc.num_qubits()
    if n_qubits > 0:
        c2a = np.arange(n_qubits, dtype=np.int32)
    else:
        c2a = np.array([], dtype=np.int32)

    # Iterate through the instruction tuples and append the instructions/swaps
    for i in range(len(compiled_instruction_tuples)):
        instruction_tuple = compiled_instruction_tuples[i]

        if instruction_tuple[2] == -1:
            # SWAP gate — update the running permutation and append to circuit.
            p0 = int(instruction_tuple[0])
            p1 = int(instruction_tuple[1])
            new_qc.swap(p0, p1)
            if n_qubits > 0:
                a2c = invert_permutation(c2a)
                c2a[a2c[p0]], c2a[a2c[p1]] = c2a[a2c[p1]], c2a[a2c[p0]]
            continue

        instruction = instruction_list[instruction_tuple[2]]

        if instruction is None:
            continue

        # Barriers need special treatment: the instruction_data encodes only
        # one qubit (single-qubit-like), but the barrier spans all qubits in
        # instruction.qubits.  We use the running permutation to map every
        # logical qubit to its current physical position.
        if instruction.op.name == "barrier":
            phys_qubits = [int(c2a[original_qc.qubits.index(q)]) for q in instruction.qubits]
            new_qc.append(instruction.op, phys_qubits, instruction.clbits)
        elif instruction_tuple[0] == instruction_tuple[1]:
            new_qc.append(instruction.op, [int(instruction_tuple[0])], instruction.clbits)
        else:
            new_qc.append(instruction.op, [int(instruction_tuple[0]), int(instruction_tuple[1])], instruction.clbits)

    return new_qc


@njit(cache=True)
def update_depth_array(instruction_list: list[InstructionRef], depth_array: np.ndarray) -> None:
    """Update the per-physical-qubit depth array from a compiled instruction list.

    Processes a list of instruction tuples and updates the depth tracking
    array in-place.  Two-qubit gates increment the depth of both qubits
    involved; SWAP gates receive a larger increment when both qubits are at
    the same depth (used as a "booster pad" to encourage efficient swap chains).

    Args:
        instruction_list:
            Instructions produced by SABRE.
        depth_array:
            Per-physical-qubit depth array to update in-place.

    """
    for i in range(len(instruction_list)):
        if instruction_list[i][0] != instruction_list[i][1]:
            if instruction_list[i][2] == -1:
                # If the depth of both qubits agrees,
                # this is most likely because the
                # gate has been executed, which can be
                # used as a "booster-pad". We want to
                # encourage the algorithm to use this,
                # which is why we compute the depth
                # cost of this accurately.
                if depth_array[instruction_list[i][0]] == depth_array[instruction_list[i][1]]:
                    increment = 1
                else:
                    increment = 3

                new_depth = (
                    np.maximum(depth_array[instruction_list[i][0]], depth_array[instruction_list[i][1]]) + increment
                )
            else:
                new_depth = np.maximum(depth_array[instruction_list[i][0]], depth_array[instruction_list[i][1]]) + 1

            depth_array[instruction_list[i][0]] = new_depth
            depth_array[instruction_list[i][1]] = new_depth
