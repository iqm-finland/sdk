# Copyright 2022-2026 IQM
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

"""Utility functions for working with Qiskit circuits."""

from collections.abc import Sequence
from enum import StrEnum, auto
import logging
from random import SystemRandom

from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Barrier, Delay, Measure, Reset
from qiskit.circuit.library import QFT, QuantumVolume
from qiskit.circuit.random import random_circuit

logger = logging.getLogger(__name__)
CZ_LOCUS_LENGTH = 2  # Number of qubits involved in CZ gate


class CircuitType(StrEnum):
    """Defines quantum circuit types supported by ``qubit_selector``."""

    GHZ = auto()
    QFT = auto()
    QUANTUM_VOLUME = auto()
    QAOA = auto()
    RANDOM = auto()
    WSTATE = auto()


def _cost_operator(qc: QuantumCircuit, gamma: float, num_qubits: int) -> None:
    """Apply the cost Hamiltonian for QAOA algorithm.

    Args:
        qc: Quantum circuit to apply the cost operator to.
        gamma: Angle for the RZ gate.
        num_qubits: Number of qubits in the circuit.

    """
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(-2 * gamma, i + 1)
        qc.cx(i, i + 1)


def _mixer_operator(qc: QuantumCircuit, beta: float, num_qubits: int) -> None:
    """Apply the mixer Hamiltonian for QAOA algorithm.

    Args:
        qc: Quantum circuit to apply the mixer operator to.
        beta: Angle for the RX gate.
        num_qubits: Number of qubits in the circuit.

    """
    for i in range(num_qubits):
        qc.rx(2 * beta, i)


def _create_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a GHZ circuit."""
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q, name=CircuitType.GHZ)
    qc.h(q[-1])
    for i in range(1, num_qubits):
        qc.cx(q[num_qubits - i], q[num_qubits - i - 1])
    return qc


def _create_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a QFT circuit."""
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q, name=CircuitType.QFT)
    qc.h(q[-1])
    for i in range(1, num_qubits):
        qc.cx(q[num_qubits - i], q[num_qubits - i - 1])
    qc.compose(QFT(num_qubits=num_qubits), inplace=True)
    return qc


def _create_quantum_volume_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a Quantum Volume circuit."""
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q, name=CircuitType.QUANTUM_VOLUME)
    num_reduced_qubits = 0
    while num_reduced_qubits < num_qubits:
        qc = QuantumVolume(num_qubits, depth=num_qubits, classical_permutation=True)
        active_qubits, _ = active_bits(qc.decompose())
        num_reduced_qubits = len(active_qubits)
        if num_reduced_qubits < num_qubits:
            qc.clear()
    return qc


def _create_qaoa_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a QAOA circuit."""
    p = 1
    np.random.seed(0)
    params = np.random.uniform(0, np.pi, 2 * p)
    gamma = params[:p]
    beta = params[p:]
    qc = QuantumCircuit(num_qubits, name=CircuitType.QAOA)
    qc.h(range(num_qubits))
    for layer in range(p):
        _cost_operator(qc, gamma[layer], num_qubits)
        _mixer_operator(qc, beta[layer], num_qubits)
    return qc


def _create_random_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a random circuit."""
    a = SystemRandom().randrange(2**32 - 1)
    return random_circuit(num_qubits, num_qubits * 2, measure=False, seed=a)


def _create_wstate_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a W-state circuit."""
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q, name=CircuitType.WSTATE)

    def f_gate(qc: QuantumCircuit, q: QuantumRegister, i: int, j: int, n: int, k: int) -> None:
        theta = np.arccos(np.sqrt(1 / (n - k + 1)))
        qc.ry(-theta, q[j])
        qc.cz(q[i], q[j])
        qc.ry(theta, q[j])

    qc.x(q[-1])
    for m in range(1, num_qubits):
        f_gate(qc, q, num_qubits - m, num_qubits - m - 1, num_qubits, m)
    for k in reversed(range(1, num_qubits)):
        qc.cx(k - 1, k)
    return qc


def get_circuit(circuit_type: CircuitType, num_qubits: int) -> QuantumCircuit:
    """Create a quantum circuit of a given type.

    Args:
        circuit_type: Type of the circuit to create.
        num_qubits: Number of qubits.

    Returns:
        Quantum circuit of a given type.

    Raises:
        ValueError: If the requested circuit type is not supported.

    """
    circuit_creators = {
        CircuitType.GHZ: _create_ghz_circuit,
        CircuitType.QFT: _create_qft_circuit,
        CircuitType.QUANTUM_VOLUME: _create_quantum_volume_circuit,
        CircuitType.QAOA: _create_qaoa_circuit,
        CircuitType.RANDOM: _create_random_circuit,
        CircuitType.WSTATE: _create_wstate_circuit,
    }

    if circuit_type not in circuit_creators:
        logger.error("Quantum circuit of unsupported type %s was requested", circuit_type)
        raise ValueError(f"Valid quantum circuit types are {', '.join([type_.value for type_ in CircuitType])}.")

    qc = circuit_creators[circuit_type](num_qubits)
    qc.measure_all()
    return qc


def perform_backend_transpilation(
    qc_list: list[QuantumCircuit],
    backend: IQMBackendBase,
    qubits: Sequence[int],
    coupling_map: list[list[int]],
    basis_gates: Sequence[str] = ("r", "cz"),
    qiskit_optim_level: int = 1,
    optimize_sqg: bool = True,
    routing_method: str | None = "sabre",
) -> list[QuantumCircuit]:
    """Transpile a list of circuits to backend specifications.

    Args:
        qc_list: The original (untranspiled) list of quantum circuits.
        backend: The backend to execute the benchmark on.
        qubits: The qubits to target in the transpilation.
        coupling_map: The target coupling map to transpile to.
        basis_gates: The basis gates.
        qiskit_optim_level: Qiskit ``optimization_level`` value.
        optimize_sqg: Whether SQG optimization is performed taking into account virtual Z.
        routing_method: The routing method employed by Qiskit's transpilation pass.

    Returns:
        A list of transpiled quantum circuits.

    """

    # Helper function considering whether optimize_sqg is done,
    # and whether the coupling map is reduced (whether final physical layout must be fixed onto an auxiliary QC)
    def transpile_and_optimize(qc: QuantumCircuit, aux_qc: QuantumCircuit | None = None) -> QuantumCircuit:
        if backend.has_resonators():
            coupling_map_red = (
                backend.coupling_map.reduce(qubits[: qc.num_qubits]) if aux_qc is not None else coupling_map
            )
            transpiled = transpile_to_IQM(
                qc,
                backend=backend,
                optimize_single_qubits=optimize_sqg,
                remove_final_rzs=True,
                coupling_map=coupling_map_red,
            )
        else:
            transpiled = transpile(
                qc,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=qiskit_optim_level,
                initial_layout=qubits if aux_qc is None else None,
                routing_method=routing_method,
            )
            if aux_qc is not None:
                transpiled = aux_qc.compose(transpiled, qubits=qubits, clbits=list(range(qc.num_clbits)))
            if optimize_sqg:
                transpiled = optimize_single_qubit_gates(transpiled, drop_final_rz=True)
        return transpiled

    if coupling_map == backend.coupling_map:
        transpiled_qc_list = [transpile_and_optimize(qc) for qc in qc_list]
    else:  # The coupling map will be reduced if the physical layout is to be fixed
        if backend.has_resonators():
            aux_qc_list = [QuantumCircuit(backend.num_qubits, q.num_clbits) for q in qc_list]
        else:
            aux_qc_list = [QuantumCircuit(backend.num_qubits, q.num_clbits) for q in qc_list]
        transpiled_qc_list = [transpile_and_optimize(qc, aux_qc=aux_qc_list[idx]) for idx, qc in enumerate(qc_list)]

    return transpiled_qc_list


def reduce_to_active_qubits(
    circuit: QuantumCircuit, backend_topology: str | None = None, backend_num_qubits: int | None = None
) -> QuantumCircuit:
    """Reduces a quantum circuit to only its active qubits.

    Args:
        circuit: The original quantum circuit.
        backend_topology: The backend topology to execute the benchmark on.
        backend_num_qubits: The number of qubits in the backend.

    Returns:
        A new quantum circuit containing only active qubits.

    """
    # Identify active qubits
    active_qubits = set()

    for instruction in circuit.data:
        for qubit in instruction.qubits:
            active_qubits.add(circuit.find_bit(qubit).index)

    if backend_topology == "star" and backend_num_qubits not in active_qubits:
        # For star systems, the resonator must always be there, regardless of whether it MOVE gates on it or not
        active_qubits.add(backend_num_qubits)

    # Create a mapping from old qubits to new qubits
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_qubits)}

    # Create a new quantum circuit with the reduced number of qubits
    reduced_circuit = QuantumCircuit(len(active_qubits))

    # Add classical registers if they exist
    if circuit.num_clbits > 0:
        creg = ClassicalRegister(circuit.num_clbits)
        reduced_circuit.add_register(creg)

    # Copy operations to the new circuit, remapping qubits and classical bits
    for instruction in circuit.data:
        new_qubits = [reduced_circuit.qubits[qubit_map[circuit.find_bit(qubit).index]] for qubit in instruction.qubits]
        new_clbits = [reduced_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        reduced_circuit.append(instruction.operation, new_qubits, new_clbits)

    return reduced_circuit


def extract_2q_interactions(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """Extract 2-qubit interactions excluding directives and single-qubit ops.

    Args:
        circuit: The quantum circuit from which to extract 2-qubit interactions.

    Returns:
        Interactions as a list of tuples, each containing the indices of two qubits that interact.
            The qubit indices are returned in a consistent order (min, max).

    """
    interactions = []

    for instr, qargs, _ in circuit.data:
        # Skip known directive or non-unitary instructions
        if isinstance(instr, (Barrier, Measure, Reset, Delay)):
            continue
        if len(qargs) == CZ_LOCUS_LENGTH:
            q0 = circuit.find_bit(qargs[0]).index
            q1 = circuit.find_bit(qargs[1]).index
            interactions.append((min(q0, q1), max(q0, q1)))  # Consistent order

    return interactions


def deflate_circuit(input_circ: QuantumCircuit) -> QuantumCircuit:
    """Reduce a transpiled circuit down to only active qubits.

    Args:
        input_circ: Input circuit.

    Returns:
        Reduced circuit.

    """
    active_qubits, active_clbits = active_bits(input_circ)
    active_qubit_map = {}
    active_bit_map = {}

    for idx, val in enumerate(sorted(active_qubits, key=lambda x: input_circ.find_bit(x).index)):
        active_qubit_map[val] = idx

    for idx, val in enumerate(sorted(active_clbits, key=lambda x: input_circ.find_bit(x).index)):
        active_bit_map[val] = idx

    new_qc = QuantumCircuit(len(active_qubits), len(active_clbits))

    for item in input_circ.data:
        # Find active qubits used by instruction (if any)
        used_active_set = [qubit for qubit in item[1] if qubit in active_qubits]
        # If any active qubits used, add to deflated circuit
        if any(used_active_set):
            ref = getattr(new_qc, item[0].name)
            params = item[0].params
            qargs = [new_qc.qubits[active_qubit_map[qubit]] for qubit in used_active_set]
            cargs = [new_qc.clbits[active_bit_map[clbit]] for clbit in item[2]]
            ref(*params, *qargs, *cargs)

    new_qc.global_phase = input_circ.global_phase
    return new_qc


def active_bits(input_circ: QuantumCircuit) -> tuple[set[int], set[int]]:
    """Find active bits (quantum and classical) in a transpiled circuit.

    Args:
        input_circ: Input circuit.

    Returns:
        A tuple containing two sets - one with active qubits and one with active classical bits.

    """
    active_qubits = set()
    active_clbits = set()

    for item in input_circ.data:
        if item[0].name not in ["barrier", "delay"]:
            qubits = item[1]
            for qubit in qubits:
                active_qubits.add(qubit)
            clbits = item[2]
            for clbit in clbits:
                active_clbits.add(clbit)

    return active_qubits, active_clbits
