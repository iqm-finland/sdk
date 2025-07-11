# Copyright (c) 2024-2025 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""This module contains several functions that build various types of circuits (e.g., in :mod:`qiskit` and :mod:`quimb`)
from the :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler.transpiler import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

with warnings.catch_warnings():
    # Importing quimb raises an annoying warning about different hyper-optimizers
    warnings.filterwarnings("ignore", category=UserWarning)
    import quimb.tensor as qtn

from iqm.iqm_client.transpile import ExistingMoveHandlingOptions
from iqm.qaoa.transpiler.hardwired.hardwired import hardwired_router
from iqm.qaoa.transpiler.quantum_hardware import CrystalQPUFromBackend, StarQPU
from iqm.qaoa.transpiler.sn.sn import sn_router
from iqm.qaoa.transpiler.sparse.greedy_router import greedy_router
from iqm.qaoa.transpiler.star.star import star_router
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_naive_move_pass import transpile_to_IQM

if TYPE_CHECKING:
    from iqm.qaoa.qubo_qaoa import QUBOQAOA


def qiskit_circuit(qaoa: QUBOQAOA, measurements: bool = True) -> QuantumCircuit:
    """Constructs a :class:`~qiskit.circuit.QuantumCircuit` from the QAOA angles, ignoring details of the QPU.

    Constructs a :class:`~qiskit.circuit.QuantumCircuit` corresponding to the QAOA, assuming perfect connectivity of
    the qubits and complete set of available quantum gates. This circuit can be used for simulations or it can be
    transpiled to be run on a real QPU.

    Args:
        qaoa: A :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose angles and interaction strengths are used in
            the construction of the :class:`~qiskit.circuit.QuantumCircuit`.
        measurements: Should measurements be added at the end of the circuit? If the circuit is used for statevector
            simulation, there shouldn't be measurements. If the circuit is used for sampling, there should be
            measurements.

    Returns:
        A quantum circuit corresponding to the QAOA, excluding any measurements.

    """
    qc = QuantumCircuit(qaoa.num_qubits)
    for qubit in range(qaoa.num_qubits):
        qc.h(qubit)
    for p in range(qaoa.num_layers):
        for qubit in range(qaoa.num_qubits):
            qc.rz(2 * qaoa.angles[2 * p] * qaoa.bqm.get_linear(qubit), qubit)
        for q1, q2 in qaoa.bqm.quadratic:
            qc.rzz(2 * qaoa.angles[2 * p] * qaoa.bqm.get_quadratic(q1, q2), q1, q2)
        for qubit in range(qaoa.num_qubits):
            qc.rx(2 * qaoa.angles[2 * p + 1], qubit)
    if measurements:
        qc.measure_all()
    return qc


def qiskit_circuit_specific_nodes(qaoa: QUBOQAOA, starting_qubits: set[int]) -> QuantumCircuit:
    """Constructs a :class:`~qiskit.circuit.QuantumCircuit` for the RCC of given qubits, ignoring details of the QPU.

    The *reverse causal cone* (RCC) of a set of qubits contains all the gates and qubits which have any influence
    on the measurement results on the initial set of qubits. This method constructs
    a :class:`~qiskit.circuit.QuantumCircuit` containing these gates and extra qubits.

    Args:
        qaoa: A :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose angles and interaction strengths are used in
            the construction of the :class:`~qiskit.circuit.QuantumCircuit`.
        starting_qubits: A set of the starting qubits for which we construct the RCC circuit.

    Returns:
        A :class:`~qiskit.circuit.QuantumCircuit` representing the RCC.

    """
    qubits_generation = [starting_qubits]
    for _ in range(qaoa.num_layers):
        nodes = set()
        for node in qubits_generation[-1]:
            nodes.add(node)
            for neighbor, _ in qaoa.bqm.iter_neighborhood(node):
                nodes.add(neighbor)

        qubits_generation.append(nodes)
    qubits_generation.reverse()

    qc = QuantumCircuit(0, len(starting_qubits))
    qrs = {}  # quantum registers
    for qubit in qubits_generation[-1]:
        qrs[qubit] = QuantumRegister(1, str(qubit))
        qc.add_register(qrs[qubit])
        qc.h(qrs[qubit])
    for qubit in qubits_generation[0] - qubits_generation[-1]:
        qrs[qubit] = QuantumRegister(1, str(qubit))
        qc.add_register(qrs[qubit])
        qc.h(qrs[qubit])
    for p in range(qaoa.num_layers):
        for qubit in qubits_generation[p]:
            qc.rz(2 * qaoa.angles[2 * p] * qaoa.bqm.get_linear(qubit), qrs[qubit])
        for q1, q2 in qaoa.bqm.quadratic:
            if (q1 in qubits_generation[p] and q2 in qubits_generation[p + 1]) or (
                q2 in qubits_generation[p] and q1 in qubits_generation[p + 1]
            ):
                qc.rzz(2 * qaoa.angles[2 * p] * qaoa.bqm.get_quadratic(q1, q2), qrs[q1], qrs[q2])  # type: ignore[index]
        for qubit in qubits_generation[p + 1]:
            qc.rx(2 * qaoa.angles[2 * p + 1], qrs[qubit])
    return qc


def quimb_tn(qaoa: QUBOQAOA) -> qtn.Circuit:
    """Constructs a :mod:`quimb` tensor network representing the quantum circuit.

    The object is constructed just like any other quantum circuit by applying quantum gates. :mod:`quimb`
    transforms those into tensors.

    Args:
        qaoa: A :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose angles and interaction strengths are used in
            the construction of the tensor network.

    Returns:
        :mod:`quimb` tensor network representing the QAOA circuit (without measurements).

    """
    tn = qtn.Circuit(qaoa.num_qubits)
    for qubit in range(qaoa.num_qubits):
        tn.apply_gate("H", qubit)
    for p in range(qaoa.num_layers):
        for qubit in range(qaoa.num_qubits):
            tn.apply_gate("RZ", 2 * qaoa.angles[2 * p] * qaoa.bqm.get_linear(qubit), qubit)
        for q1, q2 in qaoa.bqm.quadratic:
            tn.apply_gate("RZZ", 2 * qaoa.angles[2 * p] * qaoa.bqm.get_quadratic(q1, q2), q1, q2)
        for qubit in range(qaoa.num_qubits):
            tn.apply_gate("RX", 2 * qaoa.angles[2 * p + 1], qubit)
    return tn


# pylint: disable=too-many-locals
def transpiled_circuit(
    qaoa: QUBOQAOA,
    backend: BackendV2 = AerSimulator(method="statevector"),
    transpiler: str | None = None,
    seed: int = 1337,
) -> QuantumCircuit:
    """The function to return a :class:`~qiskit.circuit.QuantumCircuit` tailored to ``backend``.

    This function has highly varying outputs based on which transpiler is used. If no transpiler is used,
    the perfect :class:`~qiskit.circuit.QuantumCircuit` is returned using :meth:`qiskit_circuit`. Otherwise,
    the QAOA circuit is transpiled using one of the transpilers, respecting the topology of ``backend``.

    Args:
        qaoa: The :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose quantum circuit is constructed.
        backend: A backend that the circuit is to be run on. The connectivity of the backend is required
            for the transpilation.
        transpiler: A string that describes which algorithm should be used for transpilation (if any). Should be one
            of: ``None``, "Default", "HardwiredTranspiler", "SparseTranspiler", "SwapNetwork" or "MinimumVertexCover".
        seed: A seed used for "Default" transpilation. It fixes the circuit produced by the stochastic qiskit
            transpiler. It can be used to ensure reproducibility of a transpilation.

    Returns:
        A quantum circuit transpiled to the topology of ``backend``.

    Raises:
        TypeError: If the ``backend`` is not an IQM backend and a custom ``transpiler`` is selected (i.e., other than
            ``None`` or "Default").
        ValueError: If the provided ``transpiler`` is not one of the allowed transpilers.

    """
    # No transpilation, just the pure QAOA circuit.
    if transpiler is None:
        if backend.coupling_map is not None:
            warnings.warn("The backend has a coupling map, but the circuit is not transpiled to it.")
        return qiskit_circuit(qaoa, measurements=True)

    # Use the default Qiskit transpilation
    if transpiler == "Default":
        starting_circuit = qiskit_circuit(qaoa, measurements=True)
        return transpile(starting_circuit, backend, seed_transpiler=seed)

    if not isinstance(backend, IQMBackendBase):
        raise TypeError("Currently, only IQM backends are supported with transpilation other than 'Default' or `None`.")

    if transpiler == "HardwiredTranspiler":
        # This `qpu` object is just a carrier of the QPU connectivity for `hardwired_router`.
        qpu = CrystalQPUFromBackend(backend)
        routed = hardwired_router(qaoa.bqm, qpu)
        qc_hw = routed.build_qiskit(qaoa.betas, qaoa.gammas)  # type: ignore[arg-type]
        # Default layout method uses the VF2 algorithm to find an exact layout match.
        # An exact layout match is guaranteed to exist, so no further routing is needed.
        qc_hw_transpiled = transpile(
            qc_hw,
            backend=backend,
            layout_method="default",
            routing_method="none",
            optimization_level=3,
            seed_transpiler=seed,
        )
        return qc_hw_transpiled

    if transpiler == "SparseTranspiler":
        # This `qpu` object is just a carrier of the QPU connectivity for `greedy_router`.
        qpu = CrystalQPUFromBackend(backend)
        routed = greedy_router(qaoa.bqm, qpu)
        qc_sparse = routed.build_qiskit(qaoa.betas, qaoa.gammas)  # type: ignore[arg-type]
        # Default layout method uses the VF2 algorithm to find an exact layout match.
        # An exact layout match is guaranteed to exist, so no further routing is needed.
        qc_sparse_transpiled = transpile(
            qc_sparse,
            backend=backend,
            layout_method="default",
            routing_method="none",
            optimization_level=3,
            seed_transpiler=seed,
        )
        return qc_sparse_transpiled

    if transpiler == "SwapNetwork":
        # This `qpu` object is just a carrier of the QPU connectivity for `sn_router`.
        qpu = CrystalQPUFromBackend(backend)
        routed = sn_router(qaoa.bqm, qpu)
        qc_sn = routed.build_qiskit(qaoa.betas, qaoa.gammas)  # type: ignore[arg-type]
        # Default layout method uses the VF2 algorithm to find an exact layout match.
        # An exact layout match is guaranteed to exist, so no further routing is needed.
        qc_sn_transpiled = transpile(
            qc_sn,
            backend=backend,
            layout_method="default",
            routing_method="none",
            optimization_level=3,
            seed_transpiler=seed,
        )
        return qc_sn_transpiled

    if transpiler == "MinimumVertexCover":
        qpu = StarQPU(qaoa.bqm.num_variables)  # type: ignore[assignment]
        routed = star_router(qaoa.bqm, qpu)  # type: ignore[arg-type]

        qc_mvc = routed.build_qiskit(qaoa.betas, qaoa.gammas)  # type: ignore[arg-type]

        handling_of_errors = ExistingMoveHandlingOptions("keep")
        qc_mvc_transpiled = transpile_to_IQM(
            qc_mvc,
            backend=backend,
            perform_move_routing=False,
            existing_moves_handling=handling_of_errors,
            initial_layout=[backend.qubit_name_to_index("COMPR1")] + list(range(qaoa.bqm.num_variables)),
        )

        return qc_mvc_transpiled

    raise ValueError(f"Unknown transpiler provided: {transpiler}")
