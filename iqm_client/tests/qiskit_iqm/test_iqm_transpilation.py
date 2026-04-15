# Copyright 2022-2023 Qiskit on IQM developers
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
"""Testing IQM transpilation."""

import math

from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit
from iqm.qiskit_iqm.iqm_move_layout import generate_initial_layout
from iqm.qiskit_iqm.iqm_transpilation import TOLERANCE, IQMOptimizeSingleQubitGates, optimize_single_qubit_gates
from iqm.qiskit_iqm.move_gate import MoveGate
import numpy as np
from packaging.version import Version
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit import __version__ as qiskit_version
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit_aer import AerSimulator
from qiskit_aer import __version__ as qiskit_aer_version

from iqm.pulse import CircuitOperation

from .conftest import create_metrics_from_dqa, get_mocked_backend

RNG = np.random.default_rng()


@pytest.fixture
def backend(adonis_architecture, request):
    return get_mocked_backend(adonis_architecture, request)


def test_optimize_single_qubit_gates_preserves_unitary():
    """Test that single-qubit gate decomposition preserves the unitary of the circuit."""
    circuit = QuantumCircuit(2, 2)
    circuit.t(0)
    circuit.rx(0.4, 0)
    circuit.cx(0, 1)
    circuit.ry(0.7, 1)
    circuit.h(1)
    circuit.r(0.2, 0.8, 0)
    circuit.h(0)

    transpiled_circuit = transpile(circuit, basis_gates=["r", "cz"])
    with pytest.warns(DeprecationWarning):
        optimized_circuit = optimize_single_qubit_gates(transpiled_circuit, drop_final_rz=False)

    transpiled_circuit.save_unitary()
    optimized_circuit.save_unitary()
    simulator = AerSimulator(method="unitary")
    transpiled_unitary = simulator.run(transpiled_circuit).result().get_unitary(transpiled_circuit)
    optimized_unitary = simulator.run(optimized_circuit).result().get_unitary(optimized_circuit)

    np.testing.assert_almost_equal(transpiled_unitary.data, optimized_unitary.data)


def test_optimize_single_qubit_gates_drops_final_rz():
    """Test that single-qubit gate decomposition drops the final rz gate if requested and there is no measurement."""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.h(1)
    circuit.cz(0, 1)
    circuit.h(1)
    circuit.measure(1, 0)

    transpiled_circuit = transpile(circuit, basis_gates=["r", "cz"])
    with pytest.warns(DeprecationWarning):
        optimized_circuit_dropped_rz = optimize_single_qubit_gates(transpiled_circuit)
    with pytest.warns(DeprecationWarning):
        optimized_circuit = optimize_single_qubit_gates(transpiled_circuit, drop_final_rz=False)

    simulator = AerSimulator(method="statevector")
    shots = 100000

    transpiled_counts = simulator.run(transpiled_circuit, shots=shots).result().get_counts()
    optimized_counts = simulator.run(optimized_circuit, shots=shots).result().get_counts()
    optimized_dropped_rz_counts = simulator.run(optimized_circuit_dropped_rz, shots=shots).result().get_counts()

    for counts in [transpiled_counts, optimized_counts, optimized_dropped_rz_counts]:
        for key in counts:
            # rounding to one decimal to make stochastic failures unlikely
            # TODO should think of a better test
            counts[key] = np.round(counts[key] / shots, 1)

    assert transpiled_counts == optimized_counts == optimized_dropped_rz_counts
    assert len(optimized_circuit_dropped_rz.get_instructions("r")) == 3
    assert len(optimized_circuit.get_instructions("r")) == 5


def test_optimize_single_qubit_gates_reduces_gate_count():
    """Test that single-qubit gate decomposition optimizes the number of single-qubit gates."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    transpiled_circuit = transpile(circuit, basis_gates=["r", "cz"])
    with pytest.warns(DeprecationWarning):
        optimized_circuit = optimize_single_qubit_gates(transpiled_circuit)

    assert len(optimized_circuit.get_instructions("r")) == 3


def test_optimize_single_qubit_gates_raises_on_invalid_basis():
    """Test that optimization pass raises error if gates other than ``RZ`` and ``CZ`` are provided."""
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)

    with pytest.raises(ValueError, match="Invalid operation 'h' found "), pytest.warns(DeprecationWarning):
        optimize_single_qubit_gates(circuit)


@pytest.mark.parametrize("backend", [IQMFakeAdonis(), IQMFakeDeneb(), IQMFakeAphrodite()])
def test_optimize_single_qubit_gates_preserves_layout(backend):
    """Test optimize_single_qubit_gates returns a circuit with a layout if the circuit had a layout."""

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure_all()

    # In case the layout is not set
    with pytest.warns(DeprecationWarning):
        qc_optimized = optimize_single_qubit_gates(transpile(qc, basis_gates=["r", "cz"]))
    assert qc_optimized.layout is None

    # In case the layout is set by the user
    if backend.has_resonators():
        backend.metrics = create_metrics_from_dqa(backend.architecture)
        initial_layout = generate_initial_layout(backend.target_with_resonators, qc).get_physical_bits()
    else:
        initial_layout = {
            physical_qubit: qc.qubits[logical_qubit]
            for logical_qubit, physical_qubit in enumerate(RNG.choice(range(backend.num_qubits), qc.num_qubits, False))
        }
    transpiled_circuit_alt = transpile(qc, backend=backend, initial_layout=initial_layout)

    for physical_qubit, logical_qubit in initial_layout.items():
        assert transpiled_circuit_alt.layout.initial_layout[logical_qubit] == physical_qubit

    # In case the layout is set by the transpiler
    transpiled_circuit = transpile(qc, backend=backend)
    layout = transpiled_circuit.layout
    with pytest.warns(DeprecationWarning):
        qc_optimized = optimize_single_qubit_gates(transpiled_circuit)
    assert layout == qc_optimized.layout


@pytest.mark.parametrize("optimization_level", list(range(4)))
def test_qiskit_native_transpiler(move_architecture, optimization_level, request):
    """Tests that a simple circuit is transpiled correctly using the Qiskit transpiler."""
    backend = get_mocked_backend(move_architecture, request)
    # circuit should contain all our supported operations to make sure the transpiler can handle them
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier(0, 1)
    qc.delay(10, 0, unit="ns")
    qc.reset(0)
    qc.cx(0, 1)
    qc.measure_all()
    transpiled_circuit = transpile(qc, backend=backend, optimization_level=optimization_level)
    validate_circuit(transpiled_circuit, backend)


def test_optimize_single_qubit_gates_works_on_invalid_move_sandwich():
    """Tests that the optimization pass works on a circuit with an invalid MOVE sandwich.
    In case the user is wanting to use the higher energy levels but also optimize the SQGs in the circuit."""
    qc = QuantumCircuit(2)
    qc.rz(0.5, 1)
    qc.append(MoveGate(), [1, 0])
    qc.x(1)
    qc.append(MoveGate(), [1, 0])
    basis_circuit = PassManager([BasisTranslator(SessionEquivalenceLibrary, ["r", "cz", "move"])]).run(qc)
    transpiled_circuit = PassManager(
        [BasisTranslator(SessionEquivalenceLibrary, ["r", "cz", "move"]), IQMOptimizeSingleQubitGates(True, True)]
    ).run(basis_circuit)
    assert transpiled_circuit.count_ops()["r"] == 1
    assert transpiled_circuit.count_ops()["move"] == 2
    for gate in transpiled_circuit:
        if gate.operation.name == "r":
            assert math.isclose(gate.operation.params[0], np.pi, rel_tol=TOLERANCE)
            assert math.isclose(gate.operation.params[1], 0, abs_tol=TOLERANCE)


def test_with_if_test_works_with_transpiler(backend):
    """Tests that a circuit with an if test is transpiled correctly."""
    qc = QuantumCircuit(1, 2)

    # classically controlled gate on different qubit
    qc.measure(0, 0)
    with qc.if_test((0, 1)):
        qc.x(0)
    # final measurement
    qc.measure(0, 1)
    # Without IQM transpiler passes
    transpiled_circuit = transpile(qc, target=backend.target)
    validate_circuit(transpiled_circuit, backend)
    # Without Rz optimization pass
    transpiled_circuit = transpile(qc, backend=backend, optimization_level=0)
    validate_circuit(transpiled_circuit, backend)
    # With Rz optimization
    transpiled_circuit = transpile(qc, backend=backend)
    validate_circuit(transpiled_circuit, backend)


def test_with_if_test_sqg_opt_ghz(backend):
    """Tests that a circuit with an if test is transpiled correctly and optimized for single qubit gates."""
    qc = QuantumCircuit(4, 5)
    # Constant depth 4 Qubit GHZ
    # 2 Bell states
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    # 0000 + 0011 + 1100 + 1111
    qc.cx(1, 2)
    # 0000 + 0011 + 1110 + 1101
    qc.measure(2, 0)
    # Either (m=0) 0000 + 1101 or (m=1) 0011 + 1110
    with qc.if_test((0, 1)):  # (m=1)
        # 0011 + 1110
        qc.x(2)
        qc.x(3)
        # 0000 + 1101
    # Either (m=0) 0000 + 1101 or (m=1) 0000 + 1101
    qc.cx(1, 2)
    # 0000 + 1111
    qc.barrier()
    qc.measure((0, 1, 2, 3), (1, 2, 3, 4))
    _transpile_and_check(qc, backend, {"11110", "00001", "11111", "00000"})


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
def test_c_if_sqg_opt_ghz(backend):
    """Tests that a circuit with an c_if is transpiled correctly and optimized for single qubit gates."""
    qc = QuantumCircuit(4, 5)
    # Constant depth 4 Qubit GHZ
    # 2 Bell states
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    # 0000 + 0011 + 1100 + 1111
    qc.cx(1, 2)
    # 0000 + 0011 + 1110 + 1101
    qc.measure(2, 0)
    # Either (m=0) 0000 + 1101 or (m=1) 0011 + 1110
    qc.x(2).c_if(0, 1)
    # Either (m=0) 0000 + 1101 or (m=1) 0001 + 1100
    qc.x(3).c_if(0, 1)
    # Either (m=0) 0000 + 1101 or (m=1) 0000 + 1101
    qc.cx(1, 2)
    # 0000 + 1111
    qc.barrier()
    qc.measure((0, 1, 2, 3), (1, 2, 3, 4))
    _transpile_and_check(qc, backend, {"11110", "00001", "11111", "00000"})


def test_with_if_test_sqg_opt_z_commutation(backend):
    """Tests that the single qubit gate optimization is applied correctly with if tests."""
    # https://quirk-e.dev/#circuit={%22cols%22:[[%22H%22,%22H%22],[%22Measure%22,%22Z%22],[%22%E2%80%A2%22,%22Z%22],[1,%22H%22]]}
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.z(1)
    qc.measure(0, 0)
    with qc.if_test((0, 1)):
        qc.z(1)
    qc.h(1)
    qc.measure(1, 1)

    alt_phase = 10.995574287564276 if Version(qiskit_version) < Version("1.2.0") else 4.71238898038469
    expected_operations = (
        CircuitOperation(
            name="prx",
            locus=("QB1",),
            args={"angle": 1.5707963267948968, "phase": 4.71238898038469},
            implementation=None,
        ),
        CircuitOperation(
            name="prx",
            locus=("QB2",),
            args={"angle": 1.5707963267948968, "phase": -1.5707963267948966},
            implementation=None,
        ),
        CircuitOperation(
            name="measure", locus=("QB1",), args={"key": "c_2_0_0", "feedback_key": "c_2_0_0"}, implementation=None
        ),
        CircuitOperation(
            name="cc_prx",
            locus=("QB2",),
            args={"angle": -3.141592653589793, "phase": 0.0, "feedback_key": "c_2_0_0", "feedback_qubit": "QB1"},
            implementation=None,
        ),
        CircuitOperation(
            name="cc_prx",
            locus=("QB2",),
            args={
                "angle": 3.141592653589793,
                "phase": 1.5707963267948966,
                "feedback_key": "c_2_0_0",
                "feedback_qubit": "QB1",
            },
            implementation=None,
        ),
        CircuitOperation(
            name="prx",
            locus=("QB2",),
            args={"angle": 1.5707963267948968, "phase": alt_phase},
            implementation=None,
        ),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "c_2_0_1"}, implementation=None),
    )
    _transpile_and_check(
        qc, backend, {"01", "10"}, iqm_json=expected_operations
    )  # , error_msg="The use of an else-block with if_test is not supported.")


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
def test_c_if_sqg_opt_z_commutation(backend):
    """Tests that the single qubit gate optimization is applied correctly with c_if conditionals."""
    # https://quirk-e.dev/#circuit={%22cols%22:[[%22H%22,%22H%22],[%22Measure%22,%22Z%22],[%22%E2%80%A2%22,%22Z%22],[1,%22H%22]]}
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.z(1)
    qc.measure(0, 0)
    qc.z(1).c_if(0, 1)
    qc.h(1)
    qc.measure(1, 1)

    expected_operations = (
        CircuitOperation(
            name="prx",
            locus=("QB1",),
            args={"angle": 1.5707963267948968, "phase": 4.71238898038469},
            implementation=None,
        ),
        CircuitOperation(
            name="prx",
            locus=("QB2",),
            args={"angle": 1.5707963267948968, "phase": -1.5707963267948966},
            implementation=None,
        ),
        CircuitOperation(
            name="measure", locus=("QB1",), args={"key": "c_2_0_0", "feedback_key": "c_2_0_0"}, implementation=None
        ),
        CircuitOperation(
            name="cc_prx",
            locus=("QB2",),
            args={"angle": 3.141592653589793, "phase": 0.0, "feedback_key": "c_2_0_0", "feedback_qubit": "QB1"},
            implementation=None,
        ),
        CircuitOperation(
            name="cc_prx",
            locus=("QB2",),
            args={
                "angle": 3.141592653589793,
                "phase": 1.5707963267948966,
                "feedback_key": "c_2_0_0",
                "feedback_qubit": "QB1",
            },
            implementation=None,
        ),
        CircuitOperation(
            name="prx",
            locus=("QB2",),
            args={"angle": 1.5707963267948968, "phase": 4.71238898038469},
            implementation=None,
        ),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "c_2_0_1"}, implementation=None),
    )
    _transpile_and_check(qc, backend, {"01", "10"}, iqm_json=expected_operations)


def test_with_if_test_sqg_opt_z_commutation_in_else(backend):
    """Tests that the single qubit gate optimization properly moves the Z gate in the else clause
    to the if clause. Since now the else clause is empty, the resulting circuit is valid."""
    # https://quirk-e.dev/#circuit={%22cols%22:[[%22H%22,%22H%22],[%22Measure%22,%22Z%22],[%22%E2%97%A6%22,%22Z%22],[1,%22H%22]]}
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.z(1)
    qc.measure(0, 0)
    with qc.if_test((0, 1)) as else_:
        pass
    with else_:
        qc.z(1)
    qc.h(1)
    qc.measure(1, 1)
    _transpile_and_check(
        qc, backend, {"00", "11"}
    )  # , error_msg="The use of an else-block with if_test is not supported.")


def test_with_if_test_sqg_opt_else_clause(backend):
    """Tests that the single qubit gate optimization works for both if and else clauses.
    Note that hardware does not yet support this, so the resulting circuit is not valid."""
    # https://quirk-e.dev/#circuit={%22cols%22:[[%22H%22,%22H%22],[%22Measure%22,%22Z%22],[%22%E2%80%A2%22,%22Z^-%C2%BD%22],[1,%22X%22],[%22%E2%97%A6%22,%22Z^%C2%BD%22],[1,%22Z^%C2%BD%22],[1,%22H%22]]}
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)

    qc.h(1)
    qc.z(1)
    with qc.if_test((0, 1)) as else_:
        qc.sdg(1)
        qc.x(1)
    with else_:
        qc.x(1)
        qc.s(1)
    qc.s(1)
    qc.h(1)
    qc.measure(1, 1)
    _transpile_and_check(qc, backend, {"00", "01"}, error_msg="The use of an else-block with if_test is not supported.")


def test_with_if_test_sqg_opt_cz_optimization(backend):
    """Tests that the single qubit gate optimization works for both if and else clauses.
    Note that hardware does not yet support this, so the resulting circuit is not valid."""
    # https://quirk-e.dev/#circuit={%22cols%22:[[%22H%22,%22H%22],[%22Measure%22,%22Z%22],[%22%E2%80%A2%22,%22%E2%80%A2%22,%22X%22],[1,%22H%22,%22H%22]]}
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1)
    qc.z(1)
    qc.measure(0, 0)
    with qc.if_test((0, 1)):
        qc.cx(1, 2)
    qc.h(1)
    qc.h(2)
    qc.measure(1, 1)
    qc.measure(2, 2)
    _transpile_and_check(
        qc,
        backend,
        {"010", "011", "101", "110"},
        error_msg="This backend only supports conditionals on r, x, y, rx and ry gates, not on cz",
    )


def _transpile_and_check(qc, backend, expected_bitstrings, error_msg: str | None = None, iqm_json=None):
    transpiled_circuit = transpile(
        qc, backend=backend, scheduling_method="only_rz_optimization", initial_layout=range(qc.num_qubits)
    )
    transp_without_opt = transpile(
        qc, target=backend.target, initial_layout=range(qc.num_qubits)
    )  # Move routing errors out
    # Check that we indeed reduce the gate count
    assert transp_without_opt.count_ops()["r"] >= transpiled_circuit.count_ops()["r"]
    if error_msg is not None:
        with pytest.raises(ValueError, match=error_msg):
            validate_circuit(transpiled_circuit, backend)
    else:
        validate_circuit(transpiled_circuit, backend)
    if qiskit_aer_version >= "0.17.0":  # AerSimulator broken before 0.17.0
        _check_sample_distribution(qc, transpiled_circuit, expected_bitstrings)
    if iqm_json is not None:
        transp_iqm_circuit = backend.serialize_circuit(transpiled_circuit)
        assert transp_iqm_circuit.instructions == iqm_json


def _check_sample_distribution(qc1, qc2, expected_bitstrings):
    # Since Qiskit cannot give the exact probability of measuring each state we check the sample distribution
    # after simulating both circuits.
    sim = AerSimulator()
    sim_before = sim.run(qc1, shots=10000).result().get_counts()
    sim_after = sim.run(qc2, shots=10000).result().get_counts()
    print(qc1)
    print("Transpiled layout:", qc2.layout)
    print(qc2)
    assert sim_before.keys() == sim_after.keys()
    assert sim_before.keys() == expected_bitstrings
    for key in sim_before.keys():
        assert 0.8 <= sim_before[key] / sim_after[key] <= 1.2
