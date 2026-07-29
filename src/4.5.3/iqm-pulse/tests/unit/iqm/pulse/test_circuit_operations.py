#  ********************************************************************************
#
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
import re

import numpy as np
import pytest

from iqm.pulse.builder import CircuitOperation
from iqm.pulse.circuit_operations import Circuit, CircuitOperationList, reorder
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.gates import get_unitary_prx, get_unitary_rz, get_unitary_u
from iqm.pulse.playlist.instructions import Instruction
from iqm.pulse.quantum_ops import QuantumOp

rng = np.random.default_rng()


def test_reorder():
    """Reordering n-qubit operators works as expected."""
    A = rng.standard_normal((2, 2))
    B = rng.standard_normal((2, 2))
    C = rng.standard_normal((2, 2))
    U = np.kron(A, np.kron(B, C))
    assert np.allclose(reorder(U, [0, 1, 2]), U)
    assert np.allclose(reorder(U, [2, 0, 1]), np.kron(B, np.kron(C, A)))
    assert np.allclose(reorder(U, [0, 2, 1]), np.kron(A, np.kron(C, B)))


@pytest.mark.parametrize("perm", [[0, 1, 1], [-1, 0, 1], [1, 0, 3]])
def test_reorder_invalid(perm):
    """Reordering catches invalid permutations."""
    U = np.eye(8)
    with pytest.raises(ValueError, match="Invalid 3-qubit permutation"):
        reorder(U, perm)


class FakeImplementation1(GateImplementation):
    pass


class FakeImplementation2(GateImplementation):
    pass


def get_unitary1(param1, param2):
    return get_unitary_prx(param1, param2)


def get_unitary2(param3):
    return np.diag([1, 1, 1, np.exp(1j * param3)])


op1 = QuantumOp(
    "test_1qb",
    1,
    params={"param1": (float,), "param2": (float,)},
    implementations={
        "f1": FakeImplementation1,
        "f2": FakeImplementation2,
    },
    unitary=get_unitary1,
)
op2 = QuantumOp(
    "test_2qb",
    2,
    params={"param3": (float,)},
    implementations={
        "f1": FakeImplementation1,
        "f2": FakeImplementation2,
    },
    unitary=get_unitary2,
)
custom_table = {op1.name: op1, op2.name: op2}


@pytest.mark.parametrize(
    "x_qubit, cz_qubits, all_qubits",
    [
        (
            "QB1",
            ("QB1", "QB2"),
            ["QB1", "QB2"],
        ),
        (
            "QB2",
            ("QB1", "QB2"),
            ["QB2", "QB1"],
        ),
        (
            "QB1",
            ("QB2", "QB3"),
            ["QB1", "QB2", "QB3"],
        ),
        (
            "Alice",
            ("Bob", "Charlie"),
            ["Alice", "Bob", "Charlie"],
        ),
    ],
)
def test_find_qubits_default(x_qubit, cz_qubits, all_qubits):
    x_gate = CircuitOperation("prx", (x_qubit,), {"angle": np.pi, "phase": 0.0})
    cz_gate = CircuitOperation("cz", cz_qubits)
    circuit = CircuitOperationList([x_gate, cz_gate])
    circuit.find_qubits()

    assert circuit.qubits == all_qubits
    assert "prx" in circuit.table
    assert "cz" in circuit.table


def test_add_op():
    circuit = CircuitOperationList(num_qubits=2)
    circuit.add_op("prx", [0], np.pi, np.pi / 2)
    circuit.add_op(
        "cz",
        [0, 1],
    )
    circuit.add_op("prx", [1], np.pi, np.pi / 2, impl_name="drag_crf_sx")

    assert circuit[0] == CircuitOperation("prx", ("QB1",), {"angle": np.pi, "phase": np.pi / 2})
    assert circuit[1] == CircuitOperation("cz", ("QB1", "QB2"), {})
    assert circuit[2] == CircuitOperation(
        "prx", ("QB2",), {"angle": np.pi, "phase": np.pi / 2}, implementation="drag_crf_sx"
    )

    with pytest.raises(KeyError, match="QuantumOp with name xxx is not in"):
        circuit.add_op("xxx", [0], np.pi, np.pi / 2)

    with pytest.raises(
        IndexError, match="To add new operations in this way, make sure the attribute 'qubits' has enough qubits"
    ):
        circuit.add_op(
            "cz",
            [2, 1],
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Operation u has the following arguments: ('theta', 'phi', 'lam'), but 2 values were provided."
        ),
    ):
        circuit.add_op("u", [0], np.pi, np.pi / 2)


def test_default_shortcuts():
    circuit = CircuitOperationList(num_qubits=2)
    circuit.prx(np.pi, np.pi / 2, 0)

    assert circuit[0] == CircuitOperation("prx", ("QB1",), {"angle": np.pi, "phase": np.pi / 2})

    circuit.u(np.pi, np.pi / 2, np.pi / 4, 1)

    assert circuit[1] == CircuitOperation("u", ("QB2",), {"theta": np.pi, "phi": np.pi / 2, "lam": np.pi / 4})

    circuit.prx(np.pi, np.pi / 2, 0)
    circuit.sx(1)

    assert circuit[3] == CircuitOperation("sx", ("QB2",), {})

    circuit.rz(np.pi / 3, 0)

    assert circuit[4] == CircuitOperation("rz", ("QB1",), {"angle": np.pi / 3})

    circuit.rz_physical(1)

    assert circuit[5] == CircuitOperation("rz_physical", ("QB2",), {})

    circuit.cz(0, 1)

    assert circuit[6] == CircuitOperation("cz", ("QB1", "QB2"), {})

    circuit.cc_prx(np.pi, 0, "QB1", "feedback", 0)

    assert circuit[7] == CircuitOperation(
        "cc_prx", ("QB1",), {"angle": np.pi, "phase": 0, "feedback_qubit": "QB1", "feedback_key": "feedback"}
    )

    circuit.barrier(0)
    circuit.barrier(0, 1)

    assert circuit[8] == CircuitOperation("barrier", ("QB1",), {})
    assert circuit[9] == CircuitOperation("barrier", ("QB1", "QB2"), {})

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The operation prx requires 1 locus/qubit indices and 2 additional arguments. A total of 5 were provided."
        ),
    ):
        circuit.prx(np.pi, np.pi / 2, np.pi / 4, 0, 1)

    circuit_star = CircuitOperationList(qubits=["QB1", "COMP_R", "QB2"])

    circuit_star.move(0, 1)

    assert circuit_star[0] == CircuitOperation("move", ("QB1", "COMP_R"), {})


@pytest.mark.parametrize(
    "euler_angles", [[np.pi, np.pi / 2, np.pi / 4], [np.pi / 2, 0, np.pi], [np.pi / 3, 0, np.pi / 5]]
)
def test_default_unitary(euler_angles):
    circuit = CircuitOperationList(num_qubits=1)
    circuit.u(*euler_angles, 0)
    circuit.prx(np.pi / 2, 0, 0)
    circuit.rz(np.pi / 5, 0)

    assert circuit.get_unitary() == pytest.approx(
        get_unitary_rz(np.pi / 5) @ get_unitary_prx(np.pi / 2, 0) @ get_unitary_u(*euler_angles)
    )


@pytest.mark.parametrize(
    "qubits", [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]
)
def test_default_unitary_mapping(
    qubits,
):
    """Test unitary mapping from k to n > k qubits.

    The purpose of this test is to check if the mapping of the unitary defined for k qubits onto a circuit of n > k
    qubits is correct. This should work in general, but it is unintuitive to check, so in practice we use just the CZ
    gate unitary in this test, which is diagonal, to verify there are no obvious problems. Using just the CZ allows
    a human to understand what is going on and why the unitary is correct or not."""

    circuit = CircuitOperationList(num_qubits=4)

    circuit.cz(*qubits)

    unitary = circuit.get_unitary()

    # construct a CZ unitary diagonal. The diagonal is just 1s and -1s, with -1s corresponding to states where both
    # participating qubits are excited, so the index bitstring has 1s on those positions. There should always be 12
    # 1s and 4 -1s in this particular case of 4 qubits.
    diagonal = np.ones(16)
    for i in range(16):
        bitstring = np.binary_repr(i, width=4)
        if bitstring[qubits[0]] == "1" and bitstring[qubits[1]] == "1":
            diagonal[i] = -1.0

    assert unitary.shape == (16, 16)
    assert np.all(unitary.T.conj() @ unitary == np.eye(16))
    assert np.all(np.diagonal(unitary) == np.array(diagonal))


def test_default_compose():
    circ1 = CircuitOperationList(qubits=["QB1", "QB2"])
    circ2 = CircuitOperationList(qubits=["QB3", "QB4"])
    circ3 = CircuitOperationList(qubits=["QB5"])

    circ1.prx(np.pi / 2, 0, 0)
    circ1.prx(np.pi / 2, 0, 1)
    circ1.cz(0, 1)

    unitary1 = circ1.get_unitary()

    circ2.u(np.pi / 2, 0, np.pi, 0)
    circ2.cz(0, 1)
    circ2.u(np.pi / 3, 0, np.pi / 2, 1)

    unitary2 = circ2.get_unitary()

    circ3.u(np.pi / 5, np.pi / 5, 0, 0)

    circ1.compose(circ2, [0, 1])

    assert circ1.get_unitary() == pytest.approx(unitary2 @ unitary1)

    assert circ1[-2] == CircuitOperation("cz", ("QB1", "QB2"), {})
    assert circ1[-1] == CircuitOperation("u", ("QB2",), {"theta": np.pi / 3, "phi": 0, "lam": np.pi / 2})

    circ1.compose(circ3, [1, 0])

    assert circ1[-1] == CircuitOperation("u", ("QB2",), {"theta": np.pi / 5, "phi": np.pi / 5, "lam": 0})


@pytest.mark.parametrize("repeats", [1, 3, 6])
def test_default_count_ops(repeats):
    circuit = CircuitOperationList(num_qubits=2)

    for _ in range(repeats):
        circuit.u(np.pi / 2, np.pi / 2, np.pi / 4, 1)
        circuit.u(np.pi / 2, np.pi / 2, np.pi, 0)
        circuit.cz(0, 1)
        circuit.prx(np.pi / 2, np.pi / 2, 1)
        circuit.prx(np.pi, -np.pi / 2, 0)
        circuit.cz(1, 0)
        circuit.rz(np.pi / 4, 0)

    counted = circuit.count_ops()

    assert len(counted) == 4
    assert counted["u"] == 2 * repeats
    assert counted["cz"] == 2 * repeats
    assert counted["prx"] == 2 * repeats
    assert counted["rz"] == repeats


def test_default_map_locus():
    circ1 = CircuitOperationList(qubits=["QB1", "QB2"])

    circ1.prx(np.pi / 2, 0, 0)
    circ1.prx(np.pi / 2, 0, 1)
    circ1.cz(0, 1)
    circ1.cc_prx(np.pi, 0, "QB1", "feedback", 0)

    new = circ1.map_loci(["QB3", "QB4"])

    assert new[0].locus == ("QB3",)
    assert new[1].locus == ("QB4",)
    assert new[2].locus == ("QB3", "QB4")
    assert new[3].locus == ("QB3",)
    assert new[3].args["feedback_qubit"] == "QB3"

    new2 = circ1.map_loci(["QB4", "QB3"])

    assert new2[0].locus == ("QB4",)
    assert new2[1].locus == ("QB3",)
    assert new2[2].locus == ("QB4", "QB3")
    assert new2[3].locus == ("QB4",)
    assert new2[3].args["feedback_qubit"] == "QB4"

    # the object is not mutated
    assert circ1[0].locus == ("QB1",)

    with pytest.raises(ValueError, match="Repeated locus elements"):
        new3 = circ1.map_loci(["QB3", "QB3"])

    with pytest.raises(IndexError, match="new locus must be equal to the number"):
        new3 = circ1.map_loci(["QB3"])  # noqa: F841


def test_custom_table():
    circuit = CircuitOperationList(num_qubits=2, table=custom_table)

    circuit.test_1qb(0.3, 0.4, 0)
    circuit.test_1qb(0.5, -0.9, 1)
    circuit.test_2qb(np.pi / 4, 0, 1)

    circuit.test_1qb(-0.4, 0.5, 0, impl_name="f2")
    circuit.test_1qb(-0.4, 0.5, 1, impl_name="f1")

    unitary = circuit.get_unitary()

    assert circuit[0] == CircuitOperation("test_1qb", ("QB1",), {"param1": 0.3, "param2": 0.4})
    assert circuit[1] == CircuitOperation("test_1qb", ("QB2",), {"param1": 0.5, "param2": -0.9})
    assert circuit[2] == CircuitOperation("test_2qb", ("QB1", "QB2"), {"param3": np.pi / 4})
    assert circuit[3] == CircuitOperation("test_1qb", ("QB1",), {"param1": -0.4, "param2": 0.5}, implementation="f2")
    assert circuit[4] == CircuitOperation("test_1qb", ("QB2",), {"param1": -0.4, "param2": 0.5}, implementation="f1")
    assert unitary.shape == (4, 4)

    with pytest.raises(KeyError, match="QuantumOp with name prx is not"):
        circuit.add_op("prx", [0], np.pi, 0)


def test_circuit_all_locus_components():
    circuit = Circuit(
        name="test",
        instructions=(
            CircuitOperation(
                name="prx",
                locus=("QB1",),
                args={"angle": 0, "phase": 0},
            ),
            CircuitOperation(
                name="prx",
                locus=("QB2",),
                args={"angle": 0, "phase": 0},
            ),
            CircuitOperation(name="move", locus=("QB2", "CR1")),
        ),
    )
    assert circuit.all_locus_components() == {"QB1", "QB2", "CR1"}


@pytest.mark.parametrize(
    "circuit, expected_message",
    [
        (Circuit(name="test", instructions=(CircuitOperation(name="cz", locus=("QB1", "QB2")),)), None),
        (
            Circuit(name="", instructions=(CircuitOperation(name="cz", locus=("QB1", "QB2")),)),
            "circuit should have a non-empty string for a name",
        ),
        (Circuit(name="test", instructions=()), "circuit should have at least one instruction"),
        (
            Circuit(name="test", instructions=(CircuitOperation(name="xyz", locus=("QB1", "QB2")),)),
            "Unknown operation 'xyz'",
        ),
        (
            Circuit(name="test", instructions=(Instruction(duration=1),)),
            "Every instruction in a circuit should be of type <CircuitOperation>",
        ),
    ],
)
def test_circuit_validate(circuit, expected_message, schedule_builder):
    if expected_message is None:
        assert circuit.validate(schedule_builder.op_table) is None  # expect validation to pass
    else:
        with pytest.raises(ValueError, match=expected_message):
            circuit.validate(schedule_builder.op_table)
