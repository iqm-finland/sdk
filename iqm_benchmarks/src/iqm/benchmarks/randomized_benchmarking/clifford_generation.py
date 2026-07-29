"""Generate elements of 1-qubit and 2-qubit Clifford groups as QuantumCircuits."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford


def _generate_basic_gates() -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Generate basic gate building blocks."""
    c0 = QuantumCircuit(1)
    half = np.pi / 2

    x2 = c0.copy()
    x2.r(half, 0, 0)
    x2.to_gate()
    y2 = c0.copy()
    y2.r(half, half, 0)
    y2.to_gate()

    x2m = c0.copy()
    x2m.r(-half, 0, 0)
    x2m.to_gate()

    y2m = c0.copy()
    y2m.r(-half, half, 0)
    y2m.to_gate()

    return c0, x2, y2, x2m, y2m


def _generate_pauli_gates(c: list[QuantumCircuit], half: float) -> None:
    """Generate Pauli gates (I, X, Y, YX)."""
    c[0].name = "I"
    c[0].r(0, 0, 0)
    c[1].name = "X"
    c[1].r(np.pi, 0, 0)
    c[2].name = "Y"
    c[2].r(np.pi, half, 0)
    c[3].name = "Y, X"
    c[3].compose(c[1], inplace=True)
    c[3].compose(c[2], inplace=True)


def _generate_2pi3_rotations(
    c: list[QuantumCircuit], x2: QuantumCircuit, y2: QuantumCircuit, x2m: QuantumCircuit, y2m: QuantumCircuit
) -> None:
    """Generate 2π/3 rotation gates."""
    c[4].name = "X/2, Y/2"
    c[4].compose(y2, inplace=True)
    c[4].compose(x2, inplace=True)
    c[5].name = "X/2, -Y/2"
    c[5].compose(y2m, inplace=True)
    c[5].compose(x2, inplace=True)
    c[6].name = "-X/2, Y/2"
    c[6].compose(y2, inplace=True)
    c[6].compose(x2m, inplace=True)
    c[7].name = "-X/2, -Y/2"
    c[7].compose(y2m, inplace=True)
    c[7].compose(x2m, inplace=True)
    c[8].name = "Y/2, X/2"
    c[8].compose(x2, inplace=True)
    c[8].compose(y2, inplace=True)
    c[9].name = "Y/2, -X/2"
    c[9].compose(x2m, inplace=True)
    c[9].compose(y2, inplace=True)
    c[10].name = "-Y/2, X/2"
    c[10].compose(x2, inplace=True)
    c[10].compose(y2m, inplace=True)
    c[11].name = "-Y/2, -X/2"
    c[11].compose(x2m, inplace=True)
    c[11].compose(y2m, inplace=True)


def _generate_pi2_rotations(
    c: list[QuantumCircuit], x2: QuantumCircuit, y2: QuantumCircuit, x2m: QuantumCircuit, y2m: QuantumCircuit
) -> None:
    """Generate π/2 rotation gates."""
    c[12].name = "X/2"
    c[12].compose(x2, inplace=True)
    c[13].name = "-X/2"
    c[13].compose(x2m, inplace=True)
    c[14].name = "Y/2"
    c[14].compose(y2, inplace=True)
    c[15].name = "-Y/2"
    c[15].compose(y2m, inplace=True)
    c[16].name = "-X/2, Y/2, X/2"
    c[16].compose(x2, inplace=True)
    c[16].compose(y2, inplace=True)
    c[16].compose(x2m, inplace=True)
    c[17].name = "-X/2, -Y/2, X/2"
    c[17].compose(x2, inplace=True)
    c[17].compose(y2m, inplace=True)
    c[17].compose(x2m, inplace=True)


def _generate_hadamard_like_gates(
    c: list[QuantumCircuit], x2: QuantumCircuit, y2: QuantumCircuit, x2m: QuantumCircuit, y2m: QuantumCircuit
) -> None:
    """Generate Hadamard-like gates."""
    c[18].name = "X, Y/2"
    c[18].compose(y2, inplace=True)
    c[18].compose(c[1], inplace=True)
    c[19].name = "X, -Y/2"
    c[19].compose(y2m, inplace=True)
    c[19].compose(c[1], inplace=True)
    c[20].name = "Y, X/2"
    c[20].compose(x2, inplace=True)
    c[20].compose(c[2], inplace=True)
    c[21].name = "Y, -X/2"
    c[21].compose(x2m, inplace=True)
    c[21].compose(c[2], inplace=True)
    c[22].name = "X/2, Y/2, X/2"
    c[22].compose(x2, inplace=True)
    c[22].compose(y2, inplace=True)
    c[22].compose(x2, inplace=True)
    c[23].name = "-X/2, Y/2, -X/2"
    c[23].compose(x2m, inplace=True)
    c[23].compose(y2, inplace=True)
    c[23].compose(x2m, inplace=True)


def _generate_1q_clifford_gates(
    c0: QuantumCircuit, x2: QuantumCircuit, y2: QuantumCircuit, x2m: QuantumCircuit, y2m: QuantumCircuit
) -> dict[str, QuantumCircuit]:
    """Generate the 24 single-qubit Clifford gates."""
    half = np.pi / 2
    c = [c0.copy() for _ in range(24)]

    _generate_pauli_gates(c, half)
    _generate_2pi3_rotations(c, x2, y2, x2m, y2m)
    _generate_pi2_rotations(c, x2, y2, x2m, y2m)
    _generate_hadamard_like_gates(c, x2, y2, x2m, y2m)

    return {str(Clifford(c[i]).to_labels(mode="B")): c[i] for i in range(24)}


def _generate_2q_class1(clifford_sqg: dict[str, QuantumCircuit]) -> dict[str, QuantumCircuit]:
    """Generate Class 1: products of Cliffords."""
    clifford_2qg = {}
    c2q_0 = QuantumCircuit(2)
    c2q = [c2q_0.copy() for _ in range(24**2)]
    counter = 0

    for c1 in clifford_sqg.values():
        for c2 in clifford_sqg.values():
            c2q[counter].compose(c1, [0], inplace=True)
            c2q[counter].compose(c2, [1], inplace=True)
            label = str(Clifford(c2q[counter]).to_labels(mode="B"))
            c2q[counter].name = f"sqg_class_{label}"
            clifford_2qg[label] = c2q[counter]
            counter += 1

    return clifford_2qg


def _generate_2q_class2(clifford_sqg: dict[str, QuantumCircuit], s1: list, s1y2: list) -> dict[str, QuantumCircuit]:
    """Generate Class 2: CNOT-like."""
    clifford_2qg = {}
    c2q_0 = QuantumCircuit(2)
    c2q = [c2q_0.copy() for _ in range((24**2) * (3**2))]
    counter = 0

    for c1 in clifford_sqg.values():
        for c2 in clifford_sqg.values():
            for s1_ in s1:
                for sy2 in s1y2:
                    c2q[counter].compose(c1, [0], inplace=True)
                    c2q[counter].compose(c2, [1], inplace=True)
                    c2q[counter].compose(CZGate(), [0, 1], inplace=True)
                    c2q[counter].compose(s1_, [0], inplace=True)
                    c2q[counter].compose(sy2, [1], inplace=True)
                    label = str(Clifford(c2q[counter]).to_labels(mode="B"))
                    c2q[counter].name = f"cnot_class_{label}"
                    clifford_2qg[label] = c2q[counter]
                    counter += 1

    return clifford_2qg


def _generate_2q_class3(
    clifford_sqg: dict[str, QuantumCircuit], s1y2: list, s1x2: list, y2: QuantumCircuit, x2m: QuantumCircuit
) -> dict[str, QuantumCircuit]:
    """Generate Class 3: iSWAP-like."""
    clifford_2qg = {}
    c2q_0 = QuantumCircuit(2)
    c2q = [c2q_0.copy() for _ in range((24**2) * (3**2))]
    counter = 0

    for c1 in clifford_sqg.values():
        for c2 in clifford_sqg.values():
            for sy1 in s1y2:
                for sx2 in s1x2:
                    c2q[counter].compose(c1, [0], inplace=True)
                    c2q[counter].compose(c2, [1], inplace=True)
                    c2q[counter].compose(CZGate(), [0, 1], inplace=True)
                    c2q[counter].compose(y2, [0], inplace=True)
                    c2q[counter].compose(x2m, [1], inplace=True)
                    c2q[counter].compose(CZGate(), [0, 1], inplace=True)
                    c2q[counter].compose(sy1, [0], inplace=True)
                    c2q[counter].compose(sx2, [1], inplace=True)
                    label = str(Clifford(c2q[counter]).to_labels(mode="B"))
                    c2q[counter].name = f"i_swap_class_{label}"
                    clifford_2qg[label] = c2q[counter]
                    counter += 1

    return clifford_2qg


def _generate_2q_class4(
    clifford_sqg: dict[str, QuantumCircuit], y2: QuantumCircuit, y2m: QuantumCircuit
) -> dict[str, QuantumCircuit]:
    """Generate Class 4: SWAP-like."""
    clifford_2qg = {}
    c2q_0 = QuantumCircuit(2)
    c2q = [c2q_0.copy() for _ in range(24**2)]
    counter = 0

    for c1 in clifford_sqg.values():
        for c2 in clifford_sqg.values():
            c2q[counter].compose(c1, [0], inplace=True)
            c2q[counter].compose(c2, [1], inplace=True)
            c2q[counter].compose(CZGate(), [0, 1], inplace=True)
            c2q[counter].compose(y2m, [0], inplace=True)
            c2q[counter].compose(y2, [1], inplace=True)
            c2q[counter].compose(CZGate(), [0, 1], inplace=True)
            c2q[counter].compose(y2, [0], inplace=True)
            c2q[counter].compose(y2m, [1], inplace=True)
            c2q[counter].compose(CZGate(), [0, 1], inplace=True)
            c2q[counter].compose(y2, [1], inplace=True)
            label = str(Clifford(c2q[counter]).to_labels(mode="B"))
            c2q[counter].name = f"swap_class_{label}"
            clifford_2qg[label] = c2q[counter]
            counter += 1

    return clifford_2qg


def generate_clifford_groups() -> tuple[dict[str, QuantumCircuit], dict[str, QuantumCircuit]]:
    """Generate the 1-qubit and 2-qubit Clifford groups.

    Returns:
        A tuple containing:
            - Dictionary of 24 single-qubit Clifford gates
            - Dictionary of 11,520 two-qubit Clifford gates

    """
    # Generate basic building blocks
    c0, x2, y2, x2m, y2m = _generate_basic_gates()

    # Generate 1Q Clifford group
    clifford_1qg = _generate_1q_clifford_gates(c0, x2, y2, x2m, y2m)

    # Get named gates for 2Q generation
    clifford_sqg = {}
    for _label, circuit in clifford_1qg.items():
        clifford_sqg[circuit.name] = circuit

    # Prepare gate sets for 2Q generation
    s1 = [clifford_sqg[k] for k in ["I", "Y/2, X/2", "-X/2, -Y/2"]]
    s1x2 = [clifford_sqg[k] for k in ["X/2", "X/2, Y/2, X/2", "-Y/2"]]
    s1y2 = [clifford_sqg[k] for k in ["Y/2", "Y, X/2", "-X/2, -Y/2, X/2"]]

    # Generate 2Q Clifford group by class
    clifford_2qg = {}
    clifford_2qg.update(_generate_2q_class1(clifford_sqg))
    clifford_2qg.update(_generate_2q_class2(clifford_sqg, s1, s1y2))
    clifford_2qg.update(_generate_2q_class3(clifford_sqg, s1y2, s1x2, y2, x2m))
    clifford_2qg.update(_generate_2q_class4(clifford_sqg, y2, y2m))

    return clifford_1qg, clifford_2qg
