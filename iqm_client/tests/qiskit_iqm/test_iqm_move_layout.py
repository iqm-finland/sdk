# Copyright 2025 Qiskit on IQM developers
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

from uuid import UUID

from iqm.iqm_client import ExistingMoveHandlingOptions
from iqm.qiskit_iqm import IQMCircuit, transpile_to_IQM
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_move_layout import generate_initial_layout
from iqm.qiskit_iqm.move_gate import MoveGate
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import TranspilerError

from iqm.station_control.interface.models import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo

from .conftest import create_metrics_from_dqa


class MockBackend(IQMBackendBase):
    """Mock backend for layout generation."""

    @classmethod
    def _default_options(cls):
        return None

    @property
    def max_circuits(self) -> int:
        return 1

    def run(self):
        return None


arch_one_move = DynamicQuantumArchitecture(
    calibration_set_id=UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb"),
    qubits=["QB1", "QB2", "QB3", "QB4"],
    computational_resonators=["CR1"],
    gates={
        "prx": GateInfo(
            implementations={
                "drag_gaussian": GateImplementationInfo(
                    loci=(
                        ("QB1",),
                        ("QB2",),
                        ("QB3",),
                        ("QB4",),
                    )
                ),
            },
            default_implementation="drag_gaussian",
            override_default_implementation={},
        ),
        "cz": GateInfo(
            implementations={
                "tgss": GateImplementationInfo(
                    loci=(
                        ("QB1", "CR1"),
                        ("QB2", "CR1"),
                        ("QB3", "CR1"),
                        ("QB4", "CR1"),
                    )
                ),
            },
            default_implementation="tgss",
            override_default_implementation={},
        ),
        "move": GateInfo(
            implementations={
                "tgss_crf": GateImplementationInfo(loci=(("QB4", "CR1"),)),
            },
            default_implementation="tgss_crf",
            override_default_implementation={},
        ),
        "measure": GateInfo(
            implementations={
                "constant": GateImplementationInfo(
                    loci=(
                        ("QB1",),
                        ("QB2",),
                        ("QB3",),
                        ("QB4",),
                    )
                ),
            },
            default_implementation="constant",
            override_default_implementation={},
        ),
    },
)

arch_three_moves = DynamicQuantumArchitecture(
    calibration_set_id=UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb"),
    qubits=["QB1", "QB2", "QB3", "QB4"],
    computational_resonators=["CR1"],
    gates={
        "prx": GateInfo(
            implementations={
                "drag_gaussian": GateImplementationInfo(
                    loci=(
                        ("QB1",),
                        ("QB2",),
                        ("QB3",),
                        ("QB4",),
                    )
                ),
            },
            default_implementation="drag_gaussian",
            override_default_implementation={},
        ),
        "cz": GateInfo(
            implementations={
                "tgss": GateImplementationInfo(
                    loci=(
                        ("QB1", "CR1"),
                        ("QB2", "CR1"),
                        ("QB3", "CR1"),
                        ("QB4", "CR1"),
                    )
                ),
            },
            default_implementation="tgss",
            override_default_implementation={},
        ),
        "move": GateInfo(
            implementations={
                "tgss_crf": GateImplementationInfo(
                    loci=(
                        ("QB2", "CR1"),
                        ("QB3", "CR1"),
                        ("QB4", "CR1"),
                    )
                ),
            },
            default_implementation="tgss_crf",
            override_default_implementation={},
        ),
        "measure": GateInfo(
            implementations={
                "constant": GateImplementationInfo(
                    loci=(
                        ("QB1",),
                        ("QB2",),
                        ("QB3",),
                        ("QB4",),
                    )
                ),
            },
            default_implementation="constant",
            override_default_implementation={},
        ),
    },
)

broken_arch = DynamicQuantumArchitecture(
    calibration_set_id=UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb"),
    qubits=["QB1", "QB2"],
    computational_resonators=["QB1"],
    gates={
        "prx": GateInfo(
            implementations={
                "drag_gaussian": GateImplementationInfo(
                    loci=(
                        ("QB1",),
                        ("QB2",),
                    )
                ),
            },
            default_implementation="drag_gaussian",
            override_default_implementation={},
        ),
        "cz": GateInfo(
            implementations={
                "tgss": GateImplementationInfo(loci=(("QB2", "QB1"),)),
            },
            default_implementation="tgss",
            override_default_implementation={},
        ),
        "move": GateInfo(
            implementations={
                "tgss_crf": GateImplementationInfo(loci=(("QB2", "QB1"),)),
            },
            default_implementation="tgss_crf",
            override_default_implementation={},
        ),
        "measure": GateInfo(
            implementations={
                "constant": GateImplementationInfo(
                    loci=(
                        ("QB1",),
                        ("QB2",),
                    )
                ),
            },
            default_implementation="constant",
            override_default_implementation={},
        ),
    },
)


def test_generate_initial_layout_one_move():
    """Initial layout generation works on a circuit that requires one MOVE locus."""

    backend = MockBackend(arch_one_move, metrics=create_metrics_from_dqa(arch_one_move))

    n = 3
    # GHZ-n circuit with n qubits, 1 resonator
    # qubit 0 moved to resonator, CZs from all other qubits to resonator, move back
    qc = IQMCircuit(n + 1, n)
    qc.h(0)
    qc.move(0, n)
    for k in range(1, n):
        qc.h(k)
        qc.cz(k, n)
        qc.h(k)
    qc.move(0, n)
    for k in range(n):
        qc.measure(k, k)

    initial_layout = generate_initial_layout(backend.target_with_resonators, qc)
    # logical/virtual to physical
    layout = initial_layout.get_virtual_bits()

    qreg = qc.qregs[0]
    # resonator
    assert layout[qreg[n]] == 4
    # move qubit(s)
    assert layout[qreg[0]] == 3  # QB4 is the only move qubit


def test_generate_initial_layout_many_moves_fails():
    """Initial layout generation fails on a circuit that requires three MOVE loci."""

    backend = MockBackend(
        arch_one_move,
        metrics=create_metrics_from_dqa(arch_one_move),
    )

    n = 4
    # GHZ-n circuit with n qubits, 1 resonator
    # qubits 1..n-1 moved to resonator, CZ from 0 to resonator, move back
    qc = IQMCircuit(n + 1, n)
    qc.h(0)
    for k in range(1, n):
        qc.h(k)
        qc.move(k, n)
        qc.cz(0, n)
        qc.move(k, n)
        qc.h(k)
    for k in range(n):
        qc.measure(k, k)

    with pytest.raises(TranspilerError, match="Cannot find a physical qubit to map logical qubit 2 to"):
        generate_initial_layout(backend.target_with_resonators, qc)


def test_generate_initial_layout_many_moves():
    """Initial layout generation works on a circuit that requires three MOVE loci."""

    backend = MockBackend(
        arch_three_moves,
        metrics=create_metrics_from_dqa(arch_three_moves),
    )

    n = 4
    # GHZ-n circuit with n qubits, 1 resonator
    # qubits 1..n-1 moved to resonator, CZ from 0 to resonator, move back
    qc = IQMCircuit(n + 1, n)
    qc.h(0)
    for k in range(1, n):
        qc.h(k)
        qc.move(k, n)
        qc.cz(0, n)
        qc.move(k, n)
        qc.h(k)
    for k in range(n):
        qc.measure(k, k)

    initial_layout = generate_initial_layout(backend.target_with_resonators, qc)
    # logical/virtual to physical
    layout = initial_layout.get_virtual_bits()
    qreg = qc.qregs[0]
    # resonator
    assert layout[qreg[n]] == 4
    # move qubit(s)
    for k in range(1, n):
        assert layout[qreg[k]] in (1, 2, 3)  # QB2-4 have moves


def test_move_layout_support_arbitrary_gates():
    backend = MockBackend(
        arch_three_moves,
        metrics=create_metrics_from_dqa(arch_three_moves),
    )

    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.append(MoveGate(), [0, 1])
    qc.cx(1, 2)
    qc.append(MoveGate(), [0, 1])
    qc.measure([0, 2], [0, 1])

    transpile_to_IQM(qc, backend, existing_moves_handling=ExistingMoveHandlingOptions.KEEP)


def test_generate_conflicting_requirements():
    backend = MockBackend(
        broken_arch,
        metrics=create_metrics_from_dqa(broken_arch),
    )

    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.append(MoveGate(), [0, 1])
    qc.cx(1, 2)
    qc.append(MoveGate(), [0, 1])
    qc.measure([0, 2], [0, 1])

    error_message = (  # NOTE qubits are printed differently in different qiskit versions
        ".Virtual/logical qubit .* for the .*'move.*' operation must be a "
        "resonator, but it is already required to be a qubit.."
    )  # Formatting of the special characters also differs between qiskit versions
    # i.e. " vs ' around the string and
    # escaping of the slashes once vs twice for the quotation around the move
    with pytest.raises(TranspilerError, match=error_message):
        transpile_to_IQM(qc, backend=backend, existing_moves_handling=ExistingMoveHandlingOptions.KEEP)
