# Copyright 2024 Qiskit on IQM developers
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
"""Testing extended quantum architecture specification."""

from iqm.qiskit_iqm.iqm_target import IQMTarget
import pytest

from .conftest import get_mocked_backend

QISKIT_TO_IQM = {
    "r": "prx",
    "cz": "cz",
    "move": "move",
    "measure": "measure",
    "reset": "cc_prx",  # TODO "reset": "reset",
    "delay": None,
    "id": None,
    "if_else": "cc_prx",
}


@pytest.fixture
def dqa(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def sample_target_move_architecture(move_architecture, qb_to_idx_move_architecture):
    """Returns instance of IQM Target corresponding to dynamic quantum architecture that contains a MOVE gate."""
    return IQMTarget(
        architecture=move_architecture,
        component_to_idx=qb_to_idx_move_architecture,
        include_resonators=True,
        include_fictional_czs=True,
    )


@pytest.mark.parametrize(
    "dqa",
    [
        "move_architecture",
        "adonis_architecture",
        "hypothetical_fake_architecture",
        "star_architecture",
        "linear_3q_architecture",
    ],
    indirect=True,
)
class TestIQMTargetReflectsDQA:
    """Test that the IQM backend reflects the extended architecture."""

    @pytest.fixture(autouse=True)
    def init_backend(self, dqa, request):
        """Initialize the backend with the given architecture."""
        self.dqa = dqa
        # initialize backend without metrics because the testing scope is the DQA
        # not the metrics, in this case
        self.backend = get_mocked_backend(dqa, request, use_metrics=False)
        self.target = self.backend._target

    def test_target_init(self):
        assert self.target is not None
        if self.dqa.computational_resonators:
            assert "move" in self.target.iqm_dqa.gates

    def test_backend_size(self):
        assert self.backend.num_qubits == len(self.dqa.qubits)
        if self.backend._target_with_resonators is not None:
            assert self.backend._target_with_resonators.num_qubits == len(self.dqa.components)

    def test_physical_qubits(self):
        """Check that the physical qubits are in the correct order: resonators at the end."""
        assert self.backend.physical_qubits == self.dqa.qubits + self.dqa.computational_resonators

    def test_target_gate_set(self):
        """Check that gate set of the target is the same as we support according to the DQA."""

        # simplified architecture has no MOVE gate
        target_gates = set(self.backend.target.operation_names)
        dqa_gates = set(self.dqa.gates)
        dqa_gates.discard("move")
        assert dqa_gates == set(dqa_name for name in target_gates if (dqa_name := QISKIT_TO_IQM[name]) is not None)

        if self.backend._target_with_resonators is not None:
            target_gates = set(self.backend._target_with_resonators.operation_names)
            dqa_gates = set(self.dqa.gates)
            assert dqa_gates == set(dqa_name for name in target_gates if (dqa_name := QISKIT_TO_IQM[name]) is not None)

    @pytest.mark.parametrize(("qiskit_name", "iqm_name"), zip(["r", "measure", "reset"], ["prx", "measure", "cc_prx"]))
    def test_1_to_1_corresponding_gates(self, qiskit_name, iqm_name):
        """Check that the gates are defined for the correct qubits where the gates correspond 1-1 directly."""
        self.check_instruction(
            qiskit_name,
            iqm_name=iqm_name,
        )
        if self.backend._target_with_resonators is not None:
            self.check_instruction(qiskit_name, iqm_name=iqm_name, target=self.backend._target_with_resonators)

    def test_id_gates(self):
        """Check that the id gates are defined for both qubits and components."""
        self.check_instruction("id", expected_loci=[(q,) for q in self.dqa.qubits])
        if self.backend._target_with_resonators is not None:
            self.check_instruction(
                "id", expected_loci=[(q,) for q in self.dqa.components], target=self.backend._target_with_resonators
            )

    def test_cz_gates(self):
        """Check that the cz gates are defined for the correct qubits."""
        if "move" not in self.dqa.gates:
            self.check_instruction("cz", iqm_name="cz")
        else:
            self.validate_move_loci()
            self.validate_fake_cz_loci()

    def validate_move_loci(self):
        """Confirms that the move gate is not in the target."""
        assert "move" not in self.backend.target.operation_names

    def validate_fake_cz_loci(self):
        """
        Check that the virtual czs in the target are as expected.
        A "virtual" or "fake" CZ gate is one that is possible not due to a direct
        physical connection, but by using a MOVE gate to bring one qubit
        to a resonator that is connected to the other qubit.
        """
        # Get all CZ gate locations (qubit pairs) advertised by the target that includes virtual gates.
        # The qubit indices are converted back to their string names for easier comparison.
        target_loci = [
            tuple(self.backend.index_to_qubit_name(qb) for qb in loci)
            for i, loci in self.backend._target_with_resonators.instructions
            if i.name == "cz"
        ]
        # Get the list of CZ gates that are physically implemented on the hardware.
        real_cz_loci = list(
            self.dqa.gates["cz"].implementations[self.dqa.gates["cz"].default_implementation].loci,
        )
        # Get the list of all possible MOVE operations, which are pairs of (qubit, resonator).
        real_move_loci = list(
            self.dqa.gates["move"].implementations[self.dqa.gates["move"].default_implementation].loci,
        )

        # Iterate through all CZ gates reported by the target.
        for loci in target_loci:
            # If a reported CZ gate is not in the list of real physical gates, it must be a virtual one.
            # We must then verify that this virtual gate is valid.
            if loci not in real_cz_loci:
                sandwich_found = False
                # Check all possible MOVE operations to see if one can facilitate this virtual CZ.
                for move_qb, res in real_move_loci:
                    # Check for a "CZ sandwich": can the first qubit (loci[0]) move to a resonator (`res`)
                    # that has a real CZ connection to the second qubit (loci[1])?
                    # The check for the real CZ must consider both directions, e.g., (QB2, CR1) and (CR1, QB2).
                    if move_qb == loci[0] and ((loci[1], res) in real_cz_loci or (res, loci[1]) in real_cz_loci):
                        sandwich_found = True
                        break  # Found a valid configuration, no need to check further for this virtual gate.

                    # Symmetrically, check if the second qubit (loci[1]) can move to a resonator
                    # that has a real CZ connection to the first qubit (loci[0]).
                    if move_qb == loci[1] and ((loci[0], res) in real_cz_loci or (res, loci[0]) in real_cz_loci):
                        sandwich_found = True
                        break  # Found a valid configuration.

                # If after checking all MOVE operations we could not find a way to
                # construct this virtual CZ gate, the target is misconfigured.
                assert sandwich_found, f"Virtual CZ gate {loci} is not supported by any MOVE operation."

    def test_target_with_resonators(self):
        """Check that the fake target is correctly generated."""
        if "move" in self.dqa.gates:
            self.validate_move_loci_fake_target()
            self.validate_cz_loci_fake_target()
        else:
            assert self.backend._target_with_resonators is None

    def validate_move_loci_fake_target(self):
        """Check that the moves in the fake target are as in the dqa."""
        self.check_instruction("move", iqm_name="move", target=self.backend._target_with_resonators)

    def validate_cz_loci_fake_target(self):
        """Check that the czs in the fake target are as expected."""
        # From `validate_fake_cz_loci` we know that the virtual CZs are correct
        # Now we just need to add the real CZs
        real_loci = list(
            self.dqa.gates["cz"].implementations[self.dqa.gates["cz"].default_implementation].loci,
        )
        fake_loci = [
            tuple(self.backend.index_to_qubit_name(qb) for qb in loci)
            for i, loci in self.backend._target_with_resonators.instructions
            if i.name == "cz"
        ]
        expected_loci = real_loci + fake_loci
        self.check_instruction("cz", expected_loci=expected_loci, target=self.backend._target_with_resonators)

    def check_instruction(self, qiskit_name: str, iqm_name: str | None = None, expected_loci=None, target=None):
        """Checks that the given instruction is defined for the expected qubits (directed)."""
        if expected_loci is None:
            expected_loci = (
                [locus for impl in self.dqa.gates[iqm_name].implementations.values() for locus in impl.loci]
                if iqm_name in self.dqa.gates
                else []
            )
        if target is None:
            target = self.backend.target
        assert {
            tuple(self.backend.index_to_qubit_name(qb) for qb in loci)
            for (i, loci) in target.instructions
            if i.name == qiskit_name
        } == set(expected_loci)


@pytest.mark.parametrize(
    "dqa",
    [
        "move_architecture",
        "adonis_architecture",
        "hypothetical_fake_architecture",
        "star_architecture",
        "linear_3q_architecture",
    ],
    indirect=True,
)
class TestIQMTargetReflectsStructuredQPUData:
    """Test that the IQM backend reflects the extended architecture."""

    @pytest.fixture(autouse=True)
    def init_backend(self, dqa, request):
        """Initialize the backend with the given architecture."""
        self.dqa = dqa
        self.backend = get_mocked_backend(dqa, request, use_metrics=True)
        self.target = self.backend._target
        self.metrics = self.backend.metrics

    def test_target_init(self):
        assert self.target is not None
        if self.dqa.computational_resonators:
            assert "move" in self.target.iqm_dqa.gates

    def test_qubit_properties(self, sample_qubit_properties_by_arch):
        """Check qubit properties (T1, T2, frequencies) are correct."""
        target_qubit_properties = self.target.qubit_properties
        qubits = self.target.iqm_dqa.qubits
        assert target_qubit_properties is not None
        assert len(target_qubit_properties) == len(qubits)
        for got, expected in zip(target_qubit_properties, sample_qubit_properties_by_arch):
            assert got.t1 is not None
            assert got.t2 is not None
            assert got.frequency is not None
            assert got.t1 == expected.t1
            assert got.t2 == expected.t2
            assert got.frequency == expected.frequency

    @pytest.mark.parametrize(("qiskit_name", "iqm_name"), zip(["r", "measure", "reset"], ["prx", "measure", "cc_prx"]))
    def test_1_to_1_corresponding_gates(self, qiskit_name, iqm_name):
        """Check that the gates are defined for the correct qubits where the gates correspond 1-1 directly."""
        self.check_instruction(qiskit_name, iqm_name)
        if self.backend._target_with_resonators is not None:
            self.check_instruction(qiskit_name, iqm_name, target=self.backend._target_with_resonators)

    def test_prx_gates(self):
        """Check that the prx gates are defined for the correct qubits
        and have the correct instruction properties."""
        self.check_instruction_properties("r")

    def test_measure_gates(self):
        """Check that the prx gates are defined for the correct qubits
        and have the correct instruction properties."""
        self.check_instruction_properties("measure")

    def test_cz_gates(self):
        """Check that the cz gates are defined for the correct qubits."""
        if "move" not in self.dqa.gates:
            self.check_instruction_properties("cz")
        else:
            self.validate_move_loci()
            self.check_fake_cz_instruction_properties()

    def validate_move_loci(self):
        """Confirms that the move gate is not in the target."""
        assert "move" not in self.backend.target.operation_names

    def check_fake_cz_instruction_properties(self):
        """
        Check that the virtual czs in the target are as expected and with the right
        instruction properties. A "virtual" or "fake" CZ gate is one that is possible
        not due to a direct physical connection, but by using a MOVE gate to bring one
        qubit to a resonator that is connected to the other qubit.
        """
        # Get all CZ gate locations (qubit pairs) advertised by the target that includes virtual gates.
        # The qubit indices are converted back to their string names for easier comparison.
        real_cz_loci = list(
            self.dqa.gates["cz"].implementations[self.dqa.gates["cz"].default_implementation].loci,
        )
        fictional_cz_loci = [
            tuple(self.backend.index_to_qubit_name(qb) for qb in loci)
            for i, loci in self.backend._target_with_resonators.instructions
            if i.name == "cz"
        ]
        expected_loci = real_cz_loci + fictional_cz_loci
        self.check_instruction("cz", "cz", expected_loci=expected_loci, target=self.backend.target_with_resonators)
        target = self.backend.target_with_resonators

        move_loci = set(self.dqa.gates["move"].implementations[self.dqa.gates["move"].default_implementation].loci)
        move_impl = self.dqa.gates["move"].default_implementation
        cz_impl = self.dqa.gates["cz"].default_implementation

        for index, instruction in enumerate(target.instructions):
            if instruction[0].name == "cz":
                locus_idx = instruction[1]
                locus = tuple(self.backend.index_to_qubit_name(qb) for qb in locus_idx)

                # skip real cz loci
                if locus in real_cz_loci:
                    continue

                # find locus that has "move" operation defined for at least one element in fake cz locus
                q1, q2 = locus

                for res in self.dqa.computational_resonators:
                    # case 1: MOVE(q1, res), CZ(q2, res)
                    if (q2, res) in move_loci and (q1, res) in real_cz_loci:
                        move_locus = (q2, res)
                        cz_locus = (q1, res)
                    # case 2: MOVE(q2, res), CZ(q1, res)
                    elif (q1, res) in move_loci and (q2, res) in real_cz_loci:
                        move_locus = (q1, res)
                        cz_locus = (q2, res)

                # get durations and fidelities from metrics
                move_duration = self.metrics.get_gate_duration("move", move_impl, move_locus)
                cz_duration = self.metrics.get_gate_duration("cz", cz_impl, cz_locus)
                move_fidelity = self.metrics.get_gate_fidelity("move", move_impl, move_locus)
                cz_fidelity = self.metrics.get_gate_fidelity("cz", cz_impl, cz_locus)

                assert cz_fidelity is not None and move_fidelity is not None, (
                    f"Could not find constituent gates for fictional CZ on {locus}"
                )

                # compute the expected duration and error
                expected_duration = 2 * move_duration + cz_duration
                expected_error = (
                    None if (move_fidelity is None or cz_fidelity is None) else 1 - move_fidelity**2 * cz_fidelity
                )

                instruction_properties = target.instruction_properties(index)

                assert instruction_properties.duration == expected_duration
                assert instruction_properties.error == expected_error

    def validate_move_loci_fake_target(self):
        """Check that the moves in the fake target are as in the dqa."""
        self.check_instruction_properties("move", target=self.backend._target_with_resonators)

    def check_instruction_properties(self, qiskit_name: str, target=None):
        """Checks that instruction properties are correctly defined for the expected qubits."""
        iqm_name = QISKIT_TO_IQM.get(qiskit_name, "")
        if target is None:
            target = self.backend.target

        if iqm_name not in self.dqa.gates:
            return  # Gate not in DQA, nothing to check

        default_impl = self.dqa.gates[iqm_name].default_implementation

        for index, instruction in enumerate(target.instructions):
            if instruction[0].name == qiskit_name:
                locus_idx = instruction[1]
                locus = tuple(self.backend.index_to_qubit_name(qb) for qb in locus_idx)

                expected_duration = self.metrics.get_gate_duration(iqm_name, default_impl, locus)
                expected_fidelity = self.metrics.get_gate_fidelity(iqm_name, default_impl, locus)
                expected_error = None if expected_fidelity is None else 1 - expected_fidelity

                instruction_properties = target.instruction_properties(index)

                assert instruction_properties.duration == expected_duration
                assert instruction_properties.error == expected_error

    def check_instruction(self, qiskit_name: str, iqm_name: str | None = None, expected_loci=None, target=None):
        """Checks that the given instruction is defined for the expected qubits (directed)."""
        if expected_loci is None:
            expected_loci = (
                [locus for impl in self.dqa.gates[iqm_name].implementations.values() for locus in impl.loci]
                if iqm_name in self.dqa.gates
                else []
            )
        if target is None:
            target = self.backend.target
        assert {
            tuple(self.backend.index_to_qubit_name(qb) for qb in loci)
            for (i, loci) in target.instructions
            if i.name == qiskit_name
        } == set(expected_loci)


@pytest.mark.parametrize(
    "dqa,restriction",
    [
        ("adonis_architecture", ["QB4", "QB3", "QB1"]),
        ("move_architecture", ["QB5", "QB3", "QB1"]),
        ("move_architecture", ["QB5", "QB3", "QB1", "CR1"]),
    ],
    indirect=["dqa"],
)
def test_target_from_restricted_qubits(dqa, restriction, request):
    """Test that the restricted target is properly created."""
    backend = get_mocked_backend(dqa, request)
    restriction_idxs = [backend.qubit_name_to_index(qubit) for qubit in restriction]
    includes_resonators = any(qb in backend.architecture.computational_resonators for qb in restriction)

    for restricted in [restriction, restriction_idxs]:  # Check both string and integer restrictions
        if includes_resonators:
            restricted_target = backend.target_with_resonators.restrict_to_qubits(restricted)  # Restrict from IQMTarget
            assert restricted_target.num_qubits >= len(restricted)  # Resonators are included
        else:
            restricted_target = backend.target.restrict_to_qubits(restricted)  # Restrict from IQMTarget
            assert restricted_target.num_qubits == len(restricted)
        restricted_edges = restricted_target.build_coupling_map().get_edges()

        assert restricted_target.num_qubits == len(restriction)
        assert set(restricted_target.iqm_dqa.components) == set(restriction)

        # Check if the edges in the restricted target were allowed in the backend
        for edge in restricted_edges:
            translated_edge = (
                backend.qubit_name_to_index(restriction[edge[0]]),
                backend.qubit_name_to_index(restriction[edge[1]]),
            )
            if includes_resonators:
                assert translated_edge in backend.target_with_resonators.build_coupling_map().get_edges()
            else:
                assert translated_edge in backend.coupling_map.get_edges()
