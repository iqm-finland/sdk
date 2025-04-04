# Copyright 2022-2025 Qiskit on IQM developers
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
"""Error profile and fake backend base class for simulating IQM quantum computers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
from uuid import UUID

from iqm.iqm_client import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    Locus,
    StaticQuantumArchitecture,
)
from iqm.qiskit_iqm.iqm_backend import IQM_TO_QISKIT_GATE_NAME, IQMBackendBase
from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit
from iqm.qiskit_iqm.iqm_transpilation import IQMReplaceGateWithUnitaryPass
from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY
from qiskit import QuantumCircuit
from qiskit.providers import JobV1, Options
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error


def _dqa_from_sqa(
    sqa: StaticQuantumArchitecture,
    error_profile: IQMErrorProfile,
) -> DynamicQuantumArchitecture:
    """Create a dynamic quantum architecture from the given static quantum architecture
    and error profile.

    Since the DQA contains some attributes that are not present in an SQA or error profile,
    they are filled with mock data:

    * Each gate type is given a single mock implementation.
    * Calibration set ID is set to the all-zeros UUID.

    Args:
        sqa: Static quantum architecture to replicate.
        error_profile: Characteristics of a particular QPU specimen (including gate loci).

    Returns:
        DQA replicating the properties of ``sqa``.

    """

    def fake_gateinfo(loci: tuple[Locus, ...]) -> GateInfo:
        """Fake GateInfo for the given loci."""
        return GateInfo(
            implementations={"__fake": GateImplementationInfo(loci=loci)},
            default_implementation="__fake",
            override_default_implementation={},
        )

    # add all gates/loci found in the error profile
    gates = {
        gate_name: fake_gateinfo(tuple((component,) for component in gate_errors_1q))
        for gate_name, gate_errors_1q in error_profile.single_qubit_gate_depolarizing_error_parameters.items()
    }

    for gate_name, gate_errors_2q in error_profile.two_qubit_gate_depolarizing_error_parameters.items():
        gates[gate_name] = fake_gateinfo(tuple(gate_errors_2q))

    gates["measure"] = fake_gateinfo(tuple((component,) for component in error_profile.readout_errors))

    return DynamicQuantumArchitecture(
        calibration_set_id=UUID("00000000-0000-0000-0000-000000000000"),
        qubits=sqa.qubits,
        computational_resonators=sqa.computational_resonators,
        gates=gates,
    )


@dataclass
class IQMErrorProfile:
    r"""Characteristics of an IQM QPU specimen, used for constructing an error model.

    All the attributes of this class refer to the components of the QPU using their physical names.
    There are two types of QPU components, qubits and computational resonators.

    Args:
        t1s: maps components to their :math:`T_1` times (in ns)
        t2s: maps components to their :math:`T_2` times (in ns)
        single_qubit_gate_depolarizing_error_parameters: Depolarizing error parameters for single-qubit gates.
            Maps single-qubit gate names to a mapping of qubits (on which the gate acts) to a depolarizing error.
            The error, used in a one-qubit depolarizing channel, concatenated with a thermal relaxation channel,
            leads to average gate fidelities that would be determined by benchmarking.
        two_qubit_gate_depolarizing_error_parameters: Depolarizing error parameters for two-qubit gates.
            Maps two-qubit gate names to a mapping of pairs of qubits (on which the gate acts) to a depolarizing error.
            The error, used in a two-qubit depolarizing channel, concatenated with thermal relaxation channels for the
            qubits, leads to average gate fidelities that would be determined by benchmarking.
        single_qubit_gate_durations: Gate duration (in ns) for each single-qubit gate
        two_qubit_gate_durations: Gate duration (in ns) for each two-qubit gate.
        readout_errors: Maps physical qubit names to dicts that describe their single-qubit readout errors.
            For each qubit, the inner dict maps the state labels "0" and "1" to the probability :math:`P(\neg x|x)`
            of observing the state :math:`\ket{\neg x}` given the true state is :math:`\ket{x}`.
        name: Identifier of the QPU specimen.

    Example:
        .. code-block::

            IQMErrorProfile(
                t1s={"QB1": 10000.0, "QB2": 12000.0, "QB3": 14000.0},
                t2s={"QB1": 10000.0, "QB2": 12000.0, "QB3": 13000.0},
                single_qubit_gate_depolarizing_error_parameters={"prx": {"QB1": 0.0005, "QB2": 0.0004, "QB3": 0.0010}},
                two_qubit_gate_depolarizing_error_parameters={"cz": {("QB1", "QB2"): 0.08, ("QB2", "QB3"): 0.03}},
                single_qubit_gate_durations={"prx": 50.},
                two_qubit_gate_durations={"cz": 100.},
                readout_errors={"QB1": {"0": 0.02, "1": 0.03},
                                "QB2": {"0": 0.02, "1": 0.03},
                                "QB3": {"0": 0.02, "1": 0.03}},
                name="threequbit-example_sample"
            )

    """

    t1s: dict[str, float]
    t2s: dict[str, float]
    single_qubit_gate_depolarizing_error_parameters: dict[str, dict[str, float]]
    two_qubit_gate_depolarizing_error_parameters: dict[str, dict[tuple[str, str], float]]
    single_qubit_gate_durations: dict[str, float]
    two_qubit_gate_durations: dict[str, float]
    readout_errors: dict[str, dict[str, float]]
    name: str | None = None

    def thermal_relaxation(self, component: str, duration: float) -> QuantumError:
        """One-qubit relaxation error channel."""
        return thermal_relaxation_error(self.t1s[component], self.t2s[component], duration)


class IQMFakeBackend(IQMBackendBase):
    """Simulated backend that mimics the behaviour of IQM quantum computers.

    Can be used to perform noisy gate-level simulations of quantum circuit execution on IQM hardware.

    A fake backend contains information about a specific IQM system, such as the static quantum architecture
    (number of qubits, connectivity), and a noise model based on system parameters such as relaxation (:math:`T_1`)
    and dephasing (:math:`T_2`) times, gate infidelities, and readout errors.

    Args:
        architecture: Static quantum architecture associated with the backend instance.
        error_profile: Characteristics of a particular QPU specimen.

    """

    def __init__(
        self,
        architecture: StaticQuantumArchitecture,
        error_profile: IQMErrorProfile,
        name: str = "IQMFakeBackend",
        **kwargs,
    ):
        self._validate_architecture_and_error_profile(architecture, error_profile)
        dqa = _dqa_from_sqa(architecture, error_profile)
        super().__init__(dqa, **kwargs)
        self.__sqa = architecture
        self.__dqa = dqa
        self.__error_profile = error_profile

        self.noise_model = self._create_noise_model()
        self.name = name

    @property
    def error_profile(self) -> IQMErrorProfile:
        """Error profile of this IQM fake backend instance."""
        return deepcopy(self.__error_profile)

    @error_profile.setter
    def error_profile(self, value: IQMErrorProfile) -> None:
        raise NotImplementedError(
            "Setting error profile of existing fake backend is not allowed. "
            "You may consider using the method .copy_with_error_profile."
        )

    def copy_with_error_profile(self, new_error_profile: IQMErrorProfile) -> IQMFakeBackend:
        """Return another instance of IQMFakeBackend, which has the same quantum architecture but a different error
        profile.
        """
        return self.__class__(self.__sqa, new_error_profile, self.name)

    @staticmethod
    def _validate_architecture_and_error_profile(
        architecture: StaticQuantumArchitecture, error_profile: IQMErrorProfile
    ) -> None:
        """Verify that the parameters of the QPU error profile match the constraints of its quantum architecture.

        Raises:
            ValueError: when length of :attr:`t1s` and number of qubits do not match.
            ValueError: when length of :attr:`t2s` and number of qubits do not match.
            ValueError: when length of :attr:`one_qubit_gate` parameter lists and number of qubits do not match.
            ValueError: when length of :attr:`two_qubit_gate` parameter lists and number of couplings do not match.
            ValueError: when gates in gate parameter lists are not supported by the quantum architecture.

        """

        def compare_sets(a: set, b: set, msg: str) -> None:
            """Compare the contents of two sets."""
            if diff := a - b:
                raise ValueError(f"Error profile attribute {msg} has unknown loci: {diff}.")
            if diff := b - a:
                raise ValueError(f"Error profile attribute {msg} is missing loci: {diff}.")

        components = set(architecture.qubits + architecture.computational_resonators)
        # Check that T1 and T2 dicts have one item for each component
        compare_sets(set(error_profile.t1s.keys()), components, "t1s")
        compare_sets(set(error_profile.t2s.keys()), components, "t2s")

        # Check that one-qubit gate parameter qubits match those of the architecture
        for gate, gate_errors_1q in error_profile.single_qubit_gate_depolarizing_error_parameters.items():
            compare_sets(
                set(gate_errors_1q.keys()),
                set(architecture.qubits),
                f"single_qubit_gate_depolarizing_error_parameters['{gate}']",
            )

        # Check that two-qubit gate parameter couplings match those of the architecture
        # Locus order is ignored because the loci in architecture.connectivity are unordered.
        for gate, gate_errors_2q in error_profile.two_qubit_gate_depolarizing_error_parameters.items():
            compare_sets(
                set(frozenset(locus) for locus in gate_errors_2q.keys()),
                set(frozenset(locus) for locus in architecture.connectivity),
                f"two_qubit_gate_depolarizing_error_parameters['{gate}']",
            )

        # Check that the basis gates in the error profile are all known
        known_gates = {"prx", "cc_prx", "cz", "move"}
        for property_name, specified_gates in [
            (
                "single_qubit_gate_depolarizing_error_parameters",
                error_profile.single_qubit_gate_depolarizing_error_parameters.keys(),
            ),
            (
                "two_qubit_gate_depolarizing_error_parameters",
                error_profile.two_qubit_gate_depolarizing_error_parameters.keys(),
            ),
            ("durations", (error_profile.single_qubit_gate_durations | error_profile.two_qubit_gate_durations).keys()),
        ]:
            for gate in specified_gates:
                if gate not in known_gates:
                    raise ValueError(f"Gate '{gate}' in {property_name} is unknown. Valid gates: {known_gates}")

        compare_sets(set(error_profile.readout_errors.keys()), set(architecture.qubits), "readout_errors")

    def _create_noise_model(self) -> NoiseModel:
        """Build a noise model from the attributes."""
        error_profile = self.__error_profile
        iqm_to_qiskit_gates = IQM_TO_QISKIT_GATE_NAME.copy()
        for iqm_gate in self.__dqa.gates:
            if iqm_gate not in ["measure", "barrier"]:
                iqm_to_qiskit_gates.setdefault(iqm_gate, iqm_gate)

        noise_model = NoiseModel(basis_gates=list(iqm_to_qiskit_gates.values()))

        # Add single-qubit gate errors to noise model
        for gate, gate_errors_1q in error_profile.single_qubit_gate_depolarizing_error_parameters.items():
            gate_duration = error_profile.single_qubit_gate_durations[gate]
            for component, gate_error in gate_errors_1q.items():
                thermal_relaxation_channel = error_profile.thermal_relaxation(component, gate_duration)
                depolarizing_channel = depolarizing_error(gate_error, 1)
                noise_model.add_quantum_error(
                    thermal_relaxation_channel.compose(depolarizing_channel),
                    iqm_to_qiskit_gates[gate],
                    [self.qubit_name_to_index(component)],
                )

        # Add two-qubit gate errors to noise model
        for gate, gate_errors_2q in error_profile.two_qubit_gate_depolarizing_error_parameters.items():
            gate_duration = error_profile.two_qubit_gate_durations[gate]
            for locus, gate_error in gate_errors_2q.items():
                for qb_order in permutations(locus):
                    # TODO why do we need to add the other locus order?
                    thermal_channels = [
                        error_profile.thermal_relaxation(component, gate_duration) for component in qb_order
                    ]
                    thermal_relaxation_channel = thermal_channels[0].tensor(thermal_channels[1])
                    depolarizing_channel = depolarizing_error(gate_error, 2)
                    noise_model.add_quantum_error(
                        thermal_relaxation_channel.compose(depolarizing_channel),
                        iqm_to_qiskit_gates[gate],
                        [self.qubit_name_to_index(qb_order[0]), self.qubit_name_to_index(qb_order[1])],
                    )

        # Add readout errors
        for qb, readout_error in error_profile.readout_errors.items():
            probabilities = [[1 - readout_error["0"], readout_error["0"]], [readout_error["1"], 1 - readout_error["1"]]]
            noise_model.add_readout_error(probabilities, [self.qubit_name_to_index(qb)])

        return noise_model

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024)

    @property
    def max_circuits(self) -> int | None:
        return None

    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], **options) -> JobV1:
        """Run quantum circuits on the fake backend (by simulating them).

        This method will run the simulation with the noise model of the fake backend.
        Validity of the circuits is also checked.

        Args:
            run_input: One or more quantum circuits to simulate on the backend.
            options: Any kwarg options to pass to the backend.

        Returns:
            The job object representing the run.

        Raises:
            ValueError: Empty list of circuits was provided.

        """
        circuits_aux = [run_input] if isinstance(run_input, QuantumCircuit) else run_input

        if len(circuits_aux) == 0:
            raise ValueError("Empty list of circuits submitted for execution.")

        circuits = []
        GATE_TO_UNITARY = {
            "move": MOVE_GATE_UNITARY,
        }

        for circ in circuits_aux:
            validate_circuit(circ, self)
            circ_updated = circ

            for g in self.noise_model.basis_gates:
                if g not in IQM_TO_QISKIT_GATE_NAME.values():
                    circ_updated = IQMReplaceGateWithUnitaryPass(g, GATE_TO_UNITARY[g])(circ)
            circuits.append(circ_updated)

        shots = options.get("shots", self.options.shots)

        # Create noisy simulator backend and run circuits
        sim_noise = AerSimulator(noise_model=self.noise_model)

        job = sim_noise.run(circuits, shots=shots)
        return job

    def validate_compatible_architecture(self, architecture: StaticQuantumArchitecture) -> bool:
        """Compare a static quantum architecture to the static architecture of the fake backend.

        Args:
            architecture: static quantum architecture to compare to

        Returns:
            True iff the locus components and the component connectivity
            in the IQMFakeBackend SQA match ``architecture``.

        """
        return architecture == self.__sqa
