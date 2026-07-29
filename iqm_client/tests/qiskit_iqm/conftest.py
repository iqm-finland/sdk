# Copyright 2022 Qiskit on IQM developers
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
"""Shared definitions for tests."""

from uuid import UUID, uuid4

from iqm.iqm_client import IQMClient
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.qiskit_iqm import IQMBackend, IQMTarget
from mockito import mock, when
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.backend import QubitProperties

from iqm.station_control.client.qon import (
    QON,
    ObservationFinder,
    QONCharacterization,
    QONControllerSetting,
    QONGateMetric,
    QONGateParam,
    locus_to_locus_str,
)
from iqm.station_control.interface.models import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    ObservationBase,
    RunRequest,
    StaticQuantumArchitecture,
)


@pytest.fixture()
def sample_calset_id() -> UUID:
    return UUID("56cd3adf-4425-42e7-8789-036388563a0f")


@pytest.fixture()
def sample_quality_metric_set_id() -> UUID:
    return UUID("88c6e919-61df-4d40-b74d-9df8f68b0e85")


@pytest.fixture
def client():
    return mock(IQMClient)


def get_mocked_backend(architecture: DynamicQuantumArchitecture, request, use_metrics=True) -> IQMBackend:
    """Returns an IQM backend running on a mocked IQM client that returns the given architecture."""
    client = request.getfixturevalue("iqm_client_mock")

    calset_id = architecture.calibration_set_id
    metrics = create_metrics_from_dqa(architecture)

    when(client).get_dynamic_quantum_architecture(calset_id).thenReturn(architecture)

    if use_metrics:
        when(client)._get_calibration_quality_metrics(calset_id).thenReturn(metrics)

    backend = IQMBackend(client, calibration_set_id=calset_id, use_metrics=use_metrics)

    return backend


@pytest.fixture
def sample_target_linear_3q_architecture(
    linear_3q_architecture,
    qb_to_idx_linear_3q_architecture,
):
    return IQMTarget(
        architecture=linear_3q_architecture,
        component_to_idx=qb_to_idx_linear_3q_architecture,
        include_resonators=False,
        include_fictional_czs=True,
    )


@pytest.fixture
def circuit():
    return QuantumCircuit(3, 3)


@pytest.fixture
def circuit_2() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.cz(0, 1)
    return circuit


@pytest.fixture
def create_run_request_default_kwargs(linear_3q_architecture) -> dict:
    return {
        "calibration_set_id": linear_3q_architecture.calibration_set_id,
        "shots": 1024,
        "options": CircuitCompilationOptions(),
    }


@pytest.fixture
def job_id() -> UUID:
    return uuid4()


@pytest.fixture
def run_request():
    run_request = mock(RunRequest)
    run_request.circuits = []
    run_request.shots = 1
    return run_request


@pytest.fixture
def linear_3q_static_architecture():
    return StaticQuantumArchitecture(
        dut_label="M138_W0_XXX_Z99",
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=[],
        connectivity=[("QB1", "QB2"), ("QB2", "QB3")],
    )


def _1q_loci(qubits: list[str]) -> tuple[tuple[str, ...], ...]:
    """One-qubit loci for the given qubits."""
    return tuple((q,) for q in qubits)


@pytest.fixture
def qb_to_idx_linear_3q_architecture(linear_3q_architecture):
    """Mapping of qubit names to indices for the linear 3-qubit architecture."""
    return {
        qb: idx
        for idx, qb in enumerate(linear_3q_architecture.qubits + linear_3q_architecture.computational_resonators)
    }


@pytest.fixture
def qb_to_idx_move_architecture(move_architecture):
    """Mapping of qubit names to indices for the architecture containing a MOVE gate."""
    return {name: idx for idx, name in enumerate(move_architecture.qubits + move_architecture.computational_resonators)}


@pytest.fixture
def linear_3q_architecture(sample_calset_id) -> DynamicQuantumArchitecture:
    qubits = ["QB1", "QB2", "QB3"]
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=qubits,
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={"drag_gaussian": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cc_prx": GateInfo(
                implementations={"prx_composite": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="prx_composite",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={"tgss": GateImplementationInfo(loci=(("QB1", "QB2"), ("QB2", "QB3")))},
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={"constant": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def disconnected_3q_architecture(sample_calset_id) -> DynamicQuantumArchitecture:
    qubits = ["QB1", "QB2", "QB3"]
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=qubits,
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={"drag_gaussian": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cc_prx": GateInfo(
                implementations={"prx_composite": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="prx_composite",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={"tgss": GateImplementationInfo(loci=(("QB1", "QB2"),))},
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={"constant": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def adonis_static_architecture():
    return StaticQuantumArchitecture(
        dut_label="M138_W0_A22_Z99",
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5"],
        computational_resonators=[],
        connectivity=[("QB1", "QB3"), ("QB2", "QB3"), ("QB3", "QB4"), ("QB3", "QB5")],
    )


@pytest.fixture
def adonis_architecture(sample_calset_id):
    qubits = ["QB1", "QB2", "QB3", "QB4", "QB5"]
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=qubits,
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={"drag_gaussian": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cc_prx": GateInfo(
                implementations={"prx_composite": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="prx_composite",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={
                    "tgss": GateImplementationInfo(
                        loci=(("QB1", "QB3"), ("QB2", "QB3"), ("QB4", "QB3"), ("QB5", "QB3"))
                    )
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={"constant": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def move_architecture(sample_calset_id):
    qubits = ["QB1", "QB2", "QB3", "QB4", "QB5", "QB6"]
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=qubits,
        computational_resonators=["CR1"],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(loci=_1q_loci(qubits)),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cc_prx": GateInfo(
                implementations={"prx_composite": GateImplementationInfo(loci=_1q_loci(qubits))},
                default_implementation="prx_composite",
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
                            ("QB5", "CR1"),
                        )
                    ),
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "move": GateInfo(
                implementations={
                    "tgss_crf": GateImplementationInfo(loci=(("QB6", "CR1"),)),
                },
                default_implementation="tgss_crf",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(loci=_1q_loci(qubits)),
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture()
def hypothetical_fake_architecture(sample_calset_id):
    """Generate a hypothetical fake device for testing.

          QB1   QB2
           |    |
             CR1
           |*    *
    QB3 - QB4   QB7 - QB8
           |*    *
             CR2
            |   |
          QB5   QB6

    Here, '|' signifies a CZ connection and the '*' signify a move connection.

    """
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5", "QB6", "QB7", "QB8"],
        computational_resonators=["CR1", "CR2"],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",), ("QB5",), ("QB6",), ("QB7",), ("QB8",))
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
                            ("QB3", "QB4"),
                            ("QB4", "CR1"),
                            ("QB4", "CR2"),
                            ("QB5", "CR2"),
                            ("QB6", "CR2"),
                            ("QB7", "QB8"),
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
                            ("QB4", "CR1"),
                            ("QB4", "CR2"),
                            ("QB7", "CR1"),
                            ("QB7", "CR2"),
                        ),
                    )
                },
                default_implementation="tgss_crf",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",), ("QB5",), ("QB6",), ("QB7",), ("QB8",)),
                    )
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def star_architecture(sample_calset_id):
    qubits = ["QB1", "QB2", "QB3", "QB4", "QB5", "QB6"]
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=qubits,
        computational_resonators=["CR1"],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(loci=_1q_loci(qubits)),
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
                            ("QB5", "CR1"),
                            ("QB6", "CR1"),
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
                            ("QB1", "CR1"),
                            ("QB2", "CR1"),
                            ("QB3", "CR1"),
                            ("QB4", "CR1"),
                            ("QB5", "CR1"),
                            ("QB6", "CR1"),
                        )
                    ),
                },
                default_implementation="tgss_crf",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(loci=_1q_loci(qubits)),
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def sample_qubit_properties_by_arch(request) -> list[QubitProperties]:
    """Calculates the expected QubitProperties for a given architecture."""
    architecture = request.getfixturevalue("dqa")
    metrics = create_metrics_from_dqa(architecture)
    qubits = architecture.qubits

    t1s, t2s = metrics.get_coherence_times(qubits)

    return [QubitProperties(t1=t1s.get(q), t2=t2s.get(q), frequency=metrics.get_qubit_frequency(q)) for q in qubits]


@pytest.fixture
def mock_iqm_backend_init(monkeypatch):
    """
    Patches the IQMTarget class during initialization of IQMBackend,
    which requires metrics data.
    """

    def mock_target_factory(*args, **kwargs):
        # Extract arguments
        architecture = None
        include_resonators = kwargs.get("include_resonators", False)

        if args:
            architecture = args[0]
        elif "architecture" in kwargs:
            architecture = kwargs["architecture"]

        if architecture is not None:
            # Create different targets based on include_resonators
            if include_resonators:
                # This is for the _target_with_resonators
                return mock_iqm_target(architecture, include_resonators=True)
            else:
                # This is for the regular _target
                return mock_iqm_target(architecture, include_resonators=False)

    monkeypatch.setattr("iqm.qiskit_iqm.iqm_backend.IQMTarget", mock_target_factory)


def _observation_field(qon: QON, value: float, unit: str) -> ObservationBase:
    """Build an Observation with the given properties."""
    return ObservationBase(dut_field=str(qon), value=value, unit=unit)


def create_metrics_from_dqa(architecture: DynamicQuantumArchitecture) -> ObservationFinder:
    """Create ObservationFinder object with fake metrics for the given architecture."""
    observations = []
    for qb in architecture.qubits:
        obs = [
            _observation_field(QONCharacterization(component=qb, quantity="t1_time"), 1.0e-6, "s"),
            _observation_field(QONCharacterization(component=qb, quantity="t2_time"), 1.0e-6, "s"),
            _observation_field(QONControllerSetting(controller=qb, rest="drive.frequency"), 4.5e9, "Hz"),
            _observation_field(
                QONGateParam(gate="measure", implementation="constant", locus_str=qb, parameter="integration_length"),
                100e-9,
                "s",
            ),
        ]
        observations.extend(obs)

    METHODS = {
        "prx": "rb",
        "measure": "ssro",
    }
    SUFFIXES = {"measure": {}}

    def add_gate(gate: str, fidelity: float, duration: float) -> None:
        """Add fake metric observations for the given gate in the DQA."""
        gate_info = architecture.gates[gate]
        for locus in gate_info.loci:
            impl = gate_info.get_default_implementation(locus)
            locus_str = locus_to_locus_str(locus)
            observations.extend(
                [
                    _observation_field(
                        QONGateMetric(
                            method=METHODS.get(gate, "irb"),
                            gate=gate,
                            implementation=impl,
                            locus_str=locus_str,
                            metric="fidelity",
                            suffixes=SUFFIXES.get(gate, {"par": "d2"}),
                        ),
                        fidelity,
                        "",
                    ),
                    _observation_field(
                        QONGateParam(
                            gate=gate,
                            implementation=impl,
                            locus_str=locus_str,
                            parameter="duration",
                        ),
                        duration,
                        "s",
                    ),
                ]
            )

    for gate_name in architecture.gates:
        add_gate(gate_name, fidelity=0.999, duration=30e-9)

    return ObservationFinder(observations)


def mock_iqm_target(architecture: DynamicQuantumArchitecture, include_resonators: bool = False):
    """Create IQMTarget instance based on provided dqa and metrics."""
    metrics = create_metrics_from_dqa(architecture)
    qb_to_idx = {qb: idx for idx, qb in enumerate(architecture.qubits + architecture.computational_resonators)}
    return IQMTarget(
        architecture=architecture,
        component_to_idx=qb_to_idx,
        include_resonators=include_resonators,
        include_fictional_czs=True,
        metrics=metrics,
    )
