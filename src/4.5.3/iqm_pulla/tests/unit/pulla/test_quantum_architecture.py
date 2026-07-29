# Copyright 2024-2026 IQM
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
"""Test creating quantum architectures."""

from datetime import datetime, timezone
import uuid

from exa.common.data.value import ObservationValue
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulla.quantum_architecture import create_dynamic_quantum_architecture
from iqm.station_control.interface.models import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    ObservationLite,
)


def observation(name: str, value: str = "") -> tuple[str, ObservationValue]:
    """Observation creation helper."""
    return name, value


def full_observation(name: str, value: str = "") -> ObservationLite:
    """Observation creation helper."""
    return ObservationLite(
        dut_field=name,
        unit="s",
        value=value,
        uncertainty=None,
        invalid=False,
        observation_id=999,
        created_timestamp=datetime.now(timezone.utc),
        modified_timestamp=datetime.now(timezone.utc),
    )


def test_get_dynamic_quantum_architecture():
    """Generated DQA matches the input calset."""
    cal_set_id = uuid.uuid4()
    chip_topology = ChipTopology(
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR"],
        couplers={"TC-1": ["QB1", "QB2"], "TC-2": ["QB2", "CR"]},
        probe_lines={},
    )
    cal_set_contents = [
        observation("gates.measure.constant.QB1.some_par"),
        observation("gates.measure.constant.QB2.some_par"),
        observation("gates.measure.constant.QB2.another_par"),
        observation("gates.prx.drag_gaussian.QB1.some_par"),
        observation("gates.prx.drag_gaussian.QB2.some_par"),
        observation("gates.prx.drag_gaussian.QB3.some_par"),
        observation("gates.prx.drag_crf.QB1.some_par"),
        observation("gates.prx.drag_crf.QB2.some_par"),
        observation("gates.prx.drag_crf.QB3.some_par"),
        observation("gates.prx.drag_crf.QB4.some_par"),
        observation("not.a.gate.parameter"),  # ignored
        observation("gates.prx.unknown_implementation.QB1.some_par"),  # ignored
        observation("gates.unknown_gate.xxx.QB1.some_par"),  # ignored
        observation("gates.cz.crf.QB1__QB2.some_par"),
        observation("gates.cz.tgss_crf.QB1__QB2.some_par"),
        observation("gates.cz.tgss_crf.QB2__QB3.some_par"),
        observation("gates.cz.crf.QB2__CR.some_par"),
        observation("gate_definitions.measure.default_implementation", "constant"),
        observation("gate_definitions.prx.default_implementation", "drag_gaussian"),
        observation("gate_definitions.prx.drag_crf.override_default_for_loci", ["QB1", "QB3"]),
        observation("gate_definitions.cz.default_implementation", "tgss_crf"),
        observation("gate_definitions.cz.crf.override_default_for_loci", ["QB1__QB2", "QB2__CR"]),
    ]
    expected_dqa = DynamicQuantumArchitecture(
        calibration_set_id=str(cal_set_id),
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR"],
        gates={
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",)),
                    )
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(
                        loci=(
                            ("QB1",),
                            ("QB2",),
                            ("QB3",),
                        ),
                    ),
                    "drag_crf": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",)),
                    ),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={("QB1",): "drag_crf", ("QB3",): "drag_crf"},
            ),
            "cz": GateInfo(
                implementations={
                    "tgss_crf": GateImplementationInfo(
                        loci=(("QB1", "QB2"), ("QB2", "QB3")),
                    ),
                    "crf": GateImplementationInfo(
                        loci=(("QB1", "QB2"), ("QB2", "CR")),
                    ),
                },
                default_implementation="tgss_crf",
                override_default_implementation={("QB1", "QB2"): "crf", ("QB2", "CR"): "crf"},
            ),
        },
    )

    dqa = create_dynamic_quantum_architecture(cal_set_id, dict(cal_set_contents), chip_topology)
    assert dqa == expected_dqa


def test_get_dynamic_quantum_architecture_old_method():
    """Generated DQA matches the input calset.

    Uses the old method where the calset contains no default implementation observations.
    """
    cal_set_id = uuid.uuid4()
    chip_topology = ChipTopology(
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR"],
        couplers={"TC-1": ["QB1", "QB2"], "TC-2": ["QB2", "CR"]},
        probe_lines={},
    )
    cal_set_contents = [
        observation("gates.measure.constant.QB1.some_par"),
        observation("gates.measure.constant.QB2.some_par"),
        observation("gates.measure.constant.QB2.another_par"),
        observation("gates.prx.drag_gaussian.QB1.some_par"),  # higher priority makes this default for QB1
        observation("gates.prx.drag_crf.QB1.some_par"),
        observation("gates.prx.drag_crf.QB2.some_par"),
        observation("gates.prx.drag_crf.QB3.some_par"),
        observation("not.a.gate.parameter"),  # ignored
        observation("gates.prx.unknown_implementation.QB1.some_par"),  # ignored
        observation("gates.unknown_gate.xxx.QB1.some_par"),  # ignored
        observation("gates.cz.tgss_crf.QB1__QB2.some_par"),
        observation("gates.cz.tgss_crf.QB1__QB2.some_other_par"),
        observation("gates.cz.tgss_crf.QB2__CR.some_par"),
    ]
    expected_dqa = DynamicQuantumArchitecture(
        calibration_set_id=str(cal_set_id),
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR"],
        gates={
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",)),
                    )
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(
                        loci=(("QB1",),),
                    ),
                    "drag_crf": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",), ("QB3",)),
                    ),
                },
                default_implementation="drag_crf",
                override_default_implementation={("QB1",): "drag_gaussian"},
            ),
            "cz": GateInfo(
                implementations={
                    "tgss_crf": GateImplementationInfo(
                        loci=(("QB1", "QB2"), ("QB2", "CR")),
                    )
                },
                default_implementation="tgss_crf",
                override_default_implementation={},
            ),
        },
    )

    dqa = create_dynamic_quantum_architecture(cal_set_id, dict(cal_set_contents), chip_topology)
    assert dqa == expected_dqa


def test_get_dynamic_quantum_architecture_only_unknown_implementations():
    """Known gate that has only unknown implementations in the calset does not appear in the DQA."""
    cal_set_id = uuid.uuid4()
    chip_topology = ChipTopology(
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR"],
        couplers={"TC-1": ["QB1", "QB2"], "TC-2": ["QB2", "CR"]},
        probe_lines={},
    )
    cal_set_contents = [
        observation("gates.prx.drag_gaussian.QB1.some_par"),
        observation("gates.prx.drag_crf.QB1.some_par"),
        observation("not.a.gate.parameter"),
        # cz only has unknown implementations in the calset
        observation("gates.cz.xxx.QB1__QB2.some_par"),
        observation("gates.cz.yyy.QB2__CR.some_par"),
    ]
    expected_dqa = DynamicQuantumArchitecture(
        calibration_set_id=str(cal_set_id),
        qubits=["QB1"],
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(
                        loci=[["QB1"]],
                    ),
                    "drag_crf": GateImplementationInfo(
                        loci=[["QB1"]],
                    ),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
        },
    )
    dqa = create_dynamic_quantum_architecture(cal_set_id, dict(cal_set_contents), chip_topology)
    assert dqa == expected_dqa
