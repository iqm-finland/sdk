# Copyright 2024-2025 IQM
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
"""Circuit compilation tests."""

from collections.abc import Iterable
from math import pi
import re
from typing import Any

import pytest

from exa.common.data.setting_node import SettingNode
from iqm.cpc.compiler.compiler import Compiler, build_settings, initialize_schedule_builder
from iqm.cpc.compiler.errors import CalibrationError, CircuitError
from iqm.cpc.compiler.standard_stages import get_standard_stages
from iqm.cpc.interface.compiler import (
    CircuitBoundaryMode,
    CircuitMetrics,
    DDMode,
    DDStrategy,
    HeraldingMode,
    MeasurementMode,
)
from iqm.pulse import Circuit
from iqm.pulse import CircuitOperation as I
from iqm.pulse.playlist import Schedule
from iqm.pulse.playlist.instructions import (
    Block,
    ConditionalInstruction,
    IQPulse,
    ReadoutTrigger,
    RealPulse,
    VirtualRZ,
    Wait,
)


@pytest.fixture
def one_qubit_gates_on_one_qubit():
    return Circuit(
        name="single qubit",
        instructions=(
            I(name="prx", locus=("QB2",), args={"angle": 0.1, "phase": 0.1}),
            I(name="prx", locus=("QB2",), args={"angle": 0.2, "phase": 0.2}),
            I(name="measure", locus=("QB2",), args={"key": "result"}),
        ),
    )


@pytest.fixture
def default_options(default_options_generator):
    return default_options_generator.from_defaults()


def _check_schedule(schedule: Any, keys: Iterable[str]) -> None:
    """Checks that the given pulse schedule is valid and has the expected keys."""
    assert isinstance(schedule, Schedule)
    assert set(schedule) == set(keys)


class TestCompileCircuits:
    ### Compiling faulty circuits

    shots: int = 127

    def test_2q_gate_between_uncoupled_qubits(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options
    ):
        circuit = Circuit(
            name="xxx",
            instructions=(
                I(name="cz", locus=("QB1", "QB2"), args={}),
                I(name="measure", locus=("QB1",), args={"key": "m"}),
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=re.escape("Circuit 0: No calibration data for 'cz.tgss' at ('QB1', 'QB2')"),
        ) as outer_exception:
            Compiler(
                calibration_set_values=calibration_set_values,
                chip_topology=chip_topology,
                channel_properties=channel_properties,
                component_channels=component_channels,
                component_mapping=None,
                options=default_options,
                stages=get_standard_stages(),
            ).compile([circuit])

        assert isinstance(outer_exception.value.__cause__, CircuitError)

    def test_calibration_data_is_reported_missing_and_fails(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options,
        one_qubit_gates_on_one_qubit,
    ):
        circuit = one_qubit_gates_on_one_qubit
        del calibration_set_values["gates.prx.drag_gaussian.QB2.duration"]
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "prx.drag_gaussian at ('QB2',): Missing calibration data .{'duration'}",
            ),
        ) as outer_exception:
            Compiler(
                calibration_set_values=calibration_set_values,
                chip_topology=chip_topology,
                channel_properties=channel_properties,
                component_channels=component_channels,
                component_mapping=None,
                options=default_options,
                stages=get_standard_stages(),
                strict=True,
            ).compile([circuit])

        assert isinstance(outer_exception.value.__cause__, ValueError)

    def test_compiler_init_calibration_data_missing_does_not_fail(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options,
    ):
        del calibration_set_values["gates.prx.drag_gaussian.QB2.duration"]
        # Compiler can be initialized without failing with missing calibration data if strict is False, which is default
        Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options,
            stages=[],
        )

    def test_superfluous_calibration_data_is_rejected(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options,
        one_qubit_gates_on_one_qubit,
    ):
        circuit = one_qubit_gates_on_one_qubit
        calibration_set_values["gates.prx.drag_gaussian.QB2.invalid"] = 55
        with pytest.raises(
            CalibrationError,
            match=re.escape("prx.drag_gaussian at ('QB2',): Unknown calibration data .{'invalid'}"),
        ):
            Compiler(
                calibration_set_values=calibration_set_values,
                chip_topology=chip_topology,
                channel_properties=channel_properties,
                component_channels=component_channels,
                component_mapping=None,
                options=default_options,
                stages=get_standard_stages(),
                strict=True,
            ).compile([circuit])

    def test_unrecognized_gates(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options,
        one_qubit_gates_on_one_qubit,
    ):
        circuit = one_qubit_gates_on_one_qubit
        calibration_set_values["gates.some_new_gate.drag_gaussian.QB2.duration"] = 10e-9
        calibration_set_values["gates.prx.some_new_implementation.QB2.duration"] = 10e-9

        with pytest.raises(
            CalibrationError,
            match=re.escape("Unknown implementation 'some_new_implementation' for quantum operation 'prx'"),
        ):
            Compiler(
                calibration_set_values=calibration_set_values,
                chip_topology=chip_topology,
                channel_properties=channel_properties,
                component_channels=component_channels,
                component_mapping=None,
                options=default_options,
                stages=get_standard_stages(),
                strict=True,
            ).compile([circuit])

    ### Valid compilation cases

    @pytest.mark.parametrize(
        "qubit_mapping, logical_qubit",
        [
            ({"Alice": "QB1"}, "Alice"),  # with and without qubit_mapping
            (None, "QB1"),
        ],
    )
    def test_only_measurement(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options_generator,
        qubit_mapping,
        logical_qubit,
    ):
        circuit = Circuit(name="xxx", instructions=(I(name="measure", locus=(logical_qubit,), args={"key": "m"}),))

        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=qubit_mapping,
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]

        assert len(schedule) == 1  # Measure

        assert isinstance(settings, SettingNode)
        # 3x QB1 (flux, drive, readout), 2x QB3, QB4 (flux, drive, boundary qubits),
        # 2x PL_1, PL_2 (RO, TWPA), 1x TC-1-3, TC-1-4 (flux, boundary coupler),
        # 1x common options
        assert len(settings.subtrees) == 13
        assert isinstance(readout_mappings, tuple)
        assert len(readout_mappings) == 1
        assert isinstance(readout_mappings[0], dict)
        assert len(readout_mappings[0]) == 1
        assert readout_mappings[0]["m"] == ("QB1__m1",)

    def test_deprecated_op_names_and_args(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options_generator,
    ):
        circuit = Circuit(
            name="single qubit",
            instructions=(
                I(name="phased_rx", locus=("QB2",), args={"angle": 0.2 * pi, "phase": 0.2 * pi}),
                I(name="measurement", locus=("QB2",), args={"key": "result"}),
            ),
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options_generator.from_defaults(measurement_mode=MeasurementMode.CIRCUIT),
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])
        schedule = context["schedules"][0]
        metrics = context["circuit_metrics"]

        _check_schedule(schedule, ["QB2__drive.awg", "PL_2__readout"])
        assert len(schedule["QB2__drive.awg"]) == 3

        assert len(metrics) == 1
        assert metrics[0].components == {"QB2"}
        assert len(metrics[0].gate_loci) == 2
        assert metrics[0].gate_loci["prx"]["drag_gaussian"][("QB2",)] == 1
        assert metrics[0].gate_loci["measure"]["constant"][("QB2",)] == 1
        assert metrics[0].component_pairs_with_gates == set()

    def test_one_qubit_gates_on_one_qubit(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options_generator,
        one_qubit_gates_on_one_qubit,
    ):
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([one_qubit_gates_on_one_qubit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]
        metrics = context["circuit_metrics"]

        _check_schedule(schedule, ["QB2__drive.awg", "PL_2__readout"])

        assert len(schedule["QB2__drive.awg"]) == 4
        assert len(schedule["PL_2__readout"]) == 2

        # 2x QB2 (flux, drive), 2x QB3, QB4 (flux, drive, boundary qubits),
        # 2x PL_2,PL_1 (RO, TWPA), 1x TC_2_3, TC_2_4 (flux, boundary coupler)
        # 1x common options
        assert len(settings.subtrees) == 13
        assert isinstance(readout_mappings, tuple)
        assert len(readout_mappings) == 1
        assert isinstance(readout_mappings[0], dict)
        assert len(readout_mappings[0]) == 1
        assert readout_mappings[0]["result"] == ("QB2__m1",)

        assert len(metrics) == 1
        assert metrics[0].components == {"QB2"}
        assert metrics[0].component_pairs_with_gates == set()

    def test_settings_are_generated_correctly(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options_generator,
        one_qubit_gates_on_one_qubit,
    ):
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([one_qubit_gates_on_one_qubit])
        settings, _ = compiler.build_settings(context, shots=self.shots)
        assert set(settings.children.keys()) == {
            "QB2__flux",
            "QB2__drive",
            "PL_1__twpa",
            "PL_1__readout",
            "PL_2__twpa",
            "PL_2__readout",
            "QB4__flux",
            "QB4__drive",
            "QB3__flux",
            "QB3__drive",
            "TC-2-4__flux",
            "TC-2-3__flux",
            "options",
        }
        assert settings.QB2__flux.voltage.name == "QB2__flux.voltage"
        assert settings.QB2__flux.voltage.value == 1.0
        assert settings.QB2__drive.awg.name == "QB2__drive.awg"
        assert settings.QB2__drive.awg.trigger_delay.name == "QB2__drive.awg.trigger_delay"
        assert settings.QB2__drive.awg.trigger_delay.value == 5e-08

    def test_one_qubit_gates_on_one_qubit_with_circuit_boundary_all(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        one_qubit_gates_on_one_qubit,
        default_options_generator,
    ):
        options = default_options_generator.from_defaults(measurement_mode=MeasurementMode.ALL)
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([one_qubit_gates_on_one_qubit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]
        metrics = context["circuit_metrics"]

        _check_schedule(schedule, ["QB2__drive.awg", "PL_2__readout", "PL_1__readout"])

        assert len(schedule["QB2__drive.awg"]) == 4
        assert len(schedule["PL_2__readout"]) == 2

        # based on the calibration set, the circuit boundary mode 'all' will have:
        # 2x QB2 (flux, drive),
        # 1x QB1, QB3, QB4, QB5 (flux, boundary qubits), 1x QB1, QB3 (drive, boundary qubits)
        # 2x PL_2,PL_1 (RO, TWPA), 1x TC-1-3, TC-1-4, TC_2_3, TC_2_4 (flux, boundary coupler)
        # 1x common options
        assert len(settings.subtrees) == 17
        assert isinstance(readout_mappings, tuple)
        assert len(readout_mappings) == 1
        assert isinstance(readout_mappings[0], dict)
        assert len(readout_mappings[0]) == 1
        assert readout_mappings[0]["result"] == ("QB2__m1",)

        assert len(metrics) == 1
        assert metrics[0].components == {"QB2"}
        assert metrics[0].component_pairs_with_gates == set()

    def test_unknown_circuit_boundary_mode(self, default_options_generator):
        with pytest.raises(ValueError, match="'wrong' is not a valid CircuitBoundaryMode"):
            default_options_generator.from_defaults(circuit_boundary_mode=CircuitBoundaryMode("wrong"))

    def test_one_qubit_gates_on_one_qubit_with_measurement_mode_all(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        one_qubit_gates_on_one_qubit,
        default_options_generator,
    ):
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.ALL, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([one_qubit_gates_on_one_qubit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]
        metrics = context["circuit_metrics"]

        _check_schedule(schedule, ["QB2__drive.awg", "PL_2__readout", "PL_1__readout"])

        assert len(schedule["QB2__drive.awg"]) == 4

        # 2x QB2 (flux, drive), 2x QB3, QB4 (flux, drive, boundary qubits),
        # 2x PL1, PL_2 (RO, TWPA), 1x TC_2_3, TC_2_4 (flux, boundary coupler)
        # 1x common options
        assert len(settings.subtrees) == 13
        assert isinstance(readout_mappings, tuple)
        assert len(readout_mappings) == 1
        assert isinstance(readout_mappings[0], dict)
        assert len(readout_mappings[0]) == 1
        assert readout_mappings[0]["result"] == ("QB2__m1",)

        assert len(metrics) == 1
        assert metrics[0].components == {"QB2"}
        assert metrics[0].component_pairs_with_gates == set()

    def test_one_qubit_gates_on_two_qubits(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit = Circuit(
            name="two qubits",
            instructions=(
                I(name="prx", locus=("QB1",), args={"angle": 0.5, "phase": 0.1}),
                I(name="prx", locus=("QB1",), args={"angle": 0.2, "phase": 0.2}),
                I(name="prx", locus=("QB2",), args={"angle": 0.5, "phase": 0.3}),
                I(name="measure", locus=("QB2",), args={"key": "m2"}),
                I(name="measure", locus=("QB1",), args={"key": "m1"}),
            ),
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options_generator.from_defaults(
                measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
            ),
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]
        metrics = context["circuit_metrics"]

        _check_schedule(schedule, ["QB1__drive.awg", "QB2__drive.awg", "PL_1__readout", "PL_2__readout"])

        # 2 gates on qubit 1
        seg1 = schedule["QB1__drive.awg"]
        assert len(seg1) == 4
        assert isinstance(seg1[0], Block)
        assert isinstance(seg1[1], IQPulse)
        assert isinstance(seg1[2], IQPulse)
        assert isinstance(seg1[3], Block)

        # 1 gate on qubit 2
        seg2 = schedule["QB2__drive.awg"]
        assert len(seg2) == 4
        assert isinstance(seg2[0], Block)
        assert isinstance(seg2[1], IQPulse)
        assert isinstance(seg2[2], Wait)
        assert isinstance(seg2[3], Block)

        # 2x QB1, QB2 (flux, drive), 2x QB3, QB4 (flux, drive, boundary qubits),
        # 2x PL_1, PL_2 (RO, TWPA), 1x TC-1-3, TC-1-4, TC_2_3, TC_2_4 (flux, boundary coupler)
        # 1x common options
        assert len(settings.subtrees) == 17

        assert isinstance(readout_mappings, tuple)
        assert len(readout_mappings) == 1
        assert isinstance(readout_mappings[0], dict)
        assert len(readout_mappings[0]) == 2
        assert readout_mappings[0]["m2"] == ("QB2__m1",)
        assert readout_mappings[0]["m1"] == ("QB1__m2",)

        assert len(metrics) == 1
        assert metrics[0].components == {"QB1", "QB2"}
        assert metrics[0].component_pairs_with_gates == set()
        assert metrics[0].gate_loci["prx"]["drag_gaussian"][("QB1",)] == 2
        assert metrics[0].gate_loci["prx"]["drag_gaussian"][("QB2",)] == 1

    def test_single_cz_gate(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit = Circuit(
            name="two qubits with cz",
            instructions=(
                I(name="cz", locus=("A", "B"), args={}),
                I(name="measure", locus=("A",), args={"key": "m1"}),
                I(name="measure", locus=("B",), args={"key": "m2"}),
            ),
        )
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB3"},
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedules = context["schedules"]
        schedule = schedules[0]
        metrics = context["circuit_metrics"]

        schedule = schedules[0]
        _check_schedule(
            schedule, ["QB1__drive.awg", "QB3__drive.awg", "TC-1-3__flux.awg", "PL_1__readout", "PL_2__readout"]
        )

        assert len(schedule["QB1__drive.awg"]) == 3
        assert len(schedule["QB3__drive.awg"]) == 3
        assert isinstance(schedule["QB1__drive.awg"][0], Block)
        assert isinstance(schedule["QB1__drive.awg"][1], VirtualRZ)
        assert isinstance(schedule["QB3__drive.awg"][0], Block)
        assert isinstance(schedule["QB3__drive.awg"][1], VirtualRZ)
        assert all(isinstance(pulse, RealPulse) for pulse in schedule["TC-1-3__flux.awg"][1:-1])
        assert isinstance(schedule["TC-1-3__flux.awg"][-1], Wait)

        # 2x QB1, QB3 (flux, drive), 2x QB2, QB4 (flux, drive, boundary qubits),
        # 2x PL_1, PL_2 (RO, TWPA),  1x TC-1-3 (flux), 1x TC-1-4, TC_2_3 (flux, boundary coupler)
        # 1x common options
        assert len(settings.subtrees) == 16

        assert len(metrics) == 1
        assert metrics[0].components == {"QB1", "QB3"}
        assert metrics[0].component_pairs_with_gates == {("QB1", "QB3")}

    def test_one_qubit_gates_and_barrier_on_two_qubits(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit = Circuit(
            name="two qubits and barrier",
            instructions=(
                I(name="prx", locus=("A",), args={"angle": 0.5, "phase": 0.1}),
                I(name="barrier", locus=("A", "B"), args={}),
                I(name="prx", locus=("B",), args={"angle": 0.5, "phase": 0.1}),
                I(name="measure", locus=("A",), args={"key": "m1"}),
                I(name="measure", locus=("B",), args={"key": "m2"}),
            ),
        )
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB2"},
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])

        schedule = context["schedules"][0]
        metrics = context["circuit_metrics"]

        _check_schedule(schedule, ["QB1__drive.awg", "QB2__drive.awg", "PL_1__readout", "PL_2__readout"])

        seg1 = schedule["QB1__drive.awg"]
        seg2 = schedule["QB2__drive.awg"]
        assert len(seg1) == 5
        assert len(seg2) == 5

        # QB2 gate starts when QB1 gate finishes due to the barrier
        assert isinstance(seg1[0], Block)
        assert isinstance(seg1[1], IQPulse)
        assert isinstance(seg1[2], Block)
        assert isinstance(seg1[3], Wait)
        assert isinstance(seg1[4], Block)
        assert isinstance(seg2[0], Block)
        assert isinstance(seg2[1], Wait)
        assert isinstance(seg2[2], Block)
        assert isinstance(seg2[3], IQPulse)
        assert isinstance(seg2[4], Block)

        assert len(metrics) == 1
        assert metrics[0].components == {"QB1", "QB2"}
        assert metrics[0].component_pairs_with_gates == set()

    def test_measurement_with_heralding(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit = Circuit(
            name="two qubits with multiple measurements",
            instructions=(
                I(name="prx", locus=("A",), args={"angle": 0.5, "phase": 0.1}),
                I(name="prx", locus=("B",), args={"angle": 0.5, "phase": 0.1}),
                I(name="measure", locus=("A", "B"), args={"key": "m5"}),
            ),
        )
        options = default_options_generator.from_defaults(heralding_mode=HeraldingMode.ZEROS)
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB2"},
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])

        schedules = context["schedules"]
        readout_mappings = context["readout_mappings"]
        schedule = schedules[0]

        assert len(schedules) == 1
        _check_schedule(schedules[0], ["QB1__drive.awg", "QB2__drive.awg", "PL_1__readout", "PL_2__readout"])
        assert len(schedule["PL_1__readout"]) == 5
        assert isinstance(schedule["PL_1__readout"][0], Wait)
        assert isinstance(schedule["PL_1__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][2], Wait)
        assert isinstance(schedule["PL_1__readout"][3], Wait)
        assert isinstance(schedule["PL_1__readout"][4], ReadoutTrigger)
        assert len(schedule["PL_2__readout"]) == 5
        assert isinstance(schedule["PL_2__readout"][0], Wait)
        assert isinstance(schedule["PL_2__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][2], Wait)
        assert isinstance(schedule["PL_2__readout"][3], Wait)
        assert isinstance(schedule["PL_2__readout"][4], ReadoutTrigger)
        assert schedule["PL_1__readout"][1].acquisitions[0].label == "QB1____HERALD"

        assert len(readout_mappings) == 1
        assert set(readout_mappings[0].keys()) == {"m5", "__HERALD"}
        assert readout_mappings[0]["m5"] == ("QB1__m1", "QB2__m1")

    def test_compilation_with_dd(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        """Test circuit compilation with Dynamical Decoupling enabled succeeds."""
        circuit = Circuit(
            name="three qubits with multiple measurements",
            instructions=(
                I(name="prx", locus=("A",), args={"angle": 0.5 * pi, "phase": 1.5 * pi}, implementation=None),
                I(name="barrier", locus=("A", "B", "C"), args={}, implementation=None),
                I(name="prx", locus=("A",), args={"angle": 0.5 * pi, "phase": 2.5 * pi}, implementation=None),
                I(name="barrier", locus=("A", "B", "C"), args={}, implementation=None),
                I(name="cz", locus=("B", "C"), args={}, implementation=None),
                I(name="barrier", locus=("A", "B", "C"), args={}, implementation=None),
                I(name="cz", locus=("B", "C"), args={}, implementation=None),
                I(name="barrier", locus=("A", "B", "C"), args={}, implementation=None),
                I(name="prx", locus=("A",), args={"angle": 0.5 * pi, "phase": 4.5 * pi}, implementation=None),
                I(name="prx", locus=("B",), args={"angle": pi, "phase": 0.0}, implementation=None),
                I(name="prx", locus=("C",), args={"angle": pi, "phase": 0.0}, implementation=None),
                I(name="barrier", locus=("A", "B", "C"), args={}, implementation=None),
                I(name="measure", locus=("A",), args={"key": "meas_3_0_0"}, implementation=None),
                I(name="measure", locus=("B",), args={"key": "meas_3_0_1"}, implementation=None),
                I(name="measure", locus=("C",), args={"key": "meas_3_0_2"}, implementation=None),
            ),
        )
        options = default_options_generator.from_defaults(
            dd_mode=DDMode.ENABLED,
            dd_strategy=DDStrategy(
                gate_sequences=[(9, "XYXYYXYX", "asap"), (5, "YXYX", "asap"), (2, "XX", "center")],
                skip_leading_wait=False,
                skip_trailing_wait=False,
                target_qubits=frozenset(["QB1", "QB2", "QB3", "QB5"]),  # QB5 is not used in the circuit, ignored
            ),
        )
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB2", "C": "QB3"},
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])

        schedules = context["schedules"]
        assert len(schedules) == 1
        schedule = schedules[0]
        assert len(schedule) == 6  # drive for QB1, QB2, QB3, flux for TC-2-3, both probe lines

        ch = schedule["QB1__drive.awg"]
        # initial reset_wait must be intact
        assert ch[0] == Block(598000)
        # final wait for the measurement must be intact
        assert ch[-1] == Block(2208)

        # DD was applied in one Wait slot
        assert len(ch) == 13
        # "XX" sequence was used here
        assert ch[5] == Wait(32)
        assert isinstance(ch[6], IQPulse)
        assert ch[7] == Wait(80)
        assert isinstance(ch[8], IQPulse)
        assert ch[9] == Wait(48)

    @pytest.mark.parametrize(
        "qubit,target_qubits,error",
        [
            ("QB1", ["QB1", "QB2", "FAKE"], "Unknown target components ['FAKE']"),
        ],
    )
    def test_compilation_with_dd_errors(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options_generator,
        qubit,
        target_qubits,
        error,
    ):
        """DD pass raises an error when input is incorrect."""
        circuit = Circuit(
            name="fsd",
            instructions=(
                I(name="prx", locus=(qubit,), args={"angle": 0.5 * pi, "phase": 1.5 * pi}, implementation=None),
                I(name="measure", locus=(qubit,), args={"key": "meas_3_0_0"}, implementation=None),
            ),
        )
        options = default_options_generator.from_defaults(
            dd_mode=DDMode.ENABLED,
            dd_strategy=DDStrategy(
                gate_sequences=[(2, "XX", "center")],
                target_qubits=frozenset(target_qubits),
            ),
        )
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            options=options,
            stages=get_standard_stages(),
        )
        with pytest.raises(
            RuntimeError,
            match=re.escape('Error in stage "dynamical_decoupling" pass "apply_dd_strategy": ' + error),
        ):
            compiler.compile([circuit])

    def test_mid_circuit_measurements(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit = Circuit(
            name="two qubits with multiple measurements",
            instructions=(
                I(name="prx", locus=("A",), args={"angle": 0.5, "phase": 0.1}),
                I(name="prx", locus=("B",), args={"angle": 0.5, "phase": 0.1}),
                I(name="measure", locus=("A",), args={"key": "m1"}),
                I(name="measure", locus=("B",), args={"key": "m2"}),
                I(name="measure", locus=("A",), args={"key": "m3"}),
                I(name="measure", locus=("B",), args={"key": "m4"}),
                I(name="measure", locus=("A", "B"), args={"key": "m5"}),
            ),
        )
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB2"},
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])

        schedules = context["schedules"]
        schedule = schedules[0]
        readout_mappings = context["readout_mappings"]

        assert len(schedules) == 1
        _check_schedule(schedule, ["QB1__drive.awg", "QB2__drive.awg", "PL_1__readout", "PL_2__readout"])
        assert len(schedule["PL_1__readout"]) == 4
        assert isinstance(schedule["PL_1__readout"][0], Wait)
        assert isinstance(schedule["PL_2__readout"][0], Wait)
        assert isinstance(schedule["PL_1__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][2], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][3], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][2], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][3], ReadoutTrigger)

        assert len(readout_mappings) == 1
        assert set(readout_mappings[0].keys()) == {"m1", "m2", "m3", "m4", "m5"}
        assert readout_mappings[0]["m1"] == ("QB1__m1",)
        assert readout_mappings[0]["m2"] == ("QB2__m2",)
        assert readout_mappings[0]["m3"] == ("QB1__m3",)
        assert readout_mappings[0]["m4"] == ("QB2__m4",)
        assert readout_mappings[0]["m5"] == ("QB1__m5", "QB2__m5")

    def test_mid_circuit_measurements_with_measurement_mode_all(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        """Mid-circuit measurements must not interfere with MeasurementMode.ALL working."""
        circuit = Circuit(
            name="mid-circuit measurements",
            instructions=(
                I(name="prx", locus=("C",), args={"angle": 0.5, "phase": 0.1}),
                I(name="measure", locus=("E",), args={"key": "m1"}),  # not final, E is eclipsed
                I(name="measure", locus=("B", "D"), args={"key": "m2"}),  # not final, B is eclipsed
                I(name="measure", locus=("A", "B"), args={"key": "f1"}),  # final
                I(name="measure", locus=("C",), args={"key": "f2"}),  # final
                I(name="prx", locus=("E",), args={"angle": 0.5, "phase": 0.1}),
            ),
        )
        # compiler should add a final measurement for (D, E), and group them all together
        # into simultaneous ReadoutTriggers

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB2", "C": "QB3", "D": "QB5", "E": "QB4"},
            options=default_options_generator.from_defaults(measurement_mode=MeasurementMode.ALL),
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])

        schedules = context["schedules"]
        schedule = schedules[0]
        readout_mappings = context["readout_mappings"]

        assert len(schedules) == 1
        _check_schedule(schedule, ["QB3__drive.awg", "QB4__drive.awg", "PL_1__readout", "PL_2__readout"])
        assert len(schedule["PL_1__readout"]) == 4
        assert len(schedule["PL_2__readout"]) == 4
        assert isinstance(schedule["PL_1__readout"][0], Wait)
        assert isinstance(schedule["PL_1__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][2], Wait)
        assert isinstance(schedule["PL_1__readout"][3], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][0], Wait)
        assert isinstance(schedule["PL_2__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][2], Wait)
        assert isinstance(schedule["PL_2__readout"][3], ReadoutTrigger)

        # check that final measurements measure all qubits as required by the mode
        assert len(schedule["PL_1__readout"][-1].probe_pulse.entries) == 2
        assert len(schedule["PL_2__readout"][-1].probe_pulse.entries) == 3
        # HACK, we use unique scale_i values to identify the qubits...
        assert [k[0].scale_i for k in schedule["PL_1__readout"][3].probe_pulse.entries] == [0.1, 0.4]
        assert [k[0].scale_i for k in schedule["PL_2__readout"][3].probe_pulse.entries] == [0.2, 0.3, 0.5]

        assert len(readout_mappings) == 1
        assert set(readout_mappings[0].keys()) == {"m1", "m2", "f1", "f2"}
        assert readout_mappings[0]["m1"] == ("QB4__m1",)
        assert readout_mappings[0]["m2"] == ("QB2__m2", "QB5__m2")
        assert readout_mappings[0]["f1"] == ("QB1__m3", "QB2__m3")
        assert readout_mappings[0]["f2"] == ("QB3__m4",)

    def test_mid_circuit_measurements_with_heralding(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit = Circuit(
            name="two qubits with multiple measurements",
            instructions=(
                I(name="prx", locus=("A",), args={"angle": 0.5, "phase": 0.1}),
                I(name="prx", locus=("B",), args={"angle": 0.5, "phase": 0.1}),
                I(name="measure", locus=("A",), args={"key": "m1"}),
                I(name="measure", locus=("B",), args={"key": "m2"}),
                I(name="measure", locus=("A",), args={"key": "m3"}),
                I(name="measure", locus=("B",), args={"key": "m4"}),
                I(name="measure", locus=("A", "B"), args={"key": "m5"}),
            ),
        )
        options = default_options_generator.from_defaults(heralding_mode=HeraldingMode.ZEROS)
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping={"A": "QB1", "B": "QB2"},
            options=options,
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])

        schedules = context["schedules"]
        schedule = schedules[0]
        readout_mappings = context["readout_mappings"]

        assert len(schedules) == 1
        _check_schedule(schedule, ["QB1__drive.awg", "QB2__drive.awg", "PL_1__readout", "PL_2__readout"])
        assert len(schedule["PL_1__readout"]) == 7
        assert isinstance(schedule["PL_1__readout"][0], Wait)
        assert isinstance(schedule["PL_2__readout"][0], Wait)
        assert isinstance(schedule["PL_1__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][1], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][2], Wait)
        assert isinstance(schedule["PL_2__readout"][2], Wait)
        assert isinstance(schedule["PL_1__readout"][3], Wait)
        assert isinstance(schedule["PL_2__readout"][3], Wait)
        assert isinstance(schedule["PL_1__readout"][4], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][4], ReadoutTrigger)
        assert isinstance(schedule["PL_1__readout"][5], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][5], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][6], ReadoutTrigger)
        assert isinstance(schedule["PL_2__readout"][6], ReadoutTrigger)
        assert schedule["PL_1__readout"][1].acquisitions[0].label == "QB1____HERALD"

        assert len(readout_mappings) == 1
        assert set(readout_mappings[0].keys()) == {"__HERALD", "m1", "m2", "m3", "m4", "m5"}
        assert set(readout_mappings[0]["__HERALD"]) == {"QB1____HERALD", "QB2____HERALD"}
        assert readout_mappings[0]["m1"] == ("QB1__m1",)
        assert readout_mappings[0]["m2"] == ("QB2__m2",)
        assert readout_mappings[0]["m3"] == ("QB1__m3",)
        assert readout_mappings[0]["m4"] == ("QB2__m4",)
        assert readout_mappings[0]["m5"] == ("QB1__m5", "QB2__m5")

    def test_multiple_circuits(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        one_qubit_gates_on_one_qubit,
        default_options_generator,
    ):
        """Compile a batch of circuits."""
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options_generator.from_defaults(
                measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
            ),
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([one_qubit_gates_on_one_qubit, one_qubit_gates_on_one_qubit])
        settings, _ = compiler.build_settings(context, shots=self.shots)

        schedules = context["schedules"]
        readout_mappings = context["readout_mappings"]
        metrics = context["circuit_metrics"]

        for schedule in schedules:
            _check_schedule(schedule, ["QB2__drive.awg", "PL_2__readout"])
            assert len(schedule["QB2__drive.awg"]) == 4
            # 2x QB2 (flux, drive), 2x QB3, QB4 (flux, drive, boundary qubits),
            # 2x PL_2,PL_1 (RO, TWPA), 1x TC_2_3, TC_2_4 (flux, boundary coupler)
            # 1x common options
            assert len(settings.subtrees) == 13

        assert isinstance(readout_mappings, tuple)
        assert all(isinstance(readout_mapping, dict) for readout_mapping in readout_mappings)
        assert len(readout_mappings) == 2
        assert len(readout_mappings[0]) == 1
        assert readout_mappings[0]["result"] == ("QB2__m1",)

        assert len(metrics) == 2
        for metric in metrics:
            assert metric.components == {"QB2"}
            assert metric.component_pairs_with_gates == set()

    def test_multiple_circuits_with_different_measurement_configurations(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options
    ):
        """Each circuit in the batch shall measure the same physical qubits, but each circuit can
        have its own grouping of measurement operations and assignment of measurement keys.
        Here we test that such a scenario is indeed properly handled and correct per-circuit
        readout mappings are generated."""
        circuit_1 = Circuit(
            name="two qubits",
            instructions=(
                I(name="measure", locus=("QB1", "QB2"), args={"key": "mk1"}),
                I(name="measure", locus=("QB3",), args={"key": "mk2"}),
            ),
        )
        circuit_2 = Circuit(
            name="two qubits",
            instructions=(
                I(name="measure", locus=("QB2",), args={"key": "result7"}),
                I(name="measure", locus=("QB3", "QB1"), args={"key": "result5"}),
            ),
        )

        _, context = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options,
            stages=get_standard_stages(),
        ).compile([circuit_1, circuit_2])

        readout_mappings = context["readout_mappings"]

        assert isinstance(readout_mappings, tuple)
        assert len(readout_mappings) == 2
        assert all(isinstance(readout_mapping, dict) for readout_mapping in readout_mappings)
        assert len(readout_mappings[0]) == 2
        assert len(readout_mappings[1]) == 2

        assert readout_mappings[0]["mk1"] == ("QB1__m1", "QB2__m1")
        assert readout_mappings[0]["mk2"] == ("QB3__m2",)

        assert readout_mappings[1]["result7"] == ("QB2__m1",)
        assert readout_mappings[1]["result5"] == ("QB3__m2", "QB1__m2")

    def test_circuits_measure_different_qubits(
        self,
        chip_topology,
        channel_properties,
        component_channels,
        calibration_set_values,
        default_options,
    ):
        """Now different circuits in a batch can measure different qubits."""
        loci = [("QB1",), ("QB1", "QB2"), ("QB3",)]
        keys = ["a1", "b1", "c1"]
        circuits = [
            Circuit(
                name=f"circuit {key}",
                instructions=(I(name="measure", locus=locus, args={"key": key}),),
            )
            for locus, key in zip(loci, keys)
        ]
        _, context = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options,
            stages=get_standard_stages(),
        ).compile(circuits)

        metrics = context["circuit_metrics"]

        assert len(metrics) == 3
        for locus, key, met, rm in zip(loci, keys, metrics, context["readout_mappings"]):
            assert met.components == set(locus)
            assert len(rm) == 1
            assert len(rm[key]) == len(locus)

    def test_different_channels_for_circuits_in_batch(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options_generator
    ):
        circuit_one_qubit = Circuit(
            name="single qubit",
            instructions=(
                I(name="prx", locus=("QB1",), args={"angle": 0.1, "phase": 0.1}),
                I(name="measure", locus=("QB1",), args={"key": "result"}),
            ),
        )
        circuit_two_qubits = Circuit(
            name="two qubits",
            instructions=(
                I(name="prx", locus=("QB1",), args={"angle": 0.1, "phase": 0.1}),
                I(name="cz", locus=("QB1", "QB3"), args={}),
                I(name="measure", locus=("QB1",), args={"key": "result"}),
            ),
        )
        options = default_options_generator.from_defaults(
            measurement_mode=MeasurementMode.CIRCUIT, circuit_boundary_mode=CircuitBoundaryMode.NEIGHBOUR
        )

        _, context = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=options,
            stages=get_standard_stages(),
        ).compile([circuit_one_qubit, circuit_two_qubits])

        schedules = context["schedules"]

        assert len(schedules) == 2
        _check_schedule(schedules[0], ["QB1__drive.awg", "PL_1__readout"])
        _check_schedule(schedules[1], ["QB1__drive.awg", "QB3__drive.awg", "TC-1-3__flux.awg", "PL_1__readout"])

    def test_complicated_circuit(
        self, chip_topology, channel_properties, component_channels, calibration_set_values, default_options
    ):
        """Complicated circuit yields the correct circuit metrics."""
        circuit = Circuit(
            name="complicated circuit",
            instructions=(
                I(name="prx", locus=("QB1",), args={"angle": 0.5, "phase": 0.1}),
                I(name="prx", locus=("QB3",), args={"angle": 0.2, "phase": 0.2}),
                I(name="cz", locus=("QB1", "QB3"), args={}),
                I(name="prx", locus=("QB2",), args={"angle": 0.5, "phase": 0.3}),
                I(name="prx", locus=("QB3",), args={"angle": 0.4, "phase": 0.4}),
                I(name="cz", locus=("QB1", "QB3"), args={}),
                I(name="cz", locus=("QB3", "QB2"), args={}),  # cz.drag_gaussian is symmetric, locus will be flipped
                I(name="measure", locus=("QB2", "QB1"), args={"key": "m1"}),
                I(name="measure", locus=("QB3",), args={"key": "m2"}),
            ),
        )
        _, context = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties,
            component_channels=component_channels,
            component_mapping=None,
            options=default_options,
            stages=get_standard_stages(),
        ).compile([circuit])

        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]

        _check_schedule(
            schedule,
            [
                "QB1__drive.awg",
                "QB2__drive.awg",
                "QB3__drive.awg",
                "TC-1-3__flux.awg",
                "TC-2-3__flux.awg",
                "PL_1__readout",
                "PL_2__readout",
            ],
        )
        met = context["circuit_metrics"][0]
        qubits = ["QB1", "QB2", "QB3"]

        assert met.components == set(qubits)
        assert met.component_pairs_with_gates == set([("QB1", "QB3"), ("QB2", "QB3")])

        for q, n in zip(qubits, [1, 1, 2]):
            assert met.gate_loci["prx"]["drag_gaussian"][(q,)] == n

        for locus, n in zip([("QB1", "QB3"), ("QB2", "QB3")], [2, 1]):
            assert met.gate_loci["cz"]["tgss"][locus] == n

        for locus in [("QB2", "QB1"), ("QB3",)]:
            assert met.gate_loci["measure"]["constant"][locus] == 1

        assert readout_mappings[0] == {
            "m1": ("QB2__m1", "QB1__m1"),
            "m2": ("QB3__m2",),
        }

    ### Test the returned settings

    def test_sets_playlist_repeats_according_to_shots(
        self, calibration_set_values, chip_topology, channel_properties, component_channels, default_options
    ):
        schedule_builder = initialize_schedule_builder(
            calibration_set_values,
            chip_topology,
            channel_properties,
            component_channels,
        )
        settings = build_settings(
            shots=self.shots,
            calibration_set_values=calibration_set_values,
            builder=schedule_builder,
            circuit_metrics=[
                CircuitMetrics(
                    components=frozenset(["QB1", "QB3"]),
                    component_pairs_with_gates=frozenset([("QB1", "QB3")]),
                    # gate_loci left empty
                )
            ],
            options=default_options,
        )
        assert settings.options.playlist_repeats.value == self.shots
        assert settings.options.averaging_bins.value == self.shots

    def test_sets_readout_controller_settings(
        self, calibration_set_values, chip_topology, channel_properties, component_channels, default_options
    ):
        """Several readout controller settings are included in built settings, if present in calset."""
        schedule_builder = initialize_schedule_builder(
            calibration_set_values,
            chip_topology,
            channel_properties,
            component_channels,
        )
        settings = build_settings(
            shots=self.shots,
            calibration_set_values=calibration_set_values,
            builder=schedule_builder,
            circuit_metrics=[
                CircuitMetrics(
                    components=frozenset(["QB1"]),
                    component_pairs_with_gates=frozenset(),
                    # gate_loci left empty
                )
            ],
            options=default_options,
        )
        assert settings["PL_1__readout"].center_frequency.value == 3.8e9
        assert settings["PL_1__readout"].trigger_delay.value == 35e-9
        assert settings["PL_1__readout"].attenuation_out.value == 15.0
        assert settings["PL_1__readout"].attenuation_in.value == 20.0

    def test_sets_precompensation_settings(
        self, calibration_set_values, chip_topology, channel_properties, component_channels, default_options
    ):
        schedule_builder = initialize_schedule_builder(
            calibration_set_values,
            chip_topology,
            channel_properties,
            component_channels,
        )

        settings = build_settings(
            shots=self.shots,
            calibration_set_values=calibration_set_values,
            builder=schedule_builder,
            circuit_metrics=[
                CircuitMetrics(
                    components=frozenset(["QB1", "QB3", "QB4"]),
                    component_pairs_with_gates=frozenset(
                        [
                            (
                                "QB1",
                                "QB3",
                            ),
                            (
                                "QB1",
                                "QB4",
                            ),
                        ]
                    ),
                    # gate_loci left empty
                )
            ],
            options=default_options,
        )

        # only this coupler has precompensation settings in the calset
        assert settings["TC-1-3__flux"].awg.precompensation.timeconstants.value == [8.01e-7]
        assert settings["TC-1-3__flux"].awg.precompensation.amplitudes.value == [0.011]

        # If coupler is used in circuit but precompensation settings not available,
        # should silently create settings tree without them.
        assert "TC-1-4__flux" in settings.children
        assert "precompensation" not in settings["TC-1-4__flux"].awg.children
        assert "precompensation" not in settings["TC-1-4__flux"].awg.children

    def test_one_qubit_gates_on_two_qubits_with_active_reset(
        self,
        chip_topology,
        channel_properties_feedback,
        component_channels_feedback,
        calibration_set_values,
        default_options_generator,
    ):
        circuit = Circuit(
            name="two qubits",
            instructions=(
                I(name="prx", locus=("QB1",), args={"angle": 0.5, "phase": 0.1}),
                I(name="prx", locus=("QB1",), args={"angle": 0.2, "phase": 0.2}),
                I(name="prx", locus=("QB2",), args={"angle": 0.5, "phase": 0.3}),
                I(name="measure", locus=("QB2",), args={"key": "m2"}),
                I(name="measure", locus=("QB1",), args={"key": "m1"}),
            ),
        )

        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties_feedback,
            component_channels=component_channels_feedback,
            component_mapping=None,
            options=default_options_generator.from_defaults(
                measurement_mode=MeasurementMode.CIRCUIT, active_reset_cycles=2, convert_terminal_measurements=False
            ),
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])
        schedule = context["schedules"][0]
        readout_mappings = context["readout_mappings"]
        _check_schedule(schedule, ["QB1__drive.awg", "QB2__drive.awg", "PL_1__readout", "PL_2__readout"])

        # 2 gates on qubit 1
        seg1 = schedule["QB1__drive.awg"]
        assert len(seg1) == 9
        assert isinstance(seg1[0], Block)
        assert isinstance(seg1[1], Wait)
        assert isinstance(seg1[2], ConditionalInstruction)
        assert isinstance(seg1[3], Block)
        assert isinstance(seg1[4], Wait)
        assert isinstance(seg1[5], ConditionalInstruction)
        assert isinstance(seg1[6], IQPulse)
        assert isinstance(seg1[7], IQPulse)
        assert isinstance(seg1[8], Block)

        # 1 gate on qubit 2
        seg2 = schedule["QB2__drive.awg"]
        assert len(seg2) == 9
        assert isinstance(seg2[0], Block)
        assert isinstance(seg1[1], Wait)
        assert isinstance(seg2[2], ConditionalInstruction)
        assert isinstance(seg1[3], Block)
        assert isinstance(seg1[4], Wait)
        assert isinstance(seg2[5], ConditionalInstruction)
        assert isinstance(seg2[6], IQPulse)
        assert isinstance(seg2[7], Wait)
        assert isinstance(seg2[8], Block)

        # the measurement used in reset should not enter the readout mappings
        assert len(readout_mappings[0]) == 2
        assert readout_mappings[0]["m2"] == ("QB2__m1",)
        assert readout_mappings[0]["m1"] == ("QB1__m2",)

        # using active reset turns off the conversion (measure_fidelity is not optimized for leakage)
        compiler = Compiler(
            calibration_set_values=calibration_set_values,
            chip_topology=chip_topology,
            channel_properties=channel_properties_feedback,
            component_channels=component_channels_feedback,
            component_mapping=None,
            options=default_options_generator.from_defaults(
                measurement_mode=MeasurementMode.CIRCUIT, active_reset_cycles=2, convert_terminal_measurements=True
            ),
            stages=get_standard_stages(),
        )
        _, context = compiler.compile([circuit])
        assert not context["options"].convert_terminal_measurements
