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
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest

from iqm.pulse.builder import CircuitOperation, build_quantum_ops, load_config, validate_quantum_circuit
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.gates import (
    CZ_GaussianSmoothedSquare,
    FluxPulseGate_TGSS_CRF,
    PRX_DRAGCosineRiseFall,
    PRX_DRAGGaussian,
    register_operation,
)
from iqm.pulse.gates.default_gates import _deprecated_implementations, _deprecated_ops
from iqm.pulse.playlist import ReadoutTrigger, Schedule
from iqm.pulse.playlist.instructions import (
    Block,
    ConditionalInstruction,
    FluxPulse,
    Instruction,
    IQPulse,
    VirtualRZ,
    Wait,
)
from iqm.pulse.playlist.playlist import Playlist
from iqm.pulse.quantum_ops import QuantumOp
from iqm.pulse.timebox import SchedulingAlgorithm, SchedulingStrategy, TimeBox

CO = CircuitOperation  # shorthand


def test_load_config():
    """Test that load_config raises ValueError if the yaml file does not contain experiment"""
    invalid_path = Path(__file__).parents[3] / "resources/validation/invalid.yml"
    with pytest.raises(ValueError, match='The config YAML is missing the "experiment" section.'):
        load_config(str(invalid_path))
    exa_yaml_path = Path(__file__).parents[3] / "resources/experiment.yml"
    assert isinstance(load_config(str(exa_yaml_path)), tuple)


def test_build_quantum_ops():
    """Test building quantum ops table from an operations dict."""

    # ops_with_no_impls = build_quantum_ops({"empty": {"implementations": {}, "params": (), "arity": 0}})
    # assert ops_with_no_impls["empty"].implementations == {}

    op_table = build_quantum_ops(
        {
            "Test_1": {
                "implementations": {
                    "Test_1_Implementation_1": "CZ_GaussianSmoothedSquare",
                    "Test_1_Implementation_2": "CZ_GaussianSmoothedSquare",
                },
                "arity": 1,
                "params": ("a", "b"),
                "symmetric": "false",
                "defaults_for_locus": {("QB1",): "Test_1_Implementation_2"},
            },
            "Test_2": {
                "implementations": {
                    "Test_2_Default": "CZ_GaussianSmoothedSquare",
                },
                "arity": 0,
                "params": (),
                "symmetric": "false",
            },
        }
    )
    assert isinstance(op_table, dict)
    assert "Test_1" in op_table
    assert isinstance(op_table["Test_1"], QuantumOp)
    assert next(iter(op_table["Test_1"].implementations)) == "Test_1_Implementation_1"
    assert op_table["Test_1"].defaults_for_locus == {("QB1",): "Test_1_Implementation_2"}
    assert "Test_2" in op_table
    assert isinstance(op_table["Test_2"], QuantumOp)
    assert next(iter(op_table["Test_2"].implementations)) == "Test_2_Default"

    # change just the default implementations info
    op_table = build_quantum_ops(
        {
            "prx": {
                "implementations": {"drag_gaussian": "PRX_DRAGGaussian", "drag_crf": "PRX_DRAGCosineRiseFall"},
                "defaults_for_locus": {("QB1",): "drag_crf"},
            }
        }
    )
    assert op_table["prx"].defaults_for_locus == {("QB1",): "drag_crf"}
    # just the defaults, assert that deprecated ops/implementations are indeed missing
    op_table = build_quantum_ops({})
    for deprecated_op in _deprecated_ops:
        assert deprecated_op not in op_table
    for op, deprecated_impls in _deprecated_implementations.items():
        if op in op_table:
            for deprecated_impl in deprecated_impls:
                assert deprecated_impl not in op_table[op].implementations


def test_build_quantum_ops_errors():
    """Test building quantum ops from an erroneous operations dict."""

    # wrong attribute name
    with pytest.raises(TypeError, match="unexpected keyword argument 'horse'"):
        build_quantum_ops({"my_op": {"horse": 2.0}})

    # trying to change canonical operations's unchangeable attributes
    # TODO eventually should raise an error
    # with pytest.raises(ValueError, match="'prx' is a canonical operation, which means the fields {'arity'}"):
    ops = build_quantum_ops({"prx": {"arity": 2, "implementations": {"drag_gaussian": "PRX_DRAGGaussian"}}})
    assert ops["prx"].arity == 1  # was not changed from the default

    # trying to change the meaning of a canonical implementation name
    with pytest.raises(
        ValueError,
        match="'prx': 'drag_gaussian' is a reserved implementation name that refers to 'PRX_DRAGGaussian'",
    ):
        build_quantum_ops({"prx": {"implementations": {"drag_gaussian": "PRX_DRAGCosineRiseFall"}}})

    # requested impl class name has not been exposed
    with pytest.raises(ValueError, match="Requested implementation class 'unexposed' has not been exposed."):
        build_quantum_ops({"prx": {"implementations": {"drag_gaussian": "unexposed"}}})

    # defaults_for_locus must only refer to existing implementations
    with pytest.raises(
        ValueError, match="'prx': defaults_for_locus\\[\\('QB1',\\)\\] implementation 'unknown' does not appear"
    ):
        build_quantum_ops({"prx": {"defaults_for_locus": {("QB1",): "unknown"}}})


def test_implementation_class_conflict_from_YAML():
    """Tests that an error is raised when the user tries to change the
    waveform corresponding to an existing implementation when loading gate
    definitions from a YAML config file."""

    yml_path = Path(__file__).parents[3] / "resources/experiment_wrong_config.yml"

    with pytest.raises(
        ValueError,
        match="'prx': 'drag_gaussian' is a reserved implementation name that refers to 'PRX_DRAGGaussian'",
    ):
        load_config(yml_path)


def test_schedule_builder_validation(schedule_builder):
    """Test ScheduleBuilder validation"""
    assert schedule_builder.validate_calibration() is None


@pytest.mark.parametrize("component", ["QB1", "QB2", "QB5", "QBX"])
@pytest.mark.parametrize(
    "operation, postfix, fname",
    [
        ("drive", "drive.awg", "get_drive_channel"),
        ("flux", "flux.awg", "get_flux_channel"),
        ("readout", "readout", "get_probe_channel"),
    ],
)
def test_get_component_channel(component, operation, postfix, fname, schedule_builder):
    """Test getting the drive channel of a component"""
    fn = getattr(schedule_builder, fname)
    if component == "QBX":
        if fname == "get_probe_channel":
            regexp = f'probe line not found for component "{component}"'
        else:
            regexp = f'channel not found for component "{component}" and operation "{operation}"'
        with pytest.raises(KeyError, match=regexp):
            fn(component)
    elif fname == "get_probe_channel":
        assert fn(component) == "PL-A__readout"
    else:
        assert fn(component) == f"{component}__{postfix}"


def test_get_calibration(schedule_builder):
    """Test getting calibration fails when using invalid arguments"""
    assert isinstance(schedule_builder.get_calibration("prx", "drag_gaussian", ("QB1",)), dict)

    with pytest.raises(ValueError, match="No calibration data for 'prx.drag_gaussian' at \\('QB0',\\)."):
        schedule_builder.get_calibration("prx", "drag_gaussian", ("QB0",))
    with pytest.raises(ValueError, match="No calibration data for 'prx.unknown'."):
        schedule_builder.get_calibration("prx", "unknown", ("QB1",))
    with pytest.raises(ValueError, match="No calibration data for op 'unknown'."):
        schedule_builder.get_calibration("unknown", "drag_gaussian", ("QB1",))


def test_inject_calibration(schedule_builder):
    partial_calibration = {
        "prx": {"drag_gaussian": {("QB1",): {"amplitude_i": 0.666}, ("QB2",): {}}},
        "cz": {"gaussian_smoothed_square": {("QB1", "QB2"): {"coupler": {"amplitude": 0.111, "rise_time": 0.123}}}},
    }
    prx1 = schedule_builder.get_implementation("prx", ("QB1",), impl_name="drag_gaussian")
    prx2 = schedule_builder.get_implementation("prx", ("QB2",), impl_name="drag_gaussian")
    # assert the implementations are now cached:
    assert schedule_builder._cache["prx"]["drag_gaussian"][("QB1",)] == prx1
    assert schedule_builder._cache["prx"]["drag_gaussian"][("QB2",)] == prx2

    schedule_builder.inject_calibration(partial_calibration)

    # assert the changed prx is now invalidated:
    assert ("QB1",) not in schedule_builder._cache["prx"]["drag_gaussian"]
    # the other implementation did not actually change so it should still be cached:
    assert schedule_builder._cache["prx"]["drag_gaussian"][("QB2",)] == prx2

    # assert correct calibration data after injection:
    assert schedule_builder.calibration["prx"]["drag_gaussian"][("QB1",)]["amplitude_i"] == 0.666
    assert (
        schedule_builder.calibration["cz"]["gaussian_smoothed_square"][("QB1", "QB2")]["coupler"]["amplitude"] == 0.111
    )
    assert (
        schedule_builder.calibration["cz"]["gaussian_smoothed_square"][("QB1", "QB2")]["coupler"]["rise_time"] == 0.123
    )

    # assert invalidated factorizable gates:
    measure = schedule_builder.get_implementation("measure", ("QB1", "QB2"), impl_name="constant")
    assert schedule_builder._cache["measure"]["constant"][("QB1", "QB2")] == measure
    measure_calibration = {"measure": {"constant": {("QB1",): {"amplitude_i": 0.8}}}}
    schedule_builder.inject_calibration(measure_calibration)
    assert ("QB1", "QB2") not in schedule_builder._cache["measure"]["constant"]


def test_inject_calibration_with_provided_cache(schedule_builder):
    partial_calibration = {
        "prx": {"drag_gaussian": {("QB1",): {"amplitude_i": 0.666}, ("QB2",): {"amplitude_i": 0.777}}},
    }
    prx1 = schedule_builder.get_implementation("prx", ("QB1",), impl_name="drag_gaussian")
    _ = schedule_builder.get_implementation("prx", ("QB2",), impl_name="drag_gaussian")

    cache = schedule_builder.get_cache(ops=["prx"], implementations=["drag_gaussian"], loci=[("QB1",)])
    schedule_builder.inject_calibration(partial_calibration, cache=cache)
    assert schedule_builder._cache["prx"]["drag_gaussian"][("QB1",)] == prx1
    assert ("QB2",) not in schedule_builder._cache["prx"]["drag_gaussian"]

    # factorizable
    measure = schedule_builder.get_implementation("measure", ("QB1", "QB2"), impl_name="constant")
    measure1 = schedule_builder._cache["measure"]["constant"][("QB1",)]
    cache = schedule_builder.get_cache(ops=["measure"], implementations=["constant"], loci=[("QB1",), ("QB1", "QB2")])
    measure_calibration = {"measure": {"constant": {("QB1",): {"amplitude_i": 0.8}, ("QB2",): {"amplitude_i": 0.888}}}}
    schedule_builder.inject_calibration(measure_calibration, cache=cache)
    assert schedule_builder._cache["measure"]["constant"][("QB1",)] == measure1
    assert schedule_builder._cache["measure"]["constant"][("QB1", "QB2")] == measure
    assert ("QB2",) not in schedule_builder._cache["measure"]["constant"]


def test_get_control_channels(schedule_builder):
    """Test that unknown qubits are ignored when getting control channels for a locus"""
    expected_channels = (
        "QB1__drive.awg",
        "QB1__flux.awg",
        "QB2__drive.awg",
        "QB2__flux.awg",
    )
    assert schedule_builder.get_control_channels(("QB1", "QB2")) == expected_channels
    assert schedule_builder.get_control_channels(("QB1", "QB2", "QBX")) == expected_channels


def test_get_implementation(schedule_builder):  # noqa: PLR0915
    """Test getting implementation by op and name (full circuit execution behavior)"""

    # request an unknown operation
    with pytest.raises(ValueError, match="Unknown quantum operation 'yyy'"):
        schedule_builder.get_implementation("yyy", ("QB1",))

    # no implementation requested, will return the default one
    implementation = schedule_builder.get_implementation("prx", ("QB1",))
    assert isinstance(implementation, PRX_DRAGGaussian)
    assert implementation.pulse.scale_i == 4.65642
    assert implementation.pulse.scale_q == -0.1214

    # default one does not have cal data for this locus
    with pytest.raises(ValueError, match="No calibration data for 'prx.drag_gaussian' at \\('QB4',\\)."):
        schedule_builder.get_implementation("prx", ("QB4",))

    # but the second one does
    implementation = schedule_builder.get_implementation("prx", ("QB4",), "drag_crf")
    assert isinstance(implementation, PRX_DRAGCosineRiseFall)
    assert implementation.pulse.scale_i == 0.49
    assert implementation.pulse.scale_q == 0.5

    # request a specific implementation
    implementation = schedule_builder.get_implementation("prx", ("QB1",), "drag_gaussian")
    assert isinstance(implementation, PRX_DRAGGaussian)
    assert implementation.pulse.scale_i == 4.65642
    assert implementation.pulse.scale_q == -0.1214

    # which does not have cal data
    with pytest.raises(ValueError, match="No calibration data for 'prx.drag_gaussian' at \\('QB4',\\)."):
        schedule_builder.get_implementation("prx", ("QB4",), "drag_gaussian")

    # request an unknown implementation
    with pytest.raises(ValueError, match="Unknown quantum operation implementation 'prx.xxx'"):
        schedule_builder.get_implementation("prx", ("QB1",), "xxx")

    # use priority calibration
    priority_calibration = {"amplitude_i": 0.1}
    implementation = schedule_builder.get_implementation("prx", ("QB1",), priority_calibration=priority_calibration)
    assert isinstance(implementation, PRX_DRAGGaussian)
    assert implementation.pulse.scale_i == 0.1
    assert implementation.pulse.scale_q == -0.1214

    ## for multi-qubit gates the locus order may also be changed depending on the available calibration data

    # no implementation requested, will return the first (default) one
    implementation = schedule_builder.get_implementation("cz", ("QB1", "QB2"))
    assert isinstance(implementation, CZ_GaussianSmoothedSquare)
    assert implementation.locus == ("QB1", "QB2")
    assert implementation._schedule["TC-1-2__flux.awg"][0].scale == -0.086
    # symmetric gate and implementation, locus is always sorted
    implementation2 = schedule_builder.get_implementation("cz", ("QB2", "QB1"))
    assert implementation2 is implementation

    # default one does not have cal data for this locus
    with pytest.raises(
        ValueError, match="No calibration data for 'cz.gaussian_smoothed_square' at \\('QB2', 'QB5'\\)."
    ):
        schedule_builder.get_implementation("cz", ("QB2", "QB5"))

    # symmetric gate, non-symmetric implementation, locus order depends on cal data availability
    implementation = schedule_builder.get_implementation("cz", ("QB1", "QB2"), "tgss_crf")
    assert isinstance(implementation, FluxPulseGate_TGSS_CRF)
    assert implementation.locus == ("QB2", "QB1")
    assert implementation._schedule["TC-1-2__flux.awg"][0].scale == -0.11

    # if both locus orders have cal data, the requested one is used
    implementation = schedule_builder.get_implementation("cz", ("QB2", "QB5"), "tgss_crf")
    assert isinstance(implementation, FluxPulseGate_TGSS_CRF)
    assert implementation.locus == ("QB2", "QB5")
    assert implementation._schedule["TC-2-5__flux.awg"][0].scale == -0.12
    assert "QB2__flux.awg" in implementation._schedule

    # reverse order
    implementation = schedule_builder.get_implementation("cz", ("QB5", "QB2"), "tgss_crf")
    assert isinstance(implementation, FluxPulseGate_TGSS_CRF)
    assert implementation.locus == ("QB5", "QB2")
    assert implementation._schedule["TC-2-5__flux.awg"][0].scale == -0.13
    assert "QB5__flux.awg" in implementation._schedule

    # test defaults_for_locus (in experiment.yml)
    # tgss_crf is not symmetric, so chosen locus order is not lexicographic
    implementation = schedule_builder.get_implementation("cz", ("QB4", "QB5"))
    assert isinstance(implementation, FluxPulseGate_TGSS_CRF)
    assert implementation.locus == ("QB5", "QB4")
    assert implementation._schedule["TC-4-5__flux.awg"][0].scale == -0.15
    assert "QB5__flux.awg" in implementation._schedule
    # same cached implementation
    implementation2 = schedule_builder.get_implementation("cz", ("QB5", "QB4"))
    assert implementation2 is implementation

    # strict_locus prevents changing the locus order
    with pytest.raises(ValueError, match="No calibration data for 'cz.tgss_crf' at \\('QB4', 'QB5'\\)."):
        schedule_builder.get_implementation("cz", ("QB4", "QB5"), strict_locus=True)

    # set locus-specific defaults
    schedule_builder.op_table["prx"].set_default_implementation_for_locus("drag_crf", ["QB1"])

    # locus-specific default is returned when it has calibration
    implementation = schedule_builder.get_implementation("prx", ("QB1",))
    assert isinstance(implementation, PRX_DRAGCosineRiseFall)

    # use global default when no locus-specific default exists
    implementation = schedule_builder.get_implementation("prx", ("QB2",))
    assert isinstance(implementation, PRX_DRAGGaussian)

    # priority order skips special implementations
    class DummySpecial(GateImplementation):
        """This can be called on any locus as it requires no calibration data but it is special"""

        special_implementation = True

    with pytest.raises(ValueError, match="dummy: a special implementation 'iii' cannot be set as a default"):
        register_operation(schedule_builder.op_table, QuantumOp("dummy", 1, implementations={"iii": DummySpecial}))


def test_composite_gate_caching(schedule_builder):
    schedule_builder.u(("QB1",))(1.0, 1.0, 1.0)
    assert len(schedule_builder.composite_cache._cache) == 1
    schedule_builder.u(("QB2",))(1.0, 1.0, 1.0)
    assert len(schedule_builder.composite_cache._cache) == 2
    schedule_builder.u(("QB1",))(0.0, 0.0, 0.0)
    assert len(schedule_builder.composite_cache._cache) == 3
    schedule_builder.inject_calibration({})  # this flushes the cache even though data is empty
    assert len(schedule_builder.composite_cache._cache) == 0


def test_get_implementation_default(schedule_builder):
    """Testing the behavior required by EXA."""
    # can be retrieved
    impl = schedule_builder.get_implementation("prx", ("QB1",))
    assert impl.name == "drag_gaussian"
    assert impl.locus == ("QB1",)

    # can be changed
    schedule_builder.op_table["prx"].set_default_implementation("drag_crf")
    impl = schedule_builder.get_implementation("prx", ("QB1",))
    assert impl.name == "drag_crf"
    assert impl.locus == ("QB1",)

    # can change the locus order for symmetric gates
    impl = schedule_builder.get_implementation("cz", ("QB1", "QB2"))
    assert impl.name == "gaussian_smoothed_square"
    assert impl.locus == ("QB1", "QB2")
    impl_reversed = schedule_builder.get_implementation("cz", ("QB2", "QB1"))
    assert impl is impl_reversed

    # will raise an error for unknown gates
    with pytest.raises(ValueError, match="Unknown quantum operation 'xxx'"):
        _ = schedule_builder.get_implementation("xxx", ("QB1", "QB2"))

    # get locus-specific default
    schedule_builder.op_table["prx"].set_default_implementation_for_locus("drag_crf", ["QB1"])
    implementation = schedule_builder.get_implementation("prx", ("QB1",))
    assert isinstance(implementation, PRX_DRAGCosineRiseFall)


def test_get_implementation_class(schedule_builder):
    """Test getting implementation class by op_name and implementation name"""
    assert schedule_builder.get_implementation_class("prx", "drag_gaussian") == PRX_DRAGGaussian
    assert schedule_builder.get_implementation_class("cz") == CZ_GaussianSmoothedSquare


def test_get_implementation_shortcut_methods(schedule_builder):
    without_shortcut = schedule_builder.get_implementation("prx", ("QB1",))
    with_shortcut = schedule_builder.prx(("QB1",))
    with_getitem = schedule_builder["prx"](("QB1",))
    assert without_shortcut == with_shortcut
    assert without_shortcut == with_getitem

    without_shortcut = schedule_builder.get_implementation(
        "prx", ("QB1",), "drag_gaussian", priority_calibration={"amplitude_i": 0.1}
    )
    with_shortcut = schedule_builder.prx(("QB1",), "drag_gaussian", priority_calibration={"amplitude_i": 0.1})
    with_getitem = schedule_builder["prx"](("QB1",), "drag_gaussian", priority_calibration={"amplitude_i": 0.1})
    assert without_shortcut.pulse == with_shortcut.pulse
    assert without_shortcut.pulse == with_getitem.pulse

    without_shortcut = schedule_builder.get_implementation("measure", ("QB1",))
    with_shortcut = schedule_builder.measure(("QB1",))
    with_getitem = schedule_builder["measure"](("QB1",))
    assert without_shortcut == with_shortcut
    assert without_shortcut == with_getitem

    without_shortcut = schedule_builder.get_implementation("cz", ("QB1", "QB2"))
    with_shortcut = schedule_builder.cz(("QB1", "QB2"))
    with_getitem = schedule_builder["cz"](("QB1", "QB2"))
    assert without_shortcut == with_shortcut
    assert without_shortcut == with_getitem


def test_serial_gates(schedule_builder):
    """Single-qubit gates in series."""
    prx = schedule_builder.get_implementation("prx", ["QB1"])
    box = TimeBox.composite([prx(angle=0.3, phase=0.1), prx(angle=-0.7, phase=0.2)], label="my stuff")
    assert box.validate() is None
    schedule = schedule_builder.timebox_to_schedule(box).cleanup()

    assert len(schedule) == 1
    assert "QB1__drive.awg" in schedule
    assert len(schedule["QB1__drive.awg"]) == 2
    assert schedule.duration == 160
    assert schedule_builder.channels["QB1__drive.awg"].duration_to_seconds(schedule.duration) == pytest.approx(80e-9)


def test_empty_box_is_ok(schedule_builder):
    """No gates."""
    box = TimeBox.composite([])
    assert box.validate() is None

    schedule = schedule_builder.timebox_to_schedule(box).cleanup()
    assert len(schedule) == 0


def test_serial_gates_with_neighbourhood(schedule_builder):
    """Single-qubit gates in series."""
    prx = schedule_builder.get_implementation("prx", ["QB1"])
    box = TimeBox.composite([prx(angle=0.3, phase=0.1), prx(angle=-0.7, phase=0.2)], label="my stuff")
    assert box.validate() is None
    schedule = schedule_builder.timebox_to_schedule(box, neighborhood=2).cleanup()

    assert len(schedule) == 1
    assert "QB1__drive.awg" in schedule
    assert len(schedule["QB1__drive.awg"]) == 2
    assert schedule.duration == 160
    assert schedule_builder.channels["QB1__drive.awg"].duration_to_seconds(schedule.duration) == pytest.approx(80e-9)


def test_parallel_gates(schedule_builder):
    """Single-qubit gates in parallel."""
    prx1 = schedule_builder.get_implementation("prx", ["QB1"])
    prx2 = schedule_builder.get_implementation("prx", ["QB2"])
    box = TimeBox.composite([prx1(angle=0.3, phase=0.1), prx2(angle=-0.7, phase=0.2)], label="my stuff")
    assert box.validate() is None
    schedule = schedule_builder.timebox_to_schedule(box).cleanup()

    assert len(schedule) == 2
    assert "QB1__drive.awg" in schedule
    assert "QB2__drive.awg" in schedule
    assert len(schedule["QB1__drive.awg"]) == 1
    assert len(schedule["QB2__drive.awg"]) == 1
    assert schedule.duration == 80
    assert schedule_builder.channels["QB1__drive.awg"].duration_to_seconds(schedule.duration) == pytest.approx(40e-9)


def test_multi_qubit_gates(schedule_builder):
    cz = schedule_builder.get_implementation("cz", ("QB1", "QB2"))
    box = cz()
    schedule = schedule_builder.timebox_to_schedule(box).cleanup()

    assert len(schedule) == 3  # one neighbor qubit also included for this locus
    assert schedule.duration == 240
    assert schedule_builder.channels["QB1__drive.awg"].duration_to_seconds(schedule.duration) == pytest.approx(120e-9)
    for ch in ["TC-1-2__flux.awg", "QB1__drive.awg", "QB2__drive.awg"]:
        assert ch in schedule
        assert len(schedule[ch]) == 1
        inst = schedule[ch][0]
        assert inst.duration == 240  # samples
        if ch == "TC-1-2__flux.awg":
            assert isinstance(inst, FluxPulse)
            assert inst.rzs == (("QB5__drive.awg", 0.0142),)
        else:
            assert isinstance(inst, VirtualRZ)


prx = CO("prx", ("QB1",), {"angle": 0.0, "phase": 1.0})


@pytest.mark.parametrize(
    "operations, require_measurements, expected_message",
    [
        # Empty circuit
        ([], True, "Circuit contains no measurements."),
        # Valid PRX but no measurements
        ([prx], True, "Circuit contains no measurements."),
        # Valid PRX, measurements not expected
        ([prx], False, None),
        # multiple valid measurements for a single qubit
        (
            [
                CO("measure", ("QB1",), {"key": "m1", "feedback_key": ""}),
                CO("measure", ("QB1",), {"key": "m2", "feedback_key": ""}),
                CO("measure", ("QB1",), {"key": "m3", "feedback_key": ""}),
            ],
            True,
            None,
        ),
        # non-unique measurement keys
        (
            [
                CO("measure", ("QB1",), {"key": "m", "feedback_key": ""}),
                CO("measure", ("QB1",), {"key": "m", "feedback_key": ""}),
            ],
            True,
            "Measurement key 'm' is not unique.",
        ),
        # Unknown operation
        ([CO("unknown", ("QB1",))], False, "Unknown operation 'unknown'."),
        # Invalid implementation specification
        (
            [replace(prx, implementation="")],
            False,
            "Implementation of the instruction should be None, or a non-empty string",
        ),
        # PRX with unknown implementation
        (
            [replace(prx, implementation="unknown")],
            False,
            "Unknown implementation 'unknown' for quantum operation 'prx'.",
        ),
        # invalid locus
        (
            [replace(prx, locus=("QB1", "QB2"))],
            False,
            "The 'prx' operation acts on 1 qubit\\(s\\), but 2 were given: \\('QB1', 'QB2'\\)",
        ),
        (
            [replace(prx, locus=())],
            False,
            "The 'prx' operation acts on 1 qubit\\(s\\), but 0 were given: \\(\\)",
        ),
        (
            [CO("cz", ("QB1",))],
            False,
            "The 'cz' operation acts on 2 qubit\\(s\\), but 1 were given: \\('QB1',\\)",
        ),
        (
            [CO("cz", ())],
            False,
            "The 'cz' operation acts on 2 qubit\\(s\\), but 0 were given: \\(\\)",
        ),
        # missing args
        (
            [replace(prx, args={"angle": 0.0})],
            False,
            re.escape("The operation 'prx' requires the argument(s) ('angle', 'phase'), but ('angle',) were given."),
        ),
        # unallowed arg
        (
            [
                CO("measure", ("QB1",), {"key": "m", "lock": "m"}),
            ],
            False,
            "The operation 'measure' allows (\\('feedback_key', 'key'\\)) | (\\('key', 'feedback_key'\\)), "
            "but (\\{'key', 'lock'\\}) | (\\{'lock', 'key'\\}) were given",
        ),
        # wrong arg type
        (
            [replace(prx, args={"angle": 0.0, "phase": "0.0"})],
            False,
            re.escape(
                "The argument 'phase' should be of one of the following supported types "
                "(<class 'float'>,), but (<class 'str'>) was given."
            ),
        ),
        # repeated locus components
        (
            [CO("cz", ("QB1", "QB1"))],
            False,
            re.escape("Repeated locus components: ('QB1', 'QB1')"),
        ),
    ],
)
def test_circuit_validation(operations, require_measurements, expected_message, schedule_builder):
    """Test that validate_quantum_circuit raises ValueError if the list of operations
    does not make a valid quantum circuit.
    """
    if expected_message is None:
        assert (
            validate_quantum_circuit(operations, schedule_builder.op_table, require_measurements=require_measurements)
            is None
        )
    else:
        with pytest.raises(ValueError, match=expected_message):
            validate_quantum_circuit(operations, schedule_builder.op_table, require_measurements=require_measurements)


def check_wait_box(box: TimeBox, locus: list[str], duration: float) -> None:
    """Helper function for the wait method tests."""
    box.validate()
    assert box.label == "Wait"
    assert box.locus_components == set(locus)
    schedule = box.atom
    assert schedule is not None
    assert len(schedule) == 2 * len(locus)  # 2 channels per qubit
    assert schedule.duration == pytest.approx(duration)
    assert "QB1__drive.awg" in schedule
    for _, seg in schedule.items():
        assert len(seg) == 1
        assert isinstance(seg[0], Block)
        assert seg[0].duration == pytest.approx(duration)


@pytest.mark.parametrize("T", [0.0, 16e-9])
def test_wait(T, schedule_builder):
    """Waiting utility method."""
    locus = ["QB1", "QB2"]
    box = schedule_builder.wait(locus, T)
    check_wait_box(box, locus, schedule_builder.channels["QB1__drive.awg"].duration_to_int_samples(T) if T > 0 else 0)


@pytest.mark.parametrize("T, msg", [(8e-9, "less than 32"), (18e-9, "integer multiple of 16")])
def test_wait_bad_duration(T, msg, schedule_builder):
    locus = ["QB1", "QB2"]
    with pytest.raises(ValueError, match=msg):
        schedule_builder.wait(locus, T)


@pytest.mark.parametrize("T, R", [(18e-9, 16e-9), (21e-9, 24e-9)])
def test_wait_duration_rounding(T, R, schedule_builder):
    """Test the rounding option of the wait method, T should be rounded to R."""
    locus = ["QB1", "QB2"]
    box = schedule_builder.wait(locus, T, rounding=True)
    check_wait_box(box, locus, schedule_builder.channels["QB1__drive.awg"].duration_to_int_samples(R) if R > 0 else 0)


def test_circuit_to_timebox(schedule_builder):
    """Test converting a circuit to a timebox"""
    operations = [
        CircuitOperation(name="prx", implementation=None, locus=("QB1",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="prx", implementation=None, locus=("QB2",), args={"angle": 0.1, "phase": 0.5}),
        CircuitOperation(name="prx", implementation=None, locus=("QB5",), args={"angle": 0.7, "phase": 0.2}),
        CircuitOperation(name="measure", implementation=None, locus=("QB1", "QB2", "QB5")),
    ]
    timebox = schedule_builder.circuit_to_timebox(circuit=operations, name="test-circuit")
    assert isinstance(timebox, TimeBox)
    assert timebox.label == "test-circuit"
    assert timebox.locus_components == {"QB1", "QB2", "QB5"}
    assert len(timebox.children) == 4  # 3 * PRX + measure all three
    assert timebox.scheduling == SchedulingStrategy.ASAP


def test_timeboxes_to_front_padded_playlist(schedule_builder):
    """Test converting timeboxes to a front-padded playlist"""
    circuit = [
        CircuitOperation(name="prx", implementation=None, locus=("QB1",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="measure", implementation=None, locus=("QB1",)),
    ]
    circuit_long = [
        CircuitOperation(name="prx", implementation=None, locus=("QB1",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="prx", implementation=None, locus=("QB1",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="measure", implementation=None, locus=("QB1",)),
    ]
    boxes = [
        schedule_builder.circuit_to_timebox(circuit=circuit, name="test-circuit-1"),
        schedule_builder.circuit_to_timebox(circuit=circuit, name="test-circuit-2"),
        schedule_builder.circuit_to_timebox(circuit=circuit_long, name="test-circuit-3"),
    ]

    # before front-padding
    schedules = [schedule_builder.resolve_timebox(box, neighborhood=1).cleanup() for box in boxes]
    assert schedules[2].duration > schedules[0].duration

    playlist, readout_metrics = schedule_builder.timeboxes_to_front_padded_playlist(boxes)
    assert isinstance(playlist, Playlist)
    assert len(playlist.segments) == 3
    assert readout_metrics.num_segments == 3
    assert readout_metrics.integration_occurrences == {"QB1__readout.result": [1, 1, 1]}
    assert readout_metrics.timetrace_occurrences == {}
    assert readout_metrics.timetrace_lengths == {}
    assert readout_metrics.implementations == {"QB1__readout.result": {"measure.constant"}}

    # after front-padding
    channels = iter(playlist.channel_descriptions.values())
    first_channel = next(channels)
    sample_rate = first_channel.channel_config.sampling_rate
    assert boxes[2].atom.duration == pytest.approx(boxes[0].atom.duration)
    for channel in channels:
        assert channel.channel_config.sampling_rate == sample_rate


def test_resolve_timebox_hard_boundary(schedule_builder):
    components = ["QB1", "QB2"]
    prx0 = schedule_builder.get_implementation("prx", (components[0],))(0, 0)
    prx1 = schedule_builder.get_implementation("prx", (components[1],))(0, 0)
    short_box = TimeBox.composite([prx0])
    ragged_box = TimeBox.composite([prx0, prx1, prx1])
    composite = TimeBox.composite([ragged_box, short_box])
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)
    assert schedule.pprint() == "QB1__drive.awg    3:!====|    !====|\n" + "QB2__drive.awg    3:!====!====|    |\n"


def test_resolve_timebox_hard_boundary_deals_correctly_with_out_of_locus_pulses(schedule_builder):
    components = ["QB1", "QB2"]
    cz = schedule_builder.get_implementation("cz", components)()
    coupler_pulse = schedule_builder.wait(["TC-1-2"], 4e-8)
    composite = TimeBox.composite(coupler_pulse + cz)
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)
    assert (
        schedule.pprint()
        == "TC-1-2__flux.awg    2:B....F--------------|\n"
        + "QB1__drive.awg      2:|    Z~~~~~~~~~~~~~~|\n"
        + "QB2__drive.awg      2:|    Z~~~~~~~~~~~~~~|\n"
    )


def test_resolve_timebox_hard_boundary_with_neighborhood_1(schedule_builder):
    components = ["QB1", "QB2"]
    prx0 = schedule_builder.get_implementation("prx", (components[0],))(0, 0)
    prx1 = schedule_builder.get_implementation("prx", (components[1],))(0, 0)
    coupler_pulse = schedule_builder.wait(["TC-1-2"], 4e-8)
    composite = TimeBox.composite(prx0 + prx1 | coupler_pulse)
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=1)
    assert (
        schedule.pprint()
        == "QB1__drive.awg      2:!====|    |\n"
        + "QB2__drive.awg      2:!====|    |\n"
        + "TC-1-2__flux.awg    2:|    B....|\n"
    )


def test_resolve_timebox_hard_boundary_with_neighborhood_2(schedule_builder):
    components = ["QB1", "QB2"]
    prx0 = schedule_builder.get_implementation("prx", (components[0],))(0, 0)
    prx1 = schedule_builder.get_implementation("prx", (components[1],))(0, 0)
    composite = TimeBox.composite(prx0 + prx1)
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=2)
    assert schedule.pprint() == "QB1__drive.awg    2:!====|    |\n" + "QB2__drive.awg    2:|    !====|\n"


def test_resolve_timebox_hard_boundary_with_virtual_ff_channels(schedule_builder):
    measure = schedule_builder.measure(("QB5",))(feedback_key="horse")
    prx = schedule_builder.prx(("QB5",))(0, 0)
    cond = schedule_builder.cc_prx(("QB5",))(0, 0, feedback_qubit="QB5", feedback_key="horse")

    # this should add the full feedback delay before the conditional prx
    composite = measure + cond
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)
    inst = schedule["QB5__drive.awg"][0]
    assert isinstance(inst, Block)
    assert inst.duration == measure.children[0].atom.duration
    inst = schedule["QB5__drive.awg"][1]
    assert isinstance(inst, Wait)
    assert inst.duration == 200  # 2.0e9 sample rate, 100e-9 delay => 200 samples
    assert isinstance(schedule["QB5__drive.awg"][2], ConditionalInstruction)

    # this should add the difference of the calibrated ff delay and prx duration after the prx
    composite = measure + prx + cond
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)
    inst = schedule["QB5__drive.awg"][0]
    assert isinstance(inst, Block)
    assert inst.duration == measure.children[0].atom.duration
    assert isinstance(schedule["QB5__drive.awg"][1], IQPulse)
    inst = schedule["QB5__drive.awg"][2]
    assert isinstance(inst, Wait)
    # prx duration = 40e-9 = 80 samples => needed extra wait = 200 - 80 = 120 samples
    assert inst.duration == 120
    assert isinstance(schedule["QB5__drive.awg"][3], ConditionalInstruction)


def test_resolve_timebox_hard_boundary_with_virtual_ff_channels_and_shelved_measure(schedule_builder):
    measure = schedule_builder.measure_fidelity(("QB5",), impl_name="shelved_constant")(feedback_key="horse")
    cond = schedule_builder.cc_prx(("QB5",))(0, 0, feedback_qubit="QB5", feedback_key="horse")
    composite = measure + cond
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)
    inst = schedule["QB5__drive.awg"][1]
    assert isinstance(inst, Block)
    assert inst.duration == schedule["PL-A__readout"][1].duration
    assert isinstance(schedule["QB5__drive.awg"][2], IQPulse)
    inst = schedule["QB5__drive.awg"][3]
    assert isinstance(inst, Wait)
    assert inst.duration == 200
    assert isinstance(schedule["QB5__drive.awg"][4], ConditionalInstruction)


def test_resolve_timebox_hard_boundary_with_fast_measure(schedule_builder):
    measure_fast = schedule_builder.get_implementation("measure", ("QB1", "QB2", "QB5"), impl_name="fast_constant")(
        feedback_key="horse"
    )
    measure = schedule_builder.get_implementation("measure", ("QB1", "QB2", "QB5"))()
    cond5 = schedule_builder.get_implementation("cc_prx", ("QB5",))(0, 0, feedback_qubit="QB5", feedback_key="horse")
    prx1 = schedule_builder.get_implementation("prx", ("QB1",))(0, 0)

    composite = TimeBox.composite([measure_fast, prx1, cond5, measure], scheduling=SchedulingStrategy.ASAP)
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)

    # probe channel -- measurements should have Wait between them
    probe_channel = schedule["PL-A__readout"]
    assert isinstance(probe_channel[0], ReadoutTrigger)
    assert probe_channel[0].duration == 16 + 1200 + 160  # int deadtime + probe length + locus deadtime
    assert isinstance(probe_channel[1], Block)
    # acq delay + int length + stop deadtime - phys.probe duration
    assert probe_channel[1].duration == 1200 + 1200 + 16 - 1376
    assert isinstance(probe_channel[2], Wait)
    # control_delay + cc_prx.duration
    assert probe_channel[2].duration == 280
    assert isinstance(probe_channel[3], ReadoutTrigger)

    drive_channel1 = schedule["QB1__drive.awg"]
    # QB1 drive should play PRX immediately after the probe pulse
    assert isinstance(drive_channel1[0], Block)
    assert drive_channel1[0].duration == 1376
    assert isinstance(drive_channel1[1], IQPulse)

    drive_channel5 = schedule["QB5__drive.awg"]
    # QB5 cc_prx must wait for integration and control_delay
    assert isinstance(drive_channel5[0], Block)
    assert drive_channel5[0].duration == 1376
    assert isinstance(drive_channel5[1], Wait)
    assert drive_channel5[1].duration == 1200 + 1200 + 16 - 1376 + 200
    assert isinstance(drive_channel5[2], ConditionalInstruction)
    # test that time_trace call works
    time_trace = schedule_builder.get_implementation(
        "measure", ("QB1", "QB2", "QB5"), impl_name="fast_constant"
    ).time_trace()
    assert isinstance(time_trace, TimeBox)


def test_resolve_timebox_hard_boundary_with_variable_sampling_rates(schedule_builder_uhfqa):
    """Test the in seconds scheduling."""
    components = ["QB1", "QB2"]
    prx1_gate = schedule_builder_uhfqa.get_implementation("prx", (components[0],))
    prx2_gate = schedule_builder_uhfqa.get_implementation("prx", (components[1],))
    measure_gate = schedule_builder_uhfqa.get_implementation("measure", tuple(components))
    prx_channel = schedule_builder_uhfqa.channels["QB1__drive.awg"]
    meas_channel = schedule_builder_uhfqa.channels["PL-A__readout"]

    composite = prx1_gate(0, 0) + prx2_gate(0, 0) + measure_gate()
    schedule = schedule_builder_uhfqa.resolve_timebox(composite, neighborhood=0)

    drive_seg = schedule["QB1__drive.awg"]
    probe_seg = schedule["PL-A__readout"]

    assert len(drive_seg) == 2  # iqpulse, block
    assert len(probe_seg) == 2  # wait, readout trigger

    # prx pulse is unchanged
    prx_duration_in_s = prx1_gate.duration_in_seconds()
    assert drive_seg[0].duration == prx_channel.duration_to_samples(prx_duration_in_s)

    # probe channel wait duration is prx duration rounded up to the granularity of probe channel
    probe_wait_duration_in_s = meas_channel.round_duration_to_granularity(prx_duration_in_s, round_up=True)
    assert probe_seg[0].duration == meas_channel.duration_to_samples(probe_wait_duration_in_s)
    assert probe_wait_duration_in_s > prx_duration_in_s

    # drive channel block duration (being rounded up) determines the measure gate schedule duration
    measure_duration_in_s = measure_gate.duration_in_seconds()
    ro_trigger_duration_in_s = meas_channel.duration_to_seconds(probe_seg[1].duration)
    assert measure_duration_in_s > ro_trigger_duration_in_s
    assert drive_seg[1].duration == prx_channel.duration_to_int_samples(measure_duration_in_s)
    assert drive_seg[1].duration == prx_channel.duration_to_int_samples(
        prx_channel.round_duration_to_granularity(ro_trigger_duration_in_s, round_up=True, force_min_duration=True),
    )


def test_resolve_timebox_with_variable_granularity(schedule_builder_granularity):
    wait = schedule_builder_granularity.wait(("QB1",), 44e-9)  # this does not conform to 16 samples
    measure = schedule_builder_granularity.measure(("QB1",))()
    composite = wait + measure
    schedule = schedule_builder_granularity.resolve_timebox(composite, neighborhood=0)
    probe_seg = schedule["PL-A__readout"]
    assert probe_seg[0].duration == 96  # rounded up to the next possible multiple of 16


@pytest.mark.skip(reason="TETRIS algorithm is not supported until COMP-1281 is done.")
def test_resolve_timebox_tetris(schedule_builder):
    components = ["QB1", "QB2"]
    prx0 = schedule_builder.get_implementation("prx", (components[0],))(0, 0)
    prx1 = schedule_builder.get_implementation("prx", (components[1],))(0, 0)
    short_box = TimeBox.composite([prx0], scheduling_algorithm=SchedulingAlgorithm.TETRIS)
    ragged_box = TimeBox.composite([prx0, prx1, prx1], scheduling_algorithm=SchedulingAlgorithm.TETRIS)
    composite = TimeBox.composite([ragged_box, short_box], scheduling_algorithm=SchedulingAlgorithm.TETRIS)
    schedule = schedule_builder.resolve_timebox(composite, neighborhood=0)
    assert (
        schedule.pprint()
        == "QB1__drive.awg    2:!====!====|\n"
        + "QB1__flux.awg     2:?    ?    |\n"
        + "QB1__readout      2:?    ?    |\n"
        + "QB2__drive.awg    2:!====!====|\n"
        + "QB2__flux.awg     2:?    ?    |\n"
        + "QB2__readout      2:?    ?    |\n"
    )


def test_timeboxes_to_playlist(schedule_builder):
    """Test converting a list of timeboxes to a playlist"""
    operations = [
        CircuitOperation(name="prx", implementation=None, locus=("QB1",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="prx", implementation=None, locus=("QB2",), args={"angle": 0.1, "phase": 0.5}),
        CircuitOperation(name="measure", implementation=None, locus=("QB1", "QB2")),
    ]
    timebox = schedule_builder.circuit_to_timebox(circuit=operations, name="test-circuit")
    playlist = schedule_builder.timeboxes_to_playlist([timebox])
    assert isinstance(playlist, Playlist)
    assert len(playlist.segments) == 1


def test_build_playlist(schedule_builder_star):
    """Test converting schedules to a playlist."""
    circuit = [
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="cz", locus=("QB1", "COMP_R")),
        CircuitOperation(name="barrier", locus=("QB1", "QB2")),
        CircuitOperation(name="measure", locus=("QB1", "QB2")),
    ]
    boxes = [schedule_builder_star.circuit_to_timebox(circuit, name="test-circuit-1")]
    schedules = [schedule_builder_star.resolve_timebox(box, neighborhood=1).cleanup() for box in boxes]

    # virtual channel is present (due to the cz)
    assert "COMP_R__drive_virtual" in schedules[0]

    playlist, readout_metrics = schedule_builder_star.build_playlist(schedules)
    assert isinstance(playlist, Playlist)
    assert len(playlist.segments) == 1
    assert readout_metrics.num_segments == 1
    assert readout_metrics.integration_occurrences == {"QB1__readout.result": [1], "QB2__readout.result": [1]}
    assert readout_metrics.timetrace_occurrences == {}
    assert readout_metrics.timetrace_lengths == {}
    assert readout_metrics.implementations == {
        "QB1__readout.result": {"measure.constant"},
        "QB2__readout.result": {"measure.constant"},
    }

    # virtual channel has been filtered out
    assert set(playlist.channel_descriptions) == {"QB1__drive.awg", "QB1__flux.awg", "TC1__flux.awg", "PL__readout"}
    assert len(playlist.channel_descriptions["QB1__drive.awg"].instruction_table) == 3  # 0 duration Blocks removed
    assert len(playlist.channel_descriptions["PL__readout"].instruction_table) == 5  # Waits merged


def test_timebox_to_schedule_with_long_distance_vzs(schedule_builder):
    # the default, drag_gaussian, has no calibration for QB4
    schedule_builder.op_table["prx"].set_default_implementation_for_locus("drag_crf", ["QB4"])

    # add long-distance VZs to the calibration
    schedule_builder.calibration["cz"]["gaussian_smoothed_square"][("QB1", "QB2")]["rz"]["QB4"] = 0.01
    schedule_builder.calibration["cz"]["gaussian_smoothed_square"][("QB1", "QB2")]["rz"]["QB5"] = 0.02
    schedule_builder.calibration["cz"]["gaussian_smoothed_square"][("QB2", "QB5")] = {
        "rz": {
            "QB2": 0.0,
            "QB5": 0.1,
            "QB4": 0.001,
        }
    }
    # create the circuit
    circuit = [
        CircuitOperation(name="prx", locus=("QB4",), args={"angle": 0.5, "phase": 0.0}),
        CircuitOperation(name="prx", locus=("QB5",), args={"angle": 0.5, "phase": 0.0}),  # these do not get VZs
        CircuitOperation(name="barrier", locus=("QB1", "QB2", "QB3", "QB4", "QB5")),
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="cz", locus=("QB2", "QB5")),  # the VirtualRZ here gets an added increment from the 1st
        CircuitOperation(name="barrier", locus=("QB1", "QB2", "QB3", "QB4", "QB5")),
        CircuitOperation(name="prx", locus=("QB4",), args={"angle": 0.5, "phase": 0.0}),  # 1 Vzs from both
        CircuitOperation(name="prx", locus=("QB5",), args={"angle": 0.5, "phase": 0.0}),  # No Vzs
        CircuitOperation(name="barrier", locus=("QB1", "QB2", "QB3", "QB4", "QB5")),
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="barrier", locus=("QB1", "QB2", "QB3", "QB4", "QB5")),
        CircuitOperation(name="prx", locus=("QB4",), args={"angle": 0.5, "phase": 0.0}),  # 3 Vzs from the 1st CZ
        CircuitOperation(name="prx", locus=("QB5",), args={"angle": 0.5, "phase": 0.0}),  # 3 Vzs from the 1st CZ
    ]
    box = schedule_builder.circuit_to_timebox(circuit, name="test-circuit-1")
    schedule = schedule_builder.timebox_to_schedule(box, neighborhood=0)
    assert schedule["QB4__drive.awg"][0].phase_increment == 0.0
    assert schedule["QB5__drive.awg"][0].phase_increment == 0.0
    assert round(schedule["QB4__drive.awg"][4].phase_increment, 6) == round(-0.011, 6)
    assert round(schedule["QB5__drive.awg"][3].phase_increment, 6) == round(-0.12, 6)
    assert schedule["QB5__drive.awg"][5].phase_increment == 0.0
    assert round(schedule["QB4__drive.awg"][8].phase_increment, 6) == round(-0.03, 6)
    assert round(schedule["QB5__drive.awg"][9].phase_increment, 6) == round(-0.06, 6)


def test_timeboxes_to_playlist_conditional_instruction_with_too_many_outcomes(schedule_builder):
    """Test error cases in timeboxes_to_playlist"""
    schedule = Schedule()
    schedule.add_channels(["QB1__drive.awg"])
    schedule.append(
        "QB1__drive.awg",
        ConditionalInstruction(
            duration=20,
            condition="cond",
            outcomes=(Wait(20), Wait(15), Wait(10)),  # more than 2 outcomes
        ),
    )
    box = TimeBox.atomic(schedule, locus_components=("QB1", "QB2"), label="test_box")
    with pytest.raises(ValueError, match="ConditionalInstruction requires exactly two outcomes."):
        schedule_builder.timeboxes_to_playlist([box])


def test_timeboxes_to_playlist_unsupported_instruction(schedule_builder):
    """Test error cases in timeboxes_to_playlist"""
    schedule = Schedule()
    schedule.add_channels(["QB1__drive.awg"])
    schedule.append("QB1__drive.awg", Instruction(duration=15))
    box = TimeBox.atomic(schedule, locus_components=("QB1", "QB2"), label="test_box")
    with pytest.raises(ValueError, match="Instruction\\(duration=[0-9]+\\) not supported."):
        schedule_builder.timeboxes_to_playlist([box])
