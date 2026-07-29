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
from iqm.models.playlist.waveforms import Samples
import numpy as np
import pytest

from exa.common.data.parameter import CollectionType, Parameter
from exa.common.data.setting_node import SettingNode
from iqm.pulse.gate_implementation import Locus
from iqm.pulse.gates import (
    CZ_GaussianSmoothedSquare,
    CZ_Slepian,
    CZ_Slepian_ACStarkCRF,
    CZ_Slepian_CRF,
    FluxPulseGate_TGSS_CRF,
    GateImplementation,
    Measure_Constant,  # noqa: F401
    PRX_DRAGGaussian,
    PRX_FastDragSX,
    PRX_HdDragSX,
    expose_implementation,
    get_implementation_class,
)
from iqm.pulse.gates.conditional import CCPRX_Composite  # noqa: F401
from iqm.pulse.gates.enums import XYGate
from iqm.pulse.gates.measure import FEEDBACK_KEY
from iqm.pulse.gates.move import MoveMarker
from iqm.pulse.gates.prx import PRX_CustomWaveforms, PRX_CustomWaveformsSX, PRX_ModulatedCustomWaveForms
from iqm.pulse.gates.sx import SXGate  # noqa: F401
from iqm.pulse.gates.u import UGate  # noqa: F401
from iqm.pulse.playlist.instructions import (
    Block,
    ComplexIntegration,
    ConditionalInstruction,
    FluxPulse,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
    RealPulse,
    ThresholdStateDiscrimination,
    TimeTrace,
    VirtualRZ,
    Wait,
)
from iqm.pulse.playlist.waveforms import (
    CosineRiseFall,
    CosineRiseFallDerivative,
    Gaussian,
    GaussianDerivative,
    ModulatedCosineRiseFall,  # noqa: F401
    Slepian,
    TruncatedGaussian,
    TruncatedGaussianDerivative,
    TruncatedGaussianSmoothedSquare,
)
from iqm.pulse.quantum_ops import QuantumOp  # noqa: F401
from iqm.pulse.timebox import MultiplexedProbeTimeBox, TimeBox
from iqm.pulse.utils import normalize_angle, phase_transformation


def test_expose_implementation():
    """Test adding a gate implementation to the list of known implementations"""

    class NewImplementation(GateImplementation):
        symmetric = True
        parameters = {"a": Parameter(name="a")}

    assert get_implementation_class("NewImplementation") is None
    expose_implementation(NewImplementation)
    assert get_implementation_class("NewImplementation") == NewImplementation

    class NewImplementation(GateImplementation):
        symmetric = False
        parameters = {"b": Parameter(name="b")}

    with pytest.raises(ValueError, match="GateImplementation 'NewImplementation' has already been defined."):
        expose_implementation(NewImplementation)
    expose_implementation(NewImplementation, overwrite=True)


def check_implementation(impl: GateImplementation, op_name: str, impl_name: str, locus: Locus) -> None:
    """Check some common properties of GateImplementations."""
    assert isinstance(impl, GateImplementation)
    assert impl.parent.name == op_name
    assert impl.name == impl_name
    assert impl.locus == locus


def test_barrier(schedule_builder):
    """Test creating a Barrier gate and a timebox for it"""
    locus = ("QB1", "QB2")
    barrier = schedule_builder.get_implementation("barrier", locus, None)
    check_implementation(barrier, "barrier", "", locus)

    assert barrier.symmetric
    assert barrier.parent.symmetric
    timebox = barrier()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"Barrier on {locus}"


@pytest.mark.parametrize(
    "duration, wait_samples",
    [
        (1e-9, 32),  # up to minimum wait duration
        (23.9e-9, 48),  # up to granularity
        (24.1e-9, 64),  # up to granularity
    ],
)
def test_delay(schedule_builder, duration: float, wait_samples: int):
    """Test creating a Delay gate and a timebox for it"""
    locus = ("QB1", "QB2")
    delay = schedule_builder.get_implementation("delay", locus, None)
    check_implementation(delay, "delay", "wait", locus)

    assert delay.symmetric
    assert delay.parent.symmetric
    timebox = delay(duration)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"Delay on {locus}"
    schedule = timebox.atom
    assert schedule is not None
    assert len(schedule) == 4  # drive and flux channels for both qubits
    for _, seg in schedule.items():
        assert seg[0] == Wait(wait_samples)


def test_delay_too_long(schedule_builder):
    """Test maximum Delay duration."""
    locus = ("QB1",)
    delay = schedule_builder.get_implementation("delay", locus, None)
    check_implementation(delay, "delay", "wait", locus)

    with pytest.raises(ValueError, match="Requested delay duration 1.1 s exceeds"):
        delay(1.1)


@pytest.mark.parametrize("locus", [("QB1",), ("QB2",), ("QB5",), ("QB1", "QB2"), ("QB1", "QB5")])
def test_measure_constant(locus, schedule_builder):
    meas = schedule_builder.get_implementation("measure", locus, "constant")
    check_implementation(meas, "measure", "constant", locus)
    measure_calib_data = schedule_builder.calibration["measure"]["constant"]
    assert not meas.symmetric
    assert not meas.parent.symmetric
    timebox = meas()
    assert isinstance(timebox, TimeBox)
    assert len(timebox.children) == 1
    assert isinstance(timebox.children[0], MultiplexedProbeTimeBox)
    meas_seg = timebox.children[0].atom["PL-A__readout"]
    assert len(meas_seg) == 1
    assert isinstance(meas_seg[0], ReadoutTrigger)
    probe_pulse = meas_seg[0].probe_pulse
    assert isinstance(probe_pulse, MultiplexedIQPulse)
    assert len(probe_pulse.entries) == len(locus)
    assert probe_pulse.entries[0][1] == schedule_builder.channels["PL-A__readout"].integration_start_dead_time
    assert isinstance(probe_pulse.entries[0][0], IQPulse)
    assert probe_pulse.entries[0][0].scale_i == measure_calib_data[(locus[0],)]["amplitude_i"]
    assert len(meas_seg[0].acquisitions) == len(locus)
    for index, qubit in enumerate(locus):
        if "QB5" in locus and index:
            assert isinstance(meas_seg[0].acquisitions[index], ThresholdStateDiscrimination)
        else:
            assert isinstance(meas_seg[0].acquisitions[index], ComplexIntegration)
        assert meas_seg[0].acquisitions[index].label == f"{qubit}__readout.result"
    assert meas.duration_in_seconds() == 3.28e-07

    # test custom acquisition label
    custom_label_timebox = meas(key="poop")
    cmeas_seg = custom_label_timebox.children[0].atom["PL-A__readout"]
    for index, qubit in enumerate(locus):
        assert cmeas_seg[0].acquisitions[index].label == f"{qubit}__poop"
    # test naked probe timebox
    probe_timebox = meas.probe_timebox()
    assert isinstance(probe_timebox, MultiplexedProbeTimeBox)


@pytest.mark.parametrize("locus", [("QB1",), ("QB2",), ("QB5",), ("QB1", "QB2"), ("QB1", "QB5")])
def test_measure_constant_time_trace(locus, schedule_builder):
    meas = schedule_builder.get_implementation("measure", locus, "constant")

    # test time trace
    time_trace_box = meas.time_trace()
    assert isinstance(time_trace_box, TimeBox)
    assert len(time_trace_box.children) == 1
    assert isinstance(time_trace_box.children[0], MultiplexedProbeTimeBox)
    tr_seg = time_trace_box.children[0].atom["PL-A__readout"]
    assert len(tr_seg) == 1
    assert isinstance(tr_seg[0], ReadoutTrigger)
    assert isinstance(tr_seg[0].acquisitions[-1], TimeTrace)
    assert tr_seg[0].acquisitions[-1].label == "PL-A__readout.time_trace"
    tr_override_box = meas.time_trace(acquisition_delay=1.2e-7, acquisition_duration=1.6e-6)
    tro_seg = tr_override_box.children[0].atom["PL-A__readout"]
    assert tro_seg[0].acquisitions[-1].delay_samples == round(2.0e9 * 1.2e-7)
    assert tro_seg[0].acquisitions[-1].duration_samples == round(2.0e9 * 1.6e-6)


@pytest.mark.parametrize("locus", [("QB1",), ("QB2",), ("QB5",), ("QB1", "QB2"), ("QB1", "QB5")])
def test_measure_constant_priority_calibration(locus, schedule_builder):
    # wrong length integration weights throw an error
    with pytest.raises(ValueError, match="Integration length does not match with the provided integration weight"):
        schedule_builder.get_implementation(
            "measure",
            locus,
            "constant",
            priority_calibration_factorizable={
                locus[0:1]: {"integration_weights_I": np.zeros(1234), "integration_weights_Q": np.zeros(666)}
            },
        )

    prio_qubit = {"acquisition_type": "threshold"}
    prio = {(q,): prio_qubit for q in locus}
    measure = schedule_builder.get_implementation("measure", locus, priority_calibration_factorizable=prio)

    timebox_prio = measure()
    pmeas_seg = timebox_prio.children[0].atom["PL-A__readout"]
    for index, qubit in enumerate(locus):
        assert isinstance(pmeas_seg[0].acquisitions[index], ThresholdStateDiscrimination)

    # check the neighborhoods are correct when multiplexing and with feedback
    if len(locus) > 1:
        measure_box = measure(feedback_key="A")
        probes = {schedule_builder.chip_topology.component_to_probe_line[q] for q in locus}
        virtual_channels = set()
        for probe in probes:
            virtual_channels.update(schedule_builder.get_virtual_feedback_channels(probe))
        nbh = {0: set(locus).union(probes).union(virtual_channels)}
        assert measure_box.neighborhood_components == nbh
        assert measure_box.children[0].neighborhood_components == nbh


@pytest.mark.parametrize("locus", [("QB1",), ("QB2",), ("QB5",), ("QB1", "QB2"), ("QB1", "QB5")])
def test_shelved_measure_constant(locus, schedule_builder):
    meas = schedule_builder.get_implementation("measure_fidelity", locus, "shelved_constant")
    check_implementation(meas, "measure_fidelity", "shelved_constant", locus)
    measure_calib_data = schedule_builder.calibration["measure_fidelity"]["shelved_constant"]
    QB1_offset_in_sec = measure_calib_data[("QB1",)]["second_prx_12_offset"]
    QB1_offset_in_samples = (
        QB1_offset_in_sec
        / abs(QB1_offset_in_sec)
        * schedule_builder.channels["QB1__drive.awg"].duration_to_int_samples(abs(QB1_offset_in_sec))
    )
    assert not meas.symmetric
    assert not meas.parent.symmetric

    timebox = meas()
    assert isinstance(timebox, TimeBox)
    assert len(timebox.children) == 1
    probe_box = timebox.children[0]
    assert len(probe_box.children) == 2
    assert probe_box.children[0].children is not None
    assert isinstance(probe_box.children[1], MultiplexedProbeTimeBox)

    drive_channels_in_locus = [schedule_builder.get_drive_channel(qubit) for qubit in locus]
    probe_channels_in_locus = [schedule_builder.get_probe_channel(qubit) for qubit in locus]
    for qubit_num, drive_channel in enumerate(drive_channels_in_locus):
        assert isinstance(probe_box.children[0].children[qubit_num].atom[drive_channel][0], IQPulse)  # pre box (prx_12)
        assert isinstance(probe_box.children[1].atom[drive_channel][0], Block)
        assert isinstance(probe_box.children[1].atom[drive_channel][1], IQPulse)  # prx_12

    time_trace = meas.time_trace()
    for probe_channel in probe_channels_in_locus:
        probe = probe_channel.split("__")[0]
        locus_in_probe = [
            qubit for qubit in schedule_builder.chip_topology.probe_line_to_components[probe] if qubit in locus
        ]
        trigger = probe_box.children[1].atom[probe_channel][0]
        tr_trigger = time_trace.children[0].atom[probe_channel][0]
        trigger_duration = trigger.duration
        for drive_channel in drive_channels_in_locus:
            assert isinstance(probe_box.children[1].atom[drive_channel][0], Block)
            correct_block_duration = (
                trigger_duration if "QB1" not in drive_channel else trigger_duration + QB1_offset_in_samples
            )
            assert probe_box.children[1].atom[drive_channel][0].duration == correct_block_duration
        if locus_in_probe:
            assert isinstance(trigger, ReadoutTrigger)
            assert len(trigger.acquisitions) == len(locus_in_probe)
            assert len(tr_trigger.acquisitions) == len(locus_in_probe) + 1
            assert isinstance(tr_trigger.acquisitions[-1], TimeTrace)
    # assert that probe_timeboxes are multiplexable with basic constant measurement's
    if len(locus) == 2:
        normal_probe_box = schedule_builder.get_implementation("measure", (locus[0],), "constant").probe_timebox()
        shelved_probe_box = schedule_builder.get_implementation(
            "measure_fidelity", (locus[1],), "shelved_constant"
        ).probe_timebox()
        for multiplexed in [normal_probe_box + shelved_probe_box, shelved_probe_box + normal_probe_box]:
            mulitplexed_schedule = schedule_builder.resolve_timebox(multiplexed, neighborhood=0)
            for probe_channel in probe_channels_in_locus:
                trigger_count = 0
                for inst in mulitplexed_schedule[probe_channel]._instructions:
                    if isinstance(inst, ReadoutTrigger):
                        trigger_count += 1
                # if the multiplexing was successful, we should have just one trigger per probe
                assert trigger_count == 1


@pytest.mark.parametrize(
    "implementation",
    [
        "drag_gaussian",
        "drag_crf",
    ],
)
def test_prx_parameter_normalization(schedule_builder, implementation):
    """PRX gate parameter normalization works together with the caching feature."""

    def equal_box(a: TimeBox, b: TimeBox) -> bool:
        """True iff the two PRX implementation boxes are (approximately) equal."""
        channel = f"{locus[0]}__drive.awg"
        # both atomic
        assert not a.children
        assert not b.children
        seg_a = a.atom[channel]
        seg_b = b.atom[channel]
        assert len(seg_a) == 1
        assert len(seg_b) == 1
        ia = seg_a[0]
        ib = seg_b[0]
        return (
            ia.scale_i == pytest.approx(ib.scale_i)
            and ia.scale_q == pytest.approx(ib.scale_q)
            and ia.phase == pytest.approx(ib.phase)
        )

    half_turn = np.pi
    locus = ("QB1",)
    prx = schedule_builder.get_implementation("prx", locus, implementation)

    box1 = prx(angle=0.5, phase=0.1)
    box1_again = prx(angle=0.5, phase=0.1)

    box2 = prx(angle=-0.5, phase=0.1 + half_turn)  # different but equivalent args
    box2_again = prx(angle=-0.5, phase=0.1 + half_turn)

    box3 = prx(angle=0.5 + 2 * half_turn, phase=0.1 - 4 * half_turn)  # different but equivalent args

    box4 = prx(angle=0.5 + half_turn, phase=0.1)  # non-equivalent args
    box4_again = prx(angle=0.5 + half_turn, phase=0.1)

    # same args as before => cached copy is returned
    assert box1 is box1_again

    # different args => different cached copy, but the boxes are equivalent if the args are
    assert box2 is not box1
    assert box2 is box2_again
    assert equal_box(box1, box2)

    # different args => different cached copy, but the boxes are equivalent if the args are
    assert box3 is not box1
    assert equal_box(box1, box3)

    # non-equivalent args
    assert box4 is box4_again
    assert not equal_box(box1, box4)


@pytest.mark.parametrize(
    "implementation,waves",
    [
        ("drag_gaussian", (TruncatedGaussian, TruncatedGaussianDerivative)),
        ("drag_crf", (CosineRiseFall, CosineRiseFallDerivative)),
    ],
)
def test_prx_implementations(schedule_builder, implementation, waves):
    """Test creating a PRX gate and a timebox for it"""
    locus = ("QB1",)
    prx = schedule_builder.get_implementation("prx", locus, implementation)
    check_implementation(prx, "prx", implementation, locus)

    timebox = prx(angle=0.5, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"{prx.__class__.__name__} on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 1
    inst = seg[0]
    assert isinstance(inst, IQPulse)
    assert isinstance(inst.wave_i, waves[0])
    assert isinstance(inst.wave_q, waves[1])

    assert isinstance(prx.rx(1.0), TimeBox)
    assert isinstance(prx.ry(1.0), TimeBox)
    assert XYGate.IDENTITY not in prx._cliffords
    assert isinstance(prx.clifford(XYGate.IDENTITY), TimeBox)
    assert XYGate.IDENTITY in prx._cliffords
    assert isinstance(prx.clifford(XYGate.IDENTITY), TimeBox)  # Get same clifford again to access cache
    assert isinstance(prx.clifford(XYGate.X_90), TimeBox)
    assert isinstance(prx.clifford(XYGate.X_180), TimeBox)
    assert isinstance(prx.clifford(XYGate.X_M90), TimeBox)
    assert isinstance(prx.clifford(XYGate.Y_90), TimeBox)
    assert isinstance(prx.clifford(XYGate.Y_180), TimeBox)
    assert isinstance(prx.clifford(XYGate.Y_M90), TimeBox)
    with pytest.raises(ValueError, match="Unknown XYGate: 9"):
        prx.clifford(9)
    # test default cache has the correct keys after all of this
    assert set(prx._timebox_cache.keys()) == {
        (("angle", 0.5), ("phase", 0.1)),
        (("angle", 1.0), ("phase", 0.0)),
        (("angle", 1.0), ("phase", np.pi / 2)),
        (("angle", 0.0), ("phase", 0.0)),
        (("angle", np.pi / 2), ("phase", 0.0)),
        (("angle", np.pi / 2), ("phase", 0.0)),
        (("angle", np.pi), ("phase", 0.0)),
        (("angle", -np.pi / 2), ("phase", 0.0)),
        (("angle", np.pi / 2), ("phase", np.pi / 2)),
        (("angle", np.pi), ("phase", np.pi / 2)),
        (("angle", -np.pi / 2), ("phase", np.pi / 2)),
    }


def test_prx_drag_gaussian_gets_correct_calib_data(schedule_builder):
    locus = ("QB1",)
    prx = schedule_builder.get_implementation("prx", locus, "drag_gaussian")
    assert prx.pulse.duration == 80
    assert prx.pulse.wave_i.full_width == 0.5
    assert prx.pulse.wave_i.center_offset == 0.0
    assert prx.pulse.scale_i == 4.65642
    assert prx.pulse.scale_q == -0.1214


def test_prx_drag_crf_gets_correct_calib_data(schedule_builder):
    locus = ("QB1",)
    prx = schedule_builder.get_implementation("prx", locus, "drag_crf")
    assert prx.pulse.duration == 80
    assert prx.pulse.wave_i.full_width == 1.0
    assert prx.pulse.wave_i.rise_time == 0.5
    assert prx.pulse.wave_i.center_offset == 0.0
    assert prx.pulse.scale_i == 0.77
    assert prx.pulse.scale_q == 0.5


def test_zero_duration_prx_drag_gaussian(schedule_builder):
    """Zero-duration PRX is a special case."""
    locus = ("QB1",)
    schedule_builder.calibration["prx"]["drag_gaussian"][locus]["duration"] = 0.0
    prx = schedule_builder.get_implementation("prx", locus, "drag_gaussian")
    check_implementation(prx, "prx", "drag_gaussian", locus)

    timebox = prx(angle=0.5, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"PRX_DRAGGaussian on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 1
    assert seg.duration == 0.0
    inst = seg[0]
    assert isinstance(inst, Block)


def test_PRX_Samples(schedule_builder):
    schedule_builder.calibration["prx"]["samples"] = {
        ("QB1",): {
            "amplitude_i": 0.5,
            "amplitude_q": 0.0,
            "i": {"samples": np.ones(80)},
            "q": {"samples": 0.5 * np.ones(80)},
        }
    }
    timebox = schedule_builder.get_implementation("prx", ("QB1",), "samples")(np.pi, 0.0)
    assert isinstance(timebox.atom["QB1__drive.awg"][0], IQPulse)
    assert timebox.atom["QB1__drive.awg"][0].duration == 80
    assert isinstance(timebox.atom["QB1__drive.awg"][0].wave_i, Samples)
    assert np.array_equal(timebox.atom["QB1__drive.awg"][0].wave_i.sample(), np.ones(80))
    assert np.array_equal(timebox.atom["QB1__drive.awg"][0].wave_q.sample(), 0.5 * np.ones(80))


@pytest.mark.parametrize("use_defaults", [True, False])
def test_PRX_CustomWaveforms_get_parameters(use_defaults: bool):
    """PRX with hot-swappable waveforms."""

    class PRX_test(PRX_CustomWaveforms, wave_i=Gaussian, wave_q=GaussianDerivative, dependent_waves=False):
        pass

    settings = PRX_test.get_parameters(["QB1"], use_defaults=use_defaults)
    assert isinstance(settings, SettingNode)
    assert set(settings.settings.keys()) == {"duration", "amplitude_i", "amplitude_q"}
    assert set(settings.subtrees.keys()) == {"i", "q"}
    assert set(settings.i.settings.keys()) == {"sigma", "center_offset"}
    assert set(settings.q.settings.keys()) == {"sigma", "center_offset"}
    # default values
    if use_defaults:
        assert settings.i.center_offset.value == 0.0
        assert settings.q.center_offset.value == 0.0
    else:
        assert settings.i.center_offset.value is None
        assert settings.q.center_offset.value is None


def test_PRX_CustomWaveforms(schedule_builder):
    class PRX_test(PRX_CustomWaveforms, wave_i=GaussianDerivative, wave_q=Gaussian, dependent_waves=False):
        pass

    # monkeypatch the op_table and calibration
    schedule_builder.op_table["prx"].implementations["PRX_test"] = PRX_test
    schedule_builder.calibration["prx"]["PRX_test"] = {
        ("QB1",): {
            "duration": 40e-9,
            "amplitude_i": 0.6,
            "amplitude_q": 0.1,
            "i": {
                "sigma": 15e-9,
                "center_offset": 5e-9,
            },
            "q": {
                "sigma": 15e-9,
                "center_offset": -5e-9,
            },
        },
    }
    schedule_builder.validate_calibration()

    locus = ("QB1",)
    prx = schedule_builder.get_implementation("prx", locus, "PRX_test")
    check_implementation(prx, "prx", "PRX_test", locus)

    timebox = prx(angle=0.5, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"PRX_test on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 1
    inst = seg[0]
    assert isinstance(inst, IQPulse)
    assert isinstance(inst.wave_i, GaussianDerivative)
    assert isinstance(inst.wave_q, Gaussian)


def test_PRX_CustomWaveformsSX_get_parameters():
    """PRX with hot-swappable waveforms."""

    class PRX_test(PRX_CustomWaveformsSX, wave_i=Gaussian, wave_q=GaussianDerivative, dependent_waves=False):
        pass

    settings = PRX_test.get_parameters(["QB1"])
    assert isinstance(settings, SettingNode)
    assert set(settings.settings.keys()) == {
        "duration",
        "amplitude_i",
        "amplitude_q",
        "rz_before",
        "rz_after",
    }
    assert set(settings.subtrees.keys()) == {"i", "q"}
    assert set(settings.i.settings.keys()) == {"sigma", "center_offset"}
    assert set(settings.q.settings.keys()) == {"sigma", "center_offset"}


def test_PRX_CustomWaveformsSX(schedule_builder):
    class PRX_test(PRX_CustomWaveformsSX, wave_i=GaussianDerivative, wave_q=Gaussian, dependent_waves=False):
        pass

    # monkeypatch the op_table and calibration
    schedule_builder.op_table["prx"].implementations["PRX_test"] = PRX_test
    schedule_builder.calibration["prx"]["PRX_test"] = {
        ("QB1",): {
            "duration": 40e-9,
            "amplitude_i": 0.6,
            "amplitude_q": 0.1,
            "rz_before": 0.1,
            "rz_after": 0.1,
            "i": {
                "sigma": 15e-9,
                "center_offset": 5e-9,
            },
            "q": {
                "sigma": 15e-9,
                "center_offset": -5e-9,
            },
        },
    }
    schedule_builder.validate_calibration()

    locus = ("QB1",)
    prx = schedule_builder.get_implementation("prx", locus, "PRX_test")
    check_implementation(prx, "prx", "PRX_test", locus)

    # Test two pulses
    timebox = prx(angle=0.5, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"PRX_test on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 2
    inst = seg[0]
    assert isinstance(inst, IQPulse)
    assert isinstance(inst.wave_i, GaussianDerivative)
    assert isinstance(inst.wave_q, Gaussian)

    # Test one pulse X90 pulse
    timebox = prx(angle=np.pi / 2, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"PRX_test on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 1
    inst = seg[0]
    assert isinstance(inst, IQPulse)
    assert isinstance(inst.wave_i, GaussianDerivative)
    assert isinstance(inst.wave_q, Gaussian)

    # Test one pulse in the special case of rotation angle is zero
    timebox = prx(angle=0.0, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"PRX_test on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 1
    inst = seg[0]
    assert isinstance(inst, IQPulse)
    assert isinstance(inst.wave_i, GaussianDerivative)
    assert isinstance(inst.wave_q, Gaussian)
    assert inst.scale_i == pytest.approx(0.0)
    assert inst.scale_q == pytest.approx(0.0)
    assert inst.phase_increment == pytest.approx(0.0)


def test_PRX_FastDragSX_get_parameters():
    settings = PRX_FastDragSX.get_parameters(["QB1"])
    assert isinstance(settings, SettingNode)
    assert set(settings.settings.keys()) == {
        "duration",
        "full_width",
        "amplitude_i",
        "amplitude_q",
        "rz_before",
        "rz_after",
        "coefficients",
        "compute_coefs_from_frequencies",
        "number_of_cos_terms",
        "suppressed_frequencies",
        "suppressed_interval_widths",
        "weights",
        "center_offset",
    }


def test_PRX_HDDragSX_get_parameters():
    settings = PRX_HdDragSX.get_parameters(["QB1"])
    assert isinstance(settings, SettingNode)
    assert set(settings.settings.keys()) == {
        "duration",
        "full_width",
        "amplitude_i",
        "amplitude_q",
        "rz_before",
        "rz_after",
        "coefficients",
        "compute_coefs_from_frequencies",
        "suppressed_frequencies",
        "center_offset",
    }


def test_PRX_ModulatedCustomWaveforms(schedule_builder):
    class PRX_test(PRX_ModulatedCustomWaveForms, wave_i=Gaussian, wave_q=GaussianDerivative, dependent_waves=False):
        pass

    # monkeypatch the op_table and calibration
    schedule_builder.op_table["prx_12"].implementations["PRX_test"] = PRX_test
    schedule_builder.calibration["prx_12"]["PRX_test"] = {
        ("QB1",): {
            "duration": 40e-9,
            "amplitude_i": 0.6,
            "amplitude_q": 0.1,
            "frequency": 0.0,
            "i": {
                "sigma": 15e-9,
                "center_offset": 5e-9,
            },
            "q": {
                "sigma": 15e-9,
                "center_offset": -5e-9,
            },
        },
    }
    schedule_builder.validate_calibration()

    locus = ("QB1",)
    prx_12 = schedule_builder.get_implementation("prx_12", locus, "PRX_test")
    check_implementation(prx_12, "prx_12", "PRX_test", locus)

    timebox = prx_12(angle=0.5, phase=0.1)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"PRX_test on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    channel, seg = next(iter(schedule.items()))
    assert channel == "QB1__drive.awg"
    assert len(seg) == 1
    inst = seg[0]
    assert isinstance(inst, IQPulse)
    assert isinstance(inst.wave_i, Gaussian)
    assert isinstance(inst.wave_q, GaussianDerivative)


def test_prx12_drag_crf_gets_correct_calib_data(schedule_builder):
    locus = ("QB1",)
    prx_12 = schedule_builder.get_implementation("prx_12", locus, "modulated_drag_crf")
    assert prx_12.pulse.duration == 80
    assert prx_12.pulse.modulation_frequency == pytest.approx(-0.1)
    assert prx_12.pulse.scale_i == pytest.approx(0.25)
    assert prx_12.pulse.scale_q == pytest.approx(0.0)


def test_rz_virtual(schedule_builder):
    locus = ("QB1",)
    angle = 0.1
    rz = schedule_builder.get_implementation("rz", locus, "virtual")
    check_implementation(rz, "rz", "virtual", locus)

    timebox = rz(angle)
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"RZ_Virtual on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], VirtualRZ)
    assert schedule["QB1__drive.awg"][0].phase_increment == pytest.approx(-angle)


def test_rz_virtual_ac_stark_crf(schedule_builder):
    locus = ("QB1",)
    rz_physical = schedule_builder.get_implementation("rz_physical", locus, "ac_stark_crf")
    check_implementation(rz_physical, "rz_physical", "ac_stark_crf", locus)

    timebox = rz_physical()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"RZ_ACStarkShift_CosineRiseFall on {locus}"
    schedule = timebox.atom
    assert len(schedule) == 1
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], IQPulse)
    assert set(rz_physical._timebox_cache.keys()) == {tuple()}


def test_cz_at_invalid_locus(schedule_builder):
    """Test getting implementation for CZ at invalid locus"""
    regexp = "No calibration data for 'cz.gaussian_smoothed_square' at \\('QB1', 'QB11'\\)."
    with pytest.raises(ValueError, match=regexp):
        schedule_builder.get_implementation(
            "cz",
            ("QB1", "QB11"),
            "gaussian_smoothed_square",
        )


def test_get_parameters_CZ_GaussianSmoothedSquare():
    """Test getting parameters of a gate implementation"""
    impl = get_implementation_class("CZ_GaussianSmoothedSquare")

    settings = impl.get_parameters(["QB1", "QB2"])
    assert isinstance(settings, SettingNode)
    assert settings.name == ""
    assert set(settings.settings.keys()) == {"duration"}
    assert set(settings.subtrees.keys()) == {"coupler", "rz"}
    assert set(settings.coupler.settings.keys()) == {"amplitude", "square_width", "gaussian_sigma", "center_offset"}
    assert len(settings.coupler.child_nodes) == 0
    assert set(settings.rz.settings.keys()) == {"QB1", "QB2"}
    assert len(settings.rz.child_nodes) == 0


def test_CZ_GaussianSmoothedSquare(schedule_builder):
    """Test creating a CZ gate and a timebox for it"""
    locus = ("QB1", "QB2")
    cz = schedule_builder.get_implementation("cz", locus, "gaussian_smoothed_square")
    check_implementation(cz, "cz", "gaussian_smoothed_square", locus)

    assert cz.symmetric
    assert cz.parent.symmetric
    timebox = cz()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"CZ_GaussianSmoothedSquare on {locus}"
    expected_names = {"QB1__drive.awg", "QB2__drive.awg", "TC-1-2__flux.awg"}
    assert set(timebox.atom._contents.keys()) == expected_names

    # Test invalid CZ, no rz_angle for locus
    regexp = "cz.invalid_cz: \\('QB1', 'QB2'\\): Calibration is missing an RZ angle for locus component QB1."
    with pytest.raises(ValueError, match=regexp):
        CZ_GaussianSmoothedSquare(
            schedule_builder.op_table["cz"],
            "invalid_cz",
            ("QB1", "QB2"),
            calibration_data={
                "duration": 32e-9,
                "coupler": {
                    "amplitude": 0.7,
                    "square_width": 21e-9,
                    "gaussian_sigma": 4.7e-9,
                },
                "rz": {
                    "QB2": -0.26,
                },
            },
            builder=schedule_builder,
        )
    assert set(cz._timebox_cache.keys()) == {tuple()}


def test_zero_duration_CZ_GaussianSmoothedSquare(schedule_builder):
    locus = ("QB1", "QB2")
    schedule_builder.calibration["cz"]["gaussian_smoothed_square"][locus]["duration"] = 0.0
    cz = schedule_builder.get_implementation("cz", locus, "gaussian_smoothed_square")
    check_implementation(cz, "cz", "gaussian_smoothed_square", locus)
    assert cz.symmetric
    assert cz.parent.symmetric
    timebox = cz()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"CZ_GaussianSmoothedSquare on {locus}"
    expected_names = {"QB1__drive.awg", "QB2__drive.awg", "TC-1-2__flux.awg"}
    assert set(timebox.atom._contents.keys()) == expected_names
    for seg in timebox.atom._contents.values():
        assert len(seg._instructions) == 1
        assert isinstance(seg._instructions[0], Block)
        assert seg.duration == 0.0


def test_get_parameters_FluxPulseGate_TGSS_CRF():
    """Test getting parameters of a gate implementation"""
    impl = get_implementation_class("FluxPulseGate_TGSS_CRF")

    settings = impl.get_parameters(["QB1", "QB2"])
    assert isinstance(settings, SettingNode)
    assert settings.name == ""
    assert set(settings.settings.keys()) == {"duration"}
    assert set(settings.subtrees.keys()) == {"coupler", "qubit", "rz"}
    assert set(settings.coupler.settings.keys()) == {"amplitude", "full_width", "rise_time", "center_offset"}
    assert len(settings.coupler.child_nodes) == 0
    assert set(settings.qubit.settings.keys()) == {"amplitude", "full_width", "rise_time", "center_offset"}
    assert len(settings.qubit.child_nodes) == 0
    assert set(settings.rz.settings.keys()) == {"QB1", "QB2"}
    assert len(settings.rz.child_nodes) == 0


def test_FluxPulseGate_TGSS_CRF(schedule_builder):
    """Test the FluxPulseGate_TGSS_CRF implementation."""

    locus = ("QB1", "QB2")
    # monkeypatch the op_table and calibration and chip_topology
    schedule_builder.op_table["cz"].implementations["fast_cz"] = FluxPulseGate_TGSS_CRF
    schedule_builder.calibration["cz"] = {}
    schedule_builder.calibration["cz"]["fast_cz"] = {
        locus: {
            "duration": 40e-9,
            "coupler": {
                "amplitude": 0.6,
                "full_width": 25e-9,
                "rise_time": 5e-9,
                "center_offset": 5e-9,
            },
            "qubit": {
                "amplitude": 0.1,
                "full_width": 25e-9,
                "rise_time": 1e-9,
                "center_offset": 5e-9,
            },
            "rz": dict(zip(locus, (0.1, 0.2))),
        },
    }
    schedule_builder.validate_calibration()
    fast_cz = schedule_builder.get_implementation("cz", locus, impl_name="fast_cz")
    check_implementation(fast_cz, "cz", "fast_cz", locus)

    timebox = fast_cz()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"FluxPulseGate_TGSS_CRF on {locus}"
    expected_names = {"QB1__drive.awg", "QB2__drive.awg", "QB1__flux.awg", "TC-1-2__flux.awg"}
    assert set(timebox.atom._contents.keys()) == expected_names
    schedule = timebox.atom
    assert len(schedule) == 4

    # virtual z rotations:
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], VirtualRZ)
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC-1-2__flux.awg"]) == 1
    assert isinstance(schedule["TC-1-2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC-1-2__flux.awg"][0].wave, TruncatedGaussianSmoothedSquare)
    assert schedule["TC-1-2__flux.awg"][0].scale == 0.6
    assert len(schedule["QB1__flux.awg"]) == 1
    assert isinstance(schedule["QB1__flux.awg"][0], RealPulse)
    assert isinstance(schedule["QB1__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["QB1__flux.awg"][0].scale == 0.1


def test_FluxPulseGate_TGSS_CRF_on_star(schedule_builder_star):
    """Test the FluxPulseGate_TGSS_CRF gate implementation."""

    locus = ("QB1", "COMP_R")
    # monkeypatch the op_table and calibration and chip_topology
    schedule_builder_star.op_table["cz"].implementations["fast_cz"] = FluxPulseGate_TGSS_CRF
    schedule_builder_star.calibration["cz"] = {}
    schedule_builder_star.calibration["cz"]["fast_cz"] = {
        locus: {
            "duration": 40e-9,
            "coupler": {
                "amplitude": 0.6,
                "full_width": 25e-9,
                "rise_time": 5e-9,
                "center_offset": 5e-9,
            },
            "qubit": {
                "amplitude": 0.1,
                "full_width": 25e-9,
                "rise_time": 1e-9,
                "center_offset": 5e-9,
            },
            "rz": dict(zip(locus, (0.1, 0.2))),
        },
    }
    schedule_builder_star.validate_calibration()
    fast_cz = schedule_builder_star.get_implementation("cz", locus, impl_name="fast_cz")
    check_implementation(fast_cz, "cz", "fast_cz", locus)

    timebox = fast_cz()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"FluxPulseGate_TGSS_CRF on {locus}"
    expected_names = {"QB1__drive.awg", "QB1__flux.awg", "TC1__flux.awg", "COMP_R__drive_virtual"}
    assert set(timebox.atom._contents.keys()) == expected_names
    schedule = timebox.atom
    assert len(schedule) == 4

    # virtual z rotations:
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], VirtualRZ)
    assert len(schedule["COMP_R__drive_virtual"]) == 1
    assert isinstance(schedule["COMP_R__drive_virtual"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC1__flux.awg"]) == 1
    assert isinstance(schedule["TC1__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC1__flux.awg"][0].wave, TruncatedGaussianSmoothedSquare)
    assert schedule["TC1__flux.awg"][0].scale == 0.6
    assert len(schedule["QB1__flux.awg"]) == 1
    assert isinstance(schedule["QB1__flux.awg"][0], RealPulse)
    assert isinstance(schedule["QB1__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["QB1__flux.awg"][0].scale == 0.1


def test_MOVE_CRF_CRF(schedule_builder_star):
    """Test the MOVE_CRF_CRF implementation."""
    locus = ("QB2", "COMP_R")
    move = schedule_builder_star.get_implementation("move", locus, impl_name="crf_crf")
    check_implementation(move, "move", "crf_crf", locus)
    timebox = move()

    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"MOVE_CRF_CRF on {locus}"
    assert len(timebox.children) == 2
    # marker box
    schedule = timebox.children[0].atom
    expected_names = ["QB2__drive.awg", "COMP_R__drive_virtual"]
    assert set(schedule) == set(expected_names)
    assert len(schedule[expected_names[0]]) == 1
    assert len(schedule[expected_names[1]]) == 1
    # same MoveMarker instance in both channels
    assert isinstance(schedule[expected_names[0]][0], MoveMarker)
    assert schedule[expected_names[0]][0] is schedule[expected_names[1]][0]
    # move box
    schedule = timebox.children[1].atom
    assert set(schedule) == {"QB2__drive.awg", "QB2__flux.awg", "TC2__flux.awg", "COMP_R__drive_virtual"}

    # virtual z rotations:
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], VirtualRZ)
    assert len(schedule["COMP_R__drive_virtual"]) == 1
    assert isinstance(schedule["COMP_R__drive_virtual"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC2__flux.awg"]) == 1
    assert isinstance(schedule["TC2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC2__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["TC2__flux.awg"][0].scale == 0.12
    assert len(schedule["QB2__flux.awg"]) == 1
    assert isinstance(schedule["QB2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["QB2__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["QB2__flux.awg"][0].scale == 0.07

    # detuning value from config
    assert move.calibration_data["detuning"] == 100e6


def test_MOVE_SLEPIAN_CRF(schedule_builder_star):
    """Test the MOVE_SLEPIAN_CRF implementation."""
    locus = ("QB2", "COMP_R")
    move = schedule_builder_star.get_implementation("move", locus, impl_name="slepian_crf")
    check_implementation(move, "move", "slepian_crf", locus)
    timebox = move()

    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"MOVE_SLEPIAN_CRF on {locus}"
    assert len(timebox.children) == 2
    # marker box
    schedule = timebox.children[0].atom
    expected_names = ["QB2__drive.awg", "COMP_R__drive_virtual"]
    assert set(schedule) == set(expected_names)
    assert len(schedule[expected_names[0]]) == 1
    assert len(schedule[expected_names[1]]) == 1
    # same MoveMarker instance in both channels
    assert isinstance(schedule[expected_names[0]][0], MoveMarker)
    assert schedule[expected_names[0]][0] is schedule[expected_names[1]][0]
    # move box
    schedule = timebox.children[1].atom
    assert set(schedule) == {"QB2__drive.awg", "QB2__flux.awg", "TC2__flux.awg", "COMP_R__drive_virtual"}

    # virtual z rotations:
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], VirtualRZ)
    assert len(schedule["COMP_R__drive_virtual"]) == 1
    assert isinstance(schedule["COMP_R__drive_virtual"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC2__flux.awg"]) == 1
    assert isinstance(schedule["TC2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC2__flux.awg"][0].wave, Slepian)
    assert schedule["TC2__flux.awg"][0].scale == 0.12
    assert len(schedule["QB2__flux.awg"]) == 1
    assert isinstance(schedule["QB2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["QB2__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["QB2__flux.awg"][0].scale == 0.07

    # detuning value from config
    assert move.calibration_data["detuning"] == 100e6


def test_MOVE_TGSS_CRF(schedule_builder_star):
    """Test the MOVE_TGSS_CRF implementation."""
    locus = ("QB2", "COMP_R")
    move = schedule_builder_star.get_implementation("move", locus, impl_name="tgss_crf")
    check_implementation(move, "move", "tgss_crf", locus)
    timebox = move()

    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"MOVE_TGSS_CRF on {locus}"
    assert len(timebox.children) == 2
    # marker box
    schedule = timebox.children[0].atom
    expected_names = ["QB2__drive.awg", "COMP_R__drive_virtual"]
    assert set(schedule) == set(expected_names)
    assert len(schedule[expected_names[0]]) == 1
    assert len(schedule[expected_names[1]]) == 1
    # same MoveMarker instance in both channels
    assert isinstance(schedule[expected_names[0]][0], MoveMarker)
    assert schedule[expected_names[0]][0] is schedule[expected_names[1]][0]
    # move box
    schedule = timebox.children[1].atom
    assert set(schedule) == {"QB2__drive.awg", "QB2__flux.awg", "TC2__flux.awg", "COMP_R__drive_virtual"}

    # virtual z rotations:
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], VirtualRZ)
    assert len(schedule["COMP_R__drive_virtual"]) == 1
    assert isinstance(schedule["COMP_R__drive_virtual"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC2__flux.awg"]) == 1
    assert isinstance(schedule["TC2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC2__flux.awg"][0].wave, TruncatedGaussianSmoothedSquare)
    assert schedule["TC2__flux.awg"][0].scale == 0.12
    assert len(schedule["QB2__flux.awg"]) == 1
    assert isinstance(schedule["QB2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["QB2__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["QB2__flux.awg"][0].scale == 0.07

    # detuning value from config
    assert move.calibration_data["detuning"] == 100e6


def test_CZ_CouplerSlepian(schedule_builder):
    """Test the CZ_CouplerSlepian implementation."""

    locus = ("QB1", "QB2")
    schedule_builder.op_table["cz"].implementations["slepian"] = CZ_Slepian
    schedule_builder.calibration["cz"] = {}
    schedule_builder.calibration["cz"]["slepian"] = {
        locus: {
            "duration": 40e-9,
            "coupler": {
                "amplitude": 0.2,
                "full_width": 25e-9,
                "center_offset": 0,
                "lambda_1": -0.5,
                "lambda_2": 0.1,
                "frequency_initial_normalized": 0.7,
                "frequency_to_minimize_normalized": 0.9,
                "coupling_strength_normalized": 0.01,
                "squid_asymmetry": 0.3,
            },
            "rz": dict(zip(locus, (0.1, 0.2))),
        },
    }
    schedule_builder.validate_calibration()
    cz_slepian = schedule_builder.get_implementation("cz", locus, impl_name="slepian")
    check_implementation(cz_slepian, "cz", "slepian", locus)

    timebox = cz_slepian()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"CZ_Slepian on {locus}"
    expected_names = {"QB1__drive.awg", "QB2__drive.awg", "TC-1-2__flux.awg"}
    assert set(timebox.atom._contents.keys()) == expected_names
    schedule = timebox.atom
    assert len(schedule) == 3

    # virtual z rotations:
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], VirtualRZ)
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC-1-2__flux.awg"]) == 1
    assert isinstance(schedule["TC-1-2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC-1-2__flux.awg"][0].wave, Slepian)


def test_CZ_CouplerSlepian_QubitCRF(schedule_builder):
    """Test the CZ_CouplerSlepian_QubitCRF implementation."""

    locus = ("QB1", "QB2")
    schedule_builder.op_table["cz"].implementations["slepian_crf"] = CZ_Slepian_CRF
    schedule_builder.calibration["cz"] = {}
    schedule_builder.calibration["cz"]["slepian_crf"] = {
        locus: {
            "duration": 40e-9,
            "coupler": {
                "amplitude": 0.6,
                "full_width": 25e-9,
                "center_offset": 0,
                "lambda_1": -0.5,
                "lambda_2": 0.1,
                "frequency_initial_normalized": 0.7,
                "frequency_to_minimize_normalized": 0.9,
                "coupling_strength_normalized": 0.01,
                "squid_asymmetry": 0.3,
            },
            "qubit": {
                "amplitude": 0.1,
                "full_width": 25e-9,
                "rise_time": 1e-9,
                "center_offset": 0,
            },
            "rz": dict(zip(locus, (0.1, 0.2))),
        },
    }
    schedule_builder.validate_calibration()
    cz_slepian = schedule_builder.get_implementation("cz", locus, impl_name="slepian_crf")
    check_implementation(cz_slepian, "cz", "slepian_crf", locus)

    timebox = cz_slepian()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"CZ_Slepian_CRF on {locus}"
    expected_names = {"QB1__drive.awg", "QB2__drive.awg", "QB1__flux.awg", "TC-1-2__flux.awg"}
    assert set(timebox.atom._contents.keys()) == expected_names
    schedule = timebox.atom
    assert len(schedule) == 4

    # virtual z rotations:
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], VirtualRZ)
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], VirtualRZ)

    # flux pulses
    assert len(schedule["TC-1-2__flux.awg"]) == 1
    assert isinstance(schedule["TC-1-2__flux.awg"][0], RealPulse)
    assert isinstance(schedule["TC-1-2__flux.awg"][0].wave, Slepian)
    assert len(schedule["QB1__flux.awg"]) == 1
    assert isinstance(schedule["QB1__flux.awg"][0], RealPulse)
    assert isinstance(schedule["QB1__flux.awg"][0].wave, CosineRiseFall)
    assert schedule["QB1__flux.awg"][0].scale == 0.1


def test_CZ_CouplerSlepian_QubitACStarkPulse(schedule_builder):
    """Test the CZ_CouplerSlepian_QubitCRF implementation."""

    locus = ("QB1", "QB2")
    schedule_builder.op_table["cz"].implementations["slepian_acstarkcrf"] = CZ_Slepian_ACStarkCRF
    schedule_builder.calibration["cz"] = {}
    schedule_builder.calibration["cz"]["slepian_acstarkcrf"] = {
        locus: {
            "duration": 40e-9,
            "coupler": {
                "amplitude": 0.6,
                "full_width": 25e-9,
                "center_offset": 0,
                "lambda_1": -0.5,
                "lambda_2": 0.1,
                "frequency_initial_normalized": 0.7,
                "frequency_to_minimize_normalized": 0.9,
                "coupling_strength_normalized": 0.01,
                "squid_asymmetry": 0.3,
            },
            "first_qubit": {
                "amplitude": 0.1,
                "full_width": 40e-9,
                "rise_time": 10e-9,
                "center_offset": 0,
                "modulation_frequency": 280e6,
                "phase": 0,
            },
            "second_qubit": {
                "amplitude": 0.0,
                "full_width": 40e-9,
                "rise_time": 10e-9,
                "center_offset": 0,
                "modulation_frequency": 280e6,
                "phase": 0,
            },
            "rz": dict(zip(locus + ("QB4", "QB5"), (0.1, 0.2, 0.01, 0.02))),
        },
    }
    schedule_builder.validate_calibration()
    cz_slepian = schedule_builder.get_implementation("cz", locus, impl_name="slepian_acstarkcrf")
    check_implementation(cz_slepian, "cz", "slepian_acstarkcrf", locus)

    timebox = cz_slepian()
    assert isinstance(timebox, TimeBox)
    assert timebox.label == f"CZ_Slepian_ACStarkCRF on {locus}"
    expected_names = {"QB1__drive.awg", "QB2__drive.awg", "TC-1-2__flux.awg"}
    assert set(timebox.atom._contents.keys()) == expected_names
    schedule = timebox.atom
    assert len(schedule) == 3

    # virtual z rotations:
    assert len(schedule["QB1__drive.awg"]) == 1
    assert isinstance(schedule["QB1__drive.awg"][0], IQPulse)
    assert len(schedule["QB2__drive.awg"]) == 1
    assert isinstance(schedule["QB2__drive.awg"][0], IQPulse)

    # flux pulses
    assert len(schedule["TC-1-2__flux.awg"]) == 1
    assert isinstance(schedule["TC-1-2__flux.awg"][0], FluxPulse)
    assert isinstance(schedule["TC-1-2__flux.awg"][0].wave, Slepian)
    # off-locus vz
    assert schedule["TC-1-2__flux.awg"][0].rzs == (("QB4__drive.awg", 0.01), ("QB5__drive.awg", 0.02))


def test_subsubclassing_of_custom_waveform_baseclasses():
    class PRXDG_with_both_swapped(PRX_DRAGGaussian, wave_i=CosineRiseFall, wave_q=CosineRiseFall):
        """PRXDG_with_i_swapped"""

    assert PRXDG_with_both_swapped.wave_i == CosineRiseFall
    assert PRXDG_with_both_swapped.wave_q == CosineRiseFall
    assert PRXDG_with_both_swapped.dependent_waves

    class PRXDG_that_is_not_drag(PRX_DRAGGaussian, dependent_waves=False):
        """PRXDG_that_is_not_drag"""

    assert PRXDG_that_is_not_drag.wave_i == TruncatedGaussian
    assert PRXDG_that_is_not_drag.wave_q == TruncatedGaussianDerivative
    assert not PRXDG_that_is_not_drag.dependent_waves

    class PRX_subsubsubclass(PRXDG_with_both_swapped):
        """PRX_subsubsubclass"""

    assert PRX_subsubsubclass.wave_i == CosineRiseFall
    assert PRX_subsubsubclass.wave_q == CosineRiseFall
    assert PRX_subsubsubclass.dependent_waves

    class Slepian_with_coupler_swapped(CZ_Slepian_CRF, coupler_wave=TruncatedGaussianSmoothedSquare):
        """Slepian_with_coupler_swapped"""

    assert Slepian_with_coupler_swapped.coupler_wave == TruncatedGaussianSmoothedSquare
    assert Slepian_with_coupler_swapped.qubit_wave is None

    class Slepian_with_both_swapped(
        CZ_Slepian_CRF, coupler_wave=TruncatedGaussianSmoothedSquare, qubit_wave=TruncatedGaussianSmoothedSquare
    ):
        """Slepian_with_coupler_swapped"""

    assert Slepian_with_both_swapped.coupler_wave == TruncatedGaussianSmoothedSquare
    assert Slepian_with_both_swapped.coupler_wave == TruncatedGaussianSmoothedSquare

    class CZ_subsubclass(Slepian_with_coupler_swapped):
        """CZ_subsubclass"""

    assert CZ_subsubclass.coupler_wave == TruncatedGaussianSmoothedSquare
    assert CZ_subsubclass.qubit_wave is None


def test_convert_calib_data_with_list_parameters(schedule_builder):
    class PRX_with_array_params(PRX_DRAGGaussian):
        parameters = PRX_DRAGGaussian.parameters.copy()
        parameters["list_in_seconds"] = Parameter("", "list in seconds", unit="s", collection_type=CollectionType.LIST)
        parameters["list_in_hz"] = Parameter("", "list in Hz", unit="Hz", collection_type=CollectionType.LIST)
        parameters["array_in_seconds"] = Parameter(
            "", "array in seconds", unit="s", collection_type=CollectionType.NDARRAY
        )

    schedule_builder.op_table["prx"].implementations["array_prx"] = PRX_with_array_params
    calib_data = {
        "duration": 4e-8,
        "amplitude_i": 0.5,
        "amplitude_q": 0.1,
        "full_width": 8e-8,
        "center_offset": 0.0,
        "list_in_seconds": [4e-8, 8e-8, 16e-8],
        "list_in_hz": [0.25e8, 0.5e8, 1e8],
        "array_in_seconds": np.array([[4e-8, 4e-8], [8e-8, 8e-8]]),
    }
    converted = PRX_with_array_params.convert_calibration_data(
        calib_data, PRX_with_array_params.parameters, schedule_builder.channels["QB1__drive.awg"], duration=None
    )
    assert np.all(converted["array_in_seconds"] == np.array([[1.0, 1.0], [2.0, 2.0]]))
    del converted["array_in_seconds"]
    assert converted == {
        "n_samples": 80,
        "amplitude_i": 0.5,
        "amplitude_q": 0.1,
        "full_width": 2.0,
        "center_offset": 0.0,
        "list_in_seconds": [1.0, 2.0, 4.0],
        "list_in_hz": [1.0, 2.0, 4.0],
    }


@pytest.mark.parametrize(
    "angle,normalized",
    [
        (0.0, 0.0),
        (np.pi, np.pi),
        (-np.pi, np.pi),
        (2 * np.pi, 0.0),
        (3 * np.pi, np.pi),
        (0.1, pytest.approx(0.1)),
        (-0.1, pytest.approx(-0.1)),
        (0.99 * np.pi, 0.99 * np.pi),
        (-0.99 * np.pi, -0.99 * np.pi),
        (1.01 * np.pi, -0.99 * np.pi),
        (-1.01 * np.pi, 0.99 * np.pi),
    ],
)
def test_normalize_angle(angle, normalized):
    """Angles are correctly normalized to (-pi, pi]."""
    assert normalize_angle(angle) == normalized


def test_cc_prx(schedule_builder):
    schedule_builder.calibration["cc_prx"] = {"prx_composite": {("QB5",): {"control_delays": [500e-9, 500e-9]}}}
    cond = schedule_builder.get_implementation("cc_prx", ["QB5"])
    awg = "QB5__drive.awg"
    feedback_link = f"PL-A__feedforward_bits_to_{awg}"

    # cc_prx(pi, 0)
    delay_box, cond_box = cond(feedback_qubit="QB5", feedback_key="my_label")
    ci = cond_box.atom[awg][0]
    assert isinstance(ci, ConditionalInstruction)
    assert isinstance(ci.outcomes[0], Wait)
    assert ci.condition == f"QB5__{FEEDBACK_KEY}"
    bl = cond_box.atom[feedback_link][0]
    assert isinstance(bl, Block)
    assert bl.duration == 0
    bl = delay_box.atom[feedback_link][0]
    assert isinstance(bl, Block)
    assert bl.duration == 1000

    # check the prx pulse scaling
    assert isinstance(ci.outcomes[1], IQPulse)
    _, cond_box = cond(angle=np.pi / 2, feedback_qubit="QB5", feedback_key="my_label")
    assert cond_box.atom[awg][0].outcomes[1].scale_i == ci.outcomes[1].scale_i / 2

    # test cross feedback
    _, cond_box = cond(feedback_qubit="QB1", feedback_key="cross")
    assert cond_box.atom[awg][0].condition == f"QB1__{FEEDBACK_KEY}"


def test_conditional_reset(schedule_builder):
    reset = schedule_builder.get_implementation("reset", ["QB5"])
    box = reset()
    awg = "QB5__drive.awg"
    assert isinstance(box.children[0].children[0], MultiplexedProbeTimeBox)
    assert isinstance(box.children[2].atom[awg][0], ConditionalInstruction)

    reset = schedule_builder.get_implementation("reset", ["QB1", "QB5"])
    box = reset()
    awg1 = "QB1__drive.awg"
    assert isinstance(box.children[0].children[0], MultiplexedProbeTimeBox)
    assert isinstance(box.children[2].atom[awg1][0], ConditionalInstruction)
    assert isinstance(box.children[4].atom[awg][0], ConditionalInstruction)


def test_reset_wait(schedule_builder):
    reset = schedule_builder.get_implementation("reset_wait", ["QB1", "QB2", "QB3"])
    reset_box = reset()
    assert reset_box.children[0].neighborhood_components == {0: {"QB1", "QB2", "QB3", "TC-1-2", "PL-A", "PL-B"}}
    assert len(reset_box.children[0].children) == 3
    assert set(reset_box.children[0].children[0].atom._contents.keys()) == {"QB1__drive.awg", "QB1__flux.awg"}
    assert isinstance(reset_box.children[0].children[0].atom._contents["QB1__drive.awg"][0], Block)
    assert reset.duration_in_seconds() == 300e-6


@pytest.mark.parametrize(
    "rz_before,rz_after,phase,phase_increment",
    [
        (0.0, 0.0, 0.0, 0.0),
        (np.pi / 2, np.pi / 2, pytest.approx(np.pi / 2), pytest.approx(-np.pi)),
        (0.1, 0.2, pytest.approx(0.2), pytest.approx(-0.3)),
    ],
)
def test_phase_transformation(rz_before, rz_after, phase, phase_increment):
    assert phase_transformation(rz_before, rz_after) == (phase, phase_increment)


def test_FluxMultiplexer_SampleLinear(schedule_builder):
    # add calibration data for the multiplexer
    flux_elements = ["QB2", "TC-1-2", "TC-2-5"]
    elements = {}
    for idx1, flux_element1 in enumerate(flux_elements):
        for idx2, flux_element2 in enumerate(flux_elements):
            elements[f"{flux_element1}__{flux_element2}"] = idx1 * 0.01 + idx2 * 0.001
    schedule_builder.calibration["flux_multiplexer"] = {
        "sample_linear": {
            (): {"matrix_index": list(elements.keys()), "matrix_elements": np.array(list(elements.values()))}
        }
    }
    # do the actual multiplexing
    cz12 = schedule_builder.cz(("QB1", "QB2"), impl_name="tgss_crf")()
    cz25 = schedule_builder.cz(("QB2", "QB5"), impl_name="tgss_crf")()
    multiplexed_box = schedule_builder.flux_multiplexer((), impl_name="sample_linear")([cz12, cz25])
    for flux_element in flux_elements:
        assert len(multiplexed_box.atom[f"{flux_element}__flux.awg"]) == 2
        for inst in multiplexed_box.atom[f"{flux_element}__flux.awg"]:
            assert isinstance(inst, RealPulse)
