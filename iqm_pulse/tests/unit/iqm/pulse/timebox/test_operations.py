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
from iqm.pulse.playlist import Schedule, Segment
from iqm.pulse.playlist.instructions import (
    Block,
    ComplexIntegration,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
    ThresholdStateDiscrimination,
)
from iqm.pulse.playlist.waveforms import Constant
from iqm.pulse.timebox import MultiplexedProbeTimeBox, ProbeTimeBoxes, SchedulingStrategy, TimeBox


def get_atoms() -> tuple[TimeBox, TimeBox, TimeBox]:
    return (
        TimeBox.atomic(Schedule(), locus_components=["QB1", "QB2"], label="a"),
        TimeBox.atomic(Schedule(), locus_components=["QB1", "QB2"], label="b"),
        TimeBox.atomic(Schedule(), locus_components=["QB3", "QB4"], label="c"),
    )


def get_readout_boxes() -> tuple[MultiplexedProbeTimeBox, MultiplexedProbeTimeBox]:
    iq1 = IQPulse(16, wave_i=Constant(16), wave_q=Constant(16), scale_i=0.1, scale_q=0.0)
    iq2 = IQPulse(32, wave_i=Constant(16), wave_q=Constant(16), scale_i=0.2, scale_q=0.0)
    weights1 = IQPulse(16, wave_i=Constant(16), wave_q=Constant(16), scale_i=1.0, scale_q=0.0)
    weights2 = IQPulse(32, wave_i=Constant(16), wave_q=Constant(16), scale_i=1.0, scale_q=0.0)
    acq1 = ComplexIntegration("foo", 16, weights=weights1, implementation="measure.constant")
    acq2 = ThresholdStateDiscrimination("bar", 32, weights=weights2, threshold=0.5, implementation="measure.constant")
    mp_iq1 = MultiplexedIQPulse(entries=((iq1, 8),), duration=16)
    mp_iq2 = MultiplexedIQPulse(entries=((iq2, 8),), duration=32)
    rt1 = ReadoutTrigger(probe_pulse=mp_iq1, acquisitions=(acq1,), duration=32)
    rt2 = ReadoutTrigger(probe_pulse=mp_iq2, acquisitions=(acq2,), duration=64)
    box1 = MultiplexedProbeTimeBox.from_readout_trigger(rt1, "probe_channel", ("QB1",))
    box1.neighborhood_components[0] = {"QB1", "QB4", "PL-A"}
    box1.neighborhood_components[1] = {"QB1", "QB4", "PL-A", "QB500"}
    box1.neighborhood_components[3] = {"QB666"}
    box2 = MultiplexedProbeTimeBox.from_readout_trigger(rt2, "probe_channel", ("QB2",))
    box2.neighborhood_components[0] = {"QB2", "QB7", "PL-B"}
    box2.neighborhood_components[2] = {"QB2", "QB7", "PL-B", "QB1000"}
    box2.neighborhood_components[3] = {"QB999"}
    return box1, box2


def test_add():
    box_a, box_b, box_c = get_atoms()
    total = box_a + box_b + box_c

    assert total.locus_components == {"QB1", "QB2", "QB3", "QB4"}
    assert total.children == (box_a, box_b, box_c)


def test_add_is_associative():
    box_a, box_b, box_c = get_atoms()
    assert (box_a + box_b) + box_c == box_a + (box_b + box_c) == box_a + box_b + box_c


def test_add_iterable():
    box_a, box_b, box_c = get_atoms()
    assert (box_a, box_b) + box_c == box_a + (box_b, box_c) == box_a + box_b + box_c
    assert box_a + [] is box_a
    assert [] + box_a is box_a


def test_iadd():
    box_a, box_b, box_c = get_atoms()
    total = box_a + box_b

    total += box_c

    assert total.locus_components == {"QB1", "QB2", "QB3", "QB4"}
    assert total.children == (box_a, box_b, box_c)


def test_pipe():
    box_a, box_b, box_c = get_atoms()
    sequence_ab = box_a | box_b

    assert sequence_ab.locus_components == {"QB1", "QB2"}
    assert sequence_ab[0] is box_a
    assert sequence_ab[1] is box_b

    sequence_abc = sequence_ab | box_c

    assert sequence_abc.locus_components == {"QB1", "QB2", "QB3", "QB4"}
    assert sequence_abc[0][0] is box_a
    assert sequence_abc[0][1] is box_b
    assert sequence_abc[1] is box_c

    assert sequence_abc == box_a | box_b | box_c


def test_pipe_iterable():
    box_a, box_b, box_c = get_atoms()
    assert box_a | [box_b, box_c] == box_a | box_b + box_c
    assert [box_a, box_b] | box_c == box_a + box_b | box_c


def test_order_of_operations():
    box_a, box_b, box_c = get_atoms()
    sequence = (box_a + box_a).set_alap() | box_c | box_c + box_c + box_b

    assert sequence.locus_components == {"QB1", "QB2", "QB3", "QB4"}
    assert sequence[0][0].children == (box_a, box_a)
    assert sequence[0][0].scheduling == SchedulingStrategy.ALAP
    assert sequence[0][0].label == "a"

    assert sequence[0][1] is box_c

    assert sequence[1].children == (box_c, box_c, box_b)
    assert sequence[1].scheduling == SchedulingStrategy.ASAP
    assert sequence[1].label == "c"

    assert sequence.scheduling == SchedulingStrategy.ASAP


def test_docstring_example():
    box_a, box_b, box_c = get_atoms()
    box_d = TimeBox.atomic(Schedule(), locus_components=["QB3", "QB4"], label="d")

    a_then_b = box_a + box_b
    c_then_d = (box_c + box_d).set_alap()
    abcd = a_then_b | c_then_d

    abb = box_a + [box_b, box_b]
    ccd = [box_c, box_c] | box_d

    all_together = box_a + box_b | (box_c + box_d).set_alap() | box_a + box_b + box_b + (box_c + box_c | box_d)
    all_together.print()
    assert abcd | abb + ccd == all_together


def test_recursive_composite_flattens_lists():
    box_a, box_b, box_c = get_atoms()
    assert TimeBox.composite([box_a, [box_b, [box_c, box_b, box_a]]]) == TimeBox.composite(
        [box_a, box_b, box_c, box_b, box_a]
    )


def test_multiplexed_add():
    ro_box_1, ro_box_2 = get_readout_boxes()
    multiplexed = ro_box_1 + ro_box_2
    assert isinstance(multiplexed, MultiplexedProbeTimeBox)
    assert multiplexed.locus_components == {"QB1", "QB2"}
    assert multiplexed.atom is not None
    assert len(multiplexed.atom["probe_channel"]) == 1
    assert multiplexed.neighborhood_components == {
        0: {"QB1", "QB2", "QB4", "QB7", "PL-A", "PL-B"},
        3: {"QB666", "QB999"},
    }
    mrt = multiplexed.atom["probe_channel"][0]
    assert mrt.duration == 64
    assert isinstance(mrt, ReadoutTrigger)
    assert isinstance(mrt.probe_pulse, MultiplexedIQPulse)
    assert len(mrt.probe_pulse.entries) == 2
    assert isinstance(mrt.probe_pulse.entries[0][0], IQPulse)
    assert mrt.probe_pulse.entries[0][0].scale_i == 0.1
    assert isinstance(mrt.probe_pulse.entries[1][0], IQPulse)
    assert mrt.probe_pulse.entries[1][0].scale_i == 0.2
    assert len(mrt.acquisitions) == 2
    assert isinstance(mrt.acquisitions[0], ComplexIntegration)
    assert isinstance(mrt.acquisitions[1], ThresholdStateDiscrimination)


def test_multiplexed_boxes_add_with_normal_boxes():
    box_a, box_b, box_c = get_atoms()
    ro_box_1, ro_box_2 = get_readout_boxes()
    box = box_a + ro_box_1
    assert isinstance(box, TimeBox)
    assert not isinstance(box, MultiplexedProbeTimeBox)
    assert not isinstance(box.children[0], MultiplexedProbeTimeBox)
    assert isinstance(box.children[1], MultiplexedProbeTimeBox)
    box = ro_box_2 + box_b
    assert isinstance(box, TimeBox)
    assert not isinstance(box, MultiplexedProbeTimeBox)
    assert isinstance(box.children[0], MultiplexedProbeTimeBox)
    assert not isinstance(box.children[1], MultiplexedProbeTimeBox)
    box = ro_box_1 + [box_a, box_c]
    assert isinstance(box, TimeBox)
    assert not isinstance(box, MultiplexedProbeTimeBox)
    assert isinstance(box.children[0], MultiplexedProbeTimeBox)
    assert not isinstance(box.children[1], MultiplexedProbeTimeBox)
    assert not isinstance(box.children[2], MultiplexedProbeTimeBox)
    box = [box_a, box_c] + ro_box_1
    assert isinstance(box, TimeBox)
    assert not isinstance(box, MultiplexedProbeTimeBox)
    assert not isinstance(box.children[0], MultiplexedProbeTimeBox)
    assert not isinstance(box.children[1], MultiplexedProbeTimeBox)
    assert isinstance(box.children[2], MultiplexedProbeTimeBox)


def test_multiplexing_add_with_probe_boxes():
    ro_box_1, ro_box_2 = get_readout_boxes()
    virtual_wait_box1 = TimeBox.atomic(
        Schedule({"probe_virtual": Segment([Block(80)])}), locus_components=[], label="horse"
    )
    virtual_wait_box2 = TimeBox.atomic(
        Schedule({"probe_virtual": Segment([Block(160)])}), locus_components=[], label="horse"
    )
    # multiplex two probe boxes instances
    probe_boxes1 = ProbeTimeBoxes([ro_box_1, virtual_wait_box1])
    probe_boxes2 = ProbeTimeBoxes([ro_box_2, virtual_wait_box2])
    multiplexed = probe_boxes1 + probe_boxes2
    assert len(multiplexed) == 2
    assert isinstance(multiplexed, ProbeTimeBoxes)
    assert multiplexed[0].atom.duration == 64
    assert multiplexed[1].atom.duration == 160
    # multiplex probe boxes with a readout box
    multiplexed = probe_boxes1 + ro_box_2
    assert isinstance(multiplexed, MultiplexedProbeTimeBox)
    assert multiplexed.atom.duration == 64
    multiplexed = ro_box_2 + probe_boxes1
    assert isinstance(multiplexed, MultiplexedProbeTimeBox)
    assert multiplexed.atom.duration == 64
    # add with a normal TimeBox yields just a composite TimeBox
    box_a, box_b, _ = get_atoms()
    result = probe_boxes1 + box_a
    assert isinstance(result, TimeBox)
    assert len(result.children) == 3
    result = box_a + probe_boxes1
    assert isinstance(result, TimeBox)
    assert len(result.children) == 3
    # add with a list of boxes yields a list of boxes
    result = probe_boxes1 + [box_a, box_b]
    assert isinstance(result, list)
    assert len(result) == 4
