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
import pytest

from iqm.pulse.playlist.instructions import Instruction, IQPulse, RealPulse, VirtualRZ, Wait
from iqm.pulse.playlist.schedule import Nothing, Schedule
from iqm.pulse.playlist.waveforms import Gaussian
from iqm.pulse.scheduler import Block, SegmentPointer, extend_schedule, extend_schedule_new

# TODO: remove the above skips once COMP-1281 is done and the TETRIS mode is fixed.
pytest.skip(allow_module_level=True)

w = Wait(25)
p1 = RealPulse(25, Gaussian(2, 4), 0.7)
p2 = RealPulse(40, Gaussian(2, 4), 1.0)
n = Nothing(duration=15)
z = Block(0)


def test_segment_pointer_next():
    """Test moving segment pointer to next instruction"""
    instructions: list[Instruction] = [w, p1, n, p2, n]
    pointer = SegmentPointer(source=instructions, idx=0, frac=0.0, TOL=5.0)
    for i in range(len(instructions) - 1):
        assert pointer.get() == instructions[i]
        assert not pointer.next()
    assert pointer.get() == instructions[-1]
    assert pointer.next()  # End of list reached


def test_segment_pointer_fastforward():
    """Test moving segment pointer forward"""
    instructions: list[Instruction] = [w, p1, n, p2, n]
    pointer = SegmentPointer(source=instructions, idx=0, frac=0.0, TOL=5.0)
    with pytest.raises(ValueError, match="trying to fastforward over solid things! Wait\\(duration=25\\)"):
        pointer.fastforward(10.0)
    pointer.idx = 2
    pointer.frac = 0.0
    assert pointer.get() == instructions[2]
    assert not pointer.fastforward(n.duration - pointer.TOL + 0.5)
    assert pointer.idx == 3
    assert not pointer.next()
    pointer.frac = 0.0
    assert pointer.idx == 4
    assert not pointer.fastforward(pointer.TOL + 0.5)
    assert pointer.fastforward(n.duration)  # Ran out of sequence
    assert pointer.idx == len(instructions)


def test_segment_pointer_fastforward_with_zero_length_segment():
    instructions: list[Instruction] = [z]
    pointer = SegmentPointer(source=instructions, idx=0, frac=0.0, TOL=5.0)
    assert pointer.fastforward(0.0)
    assert pointer.idx == 1
    assert pointer.frac == 0.0
    instructions: list[Instruction] = [z, w]
    pointer = SegmentPointer(source=instructions, idx=0, frac=0.0, TOL=5.0)
    assert not pointer.fastforward(0.0)
    assert pointer.idx == 1
    assert pointer.frac == 0.0


def test_segment_pointer_rewind():
    """Test moving segment pointer backward"""
    instructions: list[Instruction] = [w, p1, n, p2, n]
    pointer = SegmentPointer(source=instructions, idx=4, frac=10.0, TOL=5.0)
    assert not pointer.rewind(9.0)  # Rewind to idx == 4, frac == 1.0 ( < TOL)
    assert pointer.idx == 4
    assert pointer.frac == 0.0  # frac was normalized
    assert not pointer.rewind(instructions[3].duration - pointer.TOL - 1.0)  # Rewind to idx == 3, frac == TOL + 1.0
    assert pointer.idx == 3
    assert pointer.frac == pointer.TOL + 1.0  # frac was not normalized
    assert not pointer.rewind(sum(instructions[i].duration for i in range(pointer.idx)))  # Rewind to idx == 0
    assert pointer.idx == 0
    assert pointer.frac == pointer.TOL + 1.0
    with pytest.raises(IndexError, match="rewinded too far"):
        pointer.rewind(instructions[0].duration)


def test_extend_schedule(schedule_builder):
    """Test extending a Schedule with another Schedule"""

    schedule_1 = Schedule(
        {
            "QB1__drive.awg": [
                IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)),
                Wait(duration=10),
                VirtualRZ(duration=20, phase_increment=0.2),
                Block(15),
            ],
        }
    )
    assert schedule_1.duration() == 50.0

    schedule_2 = Schedule(
        {
            "QB1__drive.awg": [
                Block(15),
                IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)),
            ],
            "QB2__drive.awg": [
                Wait(duration=10),
                IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)),
            ],
        }
    )
    assert schedule_2.duration() == 20.0

    schedule_3 = Schedule(
        {
            "QB1__drive.awg": [
                Block(15),
            ],
        }
    )
    assert schedule_3.duration() == 15.0

    combined = schedule_1.copy()
    assert extend_schedule(combined, schedule_2, schedule_builder.channels) is None
    assert extend_schedule(combined, schedule_3, schedule_builder.channels) is None
    assert combined.duration() == 70.0


class TestExtendScheduleNew:
    def test_extend(self, schedule_builder):
        """Test extending a Schedule with another Schedule"""

        # no overlap, gap: aa bbbb
        # overlap        : aaaXbb
        A = Schedule(
            {
                "QB1__drive.awg": [
                    IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)),
                    VirtualRZ(6.0, phase_increment=0.2),
                ],
                "QB2__drive.awg": [
                    Wait(14.0),
                    Block(3.0),
                ],
            }
        )
        assert A.duration() == pytest.approx(17.0)

        B = Schedule(
            {
                "QB1__drive.awg": [
                    IQPulse(16.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)),
                ],
                "QB2__drive.awg": [
                    Nothing(4.0),
                    Wait(10.0),
                ],
            }
        )
        assert B.duration() == pytest.approx(16.0)

        combined = A.copy()
        assert extend_schedule_new(combined, B, schedule_builder.channels) is None
        assert combined.duration() == 29.0

    def test_extend_with_empty(self, schedule_builder):
        """Test extending a Schedule with an empty Schedule"""

        A = Schedule(
            {
                "xxx": [
                    IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)),
                    VirtualRZ(20.0, phase_increment=0.2),
                ],
                "yyy": [Block(15.0)],
                "zzz": [
                    Wait(10.0),
                    Nothing(10.0),
                ],
                "www": [
                    Nothing(18.0),
                ],
            }
        )
        assert A.duration() == pytest.approx(25.0)
        assert len(A) == 4

        B = Schedule()
        combined = A.copy()
        assert extend_schedule_new(combined, B, schedule_builder.channels) is None
        assert combined.duration() == pytest.approx(25.0)
        assert len(combined) == 4

    @pytest.mark.parametrize(
        "A,B,A_duration,B_duration,C_duration,C_len",
        [
            pytest.param(
                [Wait(1.0), Block(2.0)],
                [Wait(4.0), RealPulse(8.0, Gaussian(10, sigma=0.1), 0.7)],
                3.0,
                12.0,
                15.0,
                4,
                id="aaaabbbb",  # no overlap
            ),
            pytest.param(
                [Wait(1.0), Block(2.0)],
                [Block(3.0), RealPulse(8.0, Gaussian(10, sigma=0.1), 0.7)],
                3.0,
                11.0,
                12.0,
                4,
                id="aaaXXbbb",  # X denotes overlap of A and B, A ends first
            ),
            pytest.param(
                [Block(2.0), Nothing(3.0)],
                [Block(3.0), RealPulse(4.0, Gaussian(10, sigma=0.1), 0.7)],
                5.0,
                7.0,
                7.0,
                3,
                id="XXXXXbb",
            ),
            pytest.param([Wait(2.0), Nothing(5.0)], [Block(4.0)], 7.0, 4.0, 7.0, 3, id="aaaXXXaa"),  # B ends first
            pytest.param(
                [Block(2.0), Nothing(5.0)],
                [Block(3.0), RealPulse(2.0, Gaussian(10, sigma=0.1), 0.7)],
                7.0,
                5.0,
                7.0,
                4,
                id="XXXaaaa",
            ),
            pytest.param(
                [Block(2.0), Nothing(5.0)],
                [RealPulse(1.0, Gaussian(10, sigma=0.1), 0.7), Block(4.0)],
                7.0,
                5.0,
                7.0,
                3,
                id="aaXXX",  # A and B end simultaneously
            ),
            pytest.param(
                [Block(2.0), Nothing(5.0)],
                [Block(3.0), RealPulse(4.0, Gaussian(10, sigma=0.1), 0.7)],
                7.0,
                7.0,
                7.0,
                3,
                id="XXXXX",
            ),
            pytest.param(
                [Wait(duration=5), IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)), Nothing(5)],
                [Nothing(8), IQPulse(5.0, Gaussian(10, sigma=0.1), Gaussian(10, sigma=0.1)), Wait(5)],
                15.0,
                18.0,
                20.0,
                4,
                id="aaaXXbbb, longer",
            ),
        ],
    )
    def test_overlap(self, schedule_builder, A, B, A_duration, B_duration, C_duration, C_len):
        """Test extending a schedule with overlap handling"""

        ch = "QB1__drive.awg"
        A = Schedule({ch: A})
        assert A.duration() == pytest.approx(A_duration)

        B = Schedule({ch: B})
        assert B.duration() == pytest.approx(B_duration)

        combined = A.copy()
        assert extend_schedule_new(combined, B, schedule_builder.channels) is None
        assert combined.duration() == pytest.approx(C_duration)
        assert len(combined[ch]) == C_len

    def test_overlap_A_is_just_one_solid_item(self, schedule_builder):
        A = Schedule({"QB1__drive.awg": [Wait(4.0)]})
        B = Schedule(
            {
                "QB1__drive.awg": [Nothing(1.0), Wait(3.0)],
            }
        )
        combined = A.copy()
        assert extend_schedule_new(combined, B, schedule_builder.channels) is None
        assert combined.duration() == pytest.approx(7.0)
