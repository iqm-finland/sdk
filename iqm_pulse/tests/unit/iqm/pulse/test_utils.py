#  ********************************************************************************
#
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
from dataclasses import replace

import numpy as np
import pytest

from iqm.pulse.base_utils import merge_dicts
from iqm.pulse.playlist.instructions import IQPulse
from iqm.pulse.playlist.waveforms import Constant, Samples
from iqm.pulse.utils import fuse_iq_pulses, modulate_iq


def test_merge_dict_works_as_expected():
    """Test that utils.merge_dict works as expected"""
    dict_a = {
        "key_1": "value_1",
        "key_2": 2,
        "key_3": ["a", "b", "c"],
        "key_4": {
            "key_4_1": "value_4_1",
            "key_4_2": 42,
            "key_4_3": ["d", "e", "f"],
            "key_4_4": {
                "key_4_4_1": "value_4_4_1",
            },
        },
    }
    dict_b = {
        "key_b_1": "value_b_1",
        "key_2": -2,
        "key_4": {
            "key_4_2": "value_4_2",
            "key_4_4": {"key_4_4_1": 431},
        },
    }
    expected = {
        "key_1": "value_1",
        "key_b_1": "value_b_1",  # item inserted
        "key_2": -2,  # scalar item replaced
        "key_3": ["a", "b", "c"],
        "key_4": {
            "key_4_1": "value_4_1",
            "key_4_2": "value_4_2",  # nested item replaced
            "key_4_3": ["d", "e", "f"],
            "key_4_4": {"key_4_4_1": 431},  # nested dict item replaced
        },
    }

    assert merge_dicts(dict_a, dict_b) == expected


def test_merge_dict_fails_when_merging_scalar_with_dict():
    """Test that utils.merge_dict fails when merging scalar with dict"""
    dict_a = {
        "key_1": "value_1",
        "key_2": 2,
        "key_3": ["a", "b", "c"],
    }
    dict_b = {
        "key_2": {"key_2_1": "value_2_1"},
    }

    with pytest.raises(ValueError, match="Merging scalar with dict: key_2"):
        merge_dicts(dict_a, dict_b)


def test_merge_dict_fails_when_merging_dict_with_scalar():
    """Test that utils.merge_dict fails when merging scalar with dict"""
    dict_a = {
        "key_1": {
            "key_1_1": {
                "key_1_1_1": {"key": "value"},
            },
        },
        "key_2": 2,
    }
    dict_b = {
        "key_1": {
            "key_1_1": {
                "key_1_1_1": 111,
            },
        },
    }

    with pytest.raises(ValueError, match="Merging dict with scalar: key_1.key_1_1.key_1_1_1"):
        merge_dicts(dict_a, dict_b)


def test_merge_dict_with_empty_collections():
    dict_a = {"key_1": "value_1", "key_2": {1, 2, 3}, "key_3": ["a", "b", "c"], "key_4": 1, "key_5": "foo"}
    dict_b = {"key_1": "", "key_2": set(), "key_3": [], "key_4": 2}
    merged_dict = merge_dicts(dict_a, dict_b)
    assert merged_dict == {"key_1": "", "key_2": set(), "key_3": [], "key_4": 2, "key_5": "foo"}
    merged_dict = merge_dicts(dict_a, dict_b, merge_nones=False)
    assert merged_dict == {"key_1": "value_1", "key_2": {1, 2, 3}, "key_3": ["a", "b", "c"], "key_4": 2, "key_5": "foo"}


n_samples = 4
iq_pulse = IQPulse(
    n_samples,
    Constant(n_samples),
    Constant(n_samples),
    scale_i=1.0,
    scale_q=0.5,
    phase=0.0,
    phase_increment=0.0,
    modulation_frequency=0.0,
)


@pytest.mark.parametrize(
    "pulse,expected_samples",
    [
        (
            # no modulation, zero phase
            iq_pulse,
            np.array([1.0 + 0.5j, 1.0 + 0.5j, 1.0 + 0.5j, 1.0 + 0.5j]),
        ),
        (
            # nonzero phase
            replace(iq_pulse, phase=np.pi / 2),
            np.array([1j - 0.5, 1j - 0.5, 1j - 0.5, 1j - 0.5]),
        ),
        (
            # nonzero modulation frequency
            replace(iq_pulse, modulation_frequency=0.25),
            np.array([1 + 0.5j, 1j - 0.5, -1 - 0.5j, -1j + 0.5]),
        ),
        (
            # nonzero phase and modulation frequency
            replace(iq_pulse, phase=np.pi / 2, modulation_frequency=0.25),
            np.array([1j - 0.5, -1 - 0.5j, -1j + 0.5, 1 + 0.5j]),
        ),
        (
            # phase_increment has no effect
            replace(iq_pulse, phase=np.pi / 2, modulation_frequency=0.25, phase_increment=1.234),
            np.array([1j - 0.5, -1 - 0.5j, -1j + 0.5, 1 + 0.5j]),
        ),
    ],
)
def test_modulate_iq(pulse, expected_samples):
    """modulate_iq returns the correct waveform samples."""
    modulated_samples = modulate_iq(pulse)
    assert len(modulated_samples) == pulse.duration
    assert modulated_samples == pytest.approx(expected_samples)


@pytest.mark.parametrize(
    "p1, p2, expected_samples, expected_phase, expected_scale_i, expected_scale_q",
    [
        (  # fusing preserves the waveform of the first pulse
            replace(iq_pulse, scale_q=0.0, phase=np.pi / 2, phase_increment=0.1),
            replace(iq_pulse, modulation_frequency=0.25, phase_increment=np.pi),
            np.array([1, 1, 1, 1] + [1j - 0.5, -1 - 0.5j, -1j + 0.5, 1 + 0.5j]),
            -np.pi / 2,
            1.0,
            1.0,
        ),
        (  # fusing normalizes the waveform components
            replace(iq_pulse, scale_q=0.0, phase_increment=0.777),
            replace(iq_pulse, scale_i=0.0, phase=np.pi),
            np.array([1, 1, 1, 1] + [-1j, -1j, -1j, -1j]),
            0,
            1.0,
            0.5,
        ),
    ],
)
def test_fuse_iq_pulses(p1, p2, expected_samples, expected_phase, expected_scale_i, expected_scale_q):
    """fuse_iq_pulses returns the correct output."""

    fused = fuse_iq_pulses([p1, p2])
    assert fused.duration == p1.duration + p2.duration
    assert isinstance(fused.wave_i, Samples)
    assert isinstance(fused.wave_q, Samples)
    assert fused.wave_i.samples == pytest.approx(expected_samples.real)
    assert fused.wave_q.samples == pytest.approx(expected_samples.imag)
    assert fused.scale_i == pytest.approx(expected_scale_i)
    assert fused.scale_q == pytest.approx(expected_scale_q)
    assert fused.phase == expected_phase
    assert fused.phase_increment == pytest.approx(p1.phase_increment + p2.phase_increment)
    assert fused.modulation_frequency == 0.0
