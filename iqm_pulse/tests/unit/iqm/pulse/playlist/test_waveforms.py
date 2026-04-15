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
from dataclasses import dataclass, field

from iqm.models.playlist.waveforms import to_canonical
import numpy as np
import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from iqm.pulse.gate_implementation import get_waveform_parameters
from iqm.pulse.playlist.waveforms import (
    Constant,
    Cosine,
    CosineRiseFall,
    CosineRiseFallDerivative,
    Gaussian,
    GaussianDerivative,
    GaussianSmoothedSquare,
    ModulatedCosineRiseFall,
    PolynomialCosine,
    Samples,
    Slepian,
    TruncatedGaussianSmoothedSquare,
    Waveform,
)


class DummyWaveform(Waveform):
    """Dummy waveform for testing"""

    def sample(self) -> np.ndarray:
        return np.ones([self.n_samples])

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return np.ones(sample_coords.shape)


@dataclass(frozen=True)
class DummyWithManyAttributes(Waveform):
    """Dummy waveform with a bunch of different args for testing"""

    float_arg: float
    str_arg: str
    bool_arg: bool
    complex_arg: complex
    float_list_arg: list[float]
    str_list_arg: list[str]
    complex_array_arg: np.ndarray
    default_arg1: float = 0.0
    default_arg2: str = "foo"
    default_arg3: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    def sample(self) -> np.ndarray:
        return np.ones([self.n_samples])

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return np.ones(sample_coords.shape)

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {
            "str_arg": "",
            "bool_arg": "",
            "complex_arg": "V",
            "complex_array_arg": "V",
            "str_list_arg": "",
            "default_arg2": "",
        }


@dataclass(frozen=True)
class Dummywithoutcapitalisedwords(Waveform):
    """Dummy waveform whose class name does not conform to the OOP class naming."""

    some_attribute: float

    def sample(self) -> np.ndarray:
        return np.ones([self.n_samples])

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return np.ones(sample_coords.shape)


def test_dummy_waveform():
    """Test converting a non-canonical waveform to a canonical waveform"""
    waveform = DummyWaveform(10)
    samples = to_canonical(waveform)
    assert isinstance(samples, Samples)
    assert len(samples.sample()) == waveform.n_samples


def test_samples():
    """Test creating a Samples waveform"""
    waveform = to_canonical(Samples(np.array([0.0] * 10)))
    assert isinstance(waveform, Samples)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10

    regexp = "Incorrect shape \\(3, 10\\) for waveform samples. Samples should be a 1D array."
    with pytest.raises(ValueError, match=regexp):
        Samples(np.array([[0.0] * 10] * 3))

    assert waveform == Samples(np.array([0.0] * 10))
    assert hash(waveform) == hash(Samples(np.array([0.0] * 10)))


def test_gaussian():
    """Test creating a Gaussian waveform"""
    waveform = Gaussian(10, sigma=0.2, center_offset=0.25)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10


def test_gaussian_gives_expected_result():
    samples = Gaussian(3, 0.2, 0).sample()
    assert samples == pytest.approx([0.25, 1, 0.25], abs=0.01)


def test_gaussian_samples_are_asymptotically_zero():
    samples = Gaussian(101, 0.01, 0).sample()
    assert len(samples) == 101
    assert samples[0] == 0.0
    assert samples[50] == 1.0
    assert samples[100] == 0.0


def test_gaussian_gives_expected_result_with_width_varied():
    samples = Gaussian(5, center_offset=0, sigma=0.8 / 5).sample()
    assert samples == pytest.approx([0.04, 0.46, 1.0, 0.46, 0.04], abs=0.01)


def test_gaussian_derivative():
    """Test creating a GaussianDerivative waveform"""
    waveform = GaussianDerivative(10, sigma=0.2, center_offset=0.25)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10


def test_gaussian_derivative_gives_expected_times():
    samples = GaussianDerivative(5, sigma=1 / 5, center_offset=0).sample()
    assert samples == pytest.approx([0.44, 1, 0.0, -1, -0.44], abs=0.1)


def test_gaussian_derivative_result_is_asymptotically_zero():
    samples = GaussianDerivative(101, center_offset=0, sigma=0.01).sample()
    assert len(samples) == 101
    assert samples[0] == pytest.approx(0.0)
    assert samples[100] == pytest.approx(0.0)


def test_gaussian_derivative_result_is_asymmetric():
    samples = GaussianDerivative(5, sigma=1 / 5, center_offset=0).sample()
    assert len(samples) == 5
    assert samples[2] == pytest.approx(0)
    assert samples[1] == pytest.approx(-samples[3])


def test_gss():
    """Test creating a GaussianSmoothedSquare waveform"""
    waveform = GaussianSmoothedSquare(10, square_width=0.5, gaussian_sigma=0.2, center_offset=0.25)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10


def test_gss_produces_correct_waveform():
    samples = GaussianSmoothedSquare(6, square_width=2.5 / 6, gaussian_sigma=0.25 / 6, center_offset=0).sample()
    assert samples == pytest.approx(
        [
            0.0,
            0.159,
            0.999,
            0.999,
            0.159,
            0.0,
        ],
        abs=0.001,
    )


def test_tgss():
    """Test creating a TruncatedGaussianSmoothedSquare waveform"""
    waveform = TruncatedGaussianSmoothedSquare(10, full_width=0.5, rise_time=0.2, center_offset=0.25)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10


def test_tgss_produces_correct_waveform():
    samples = TruncatedGaussianSmoothedSquare(
        6, full_width=2.5 / 6 + 0.25 / 6 * 5.49563, rise_time=0.25 / 6 * 5.49563, center_offset=0
    ).sample()
    assert samples == pytest.approx(
        [
            0.0,
            0.156,
            0.999,
            0.999,
            0.156,
            0.0,
        ],
        abs=0.001,
    )


def test_constant():
    """Test creating a Constant waveform"""
    waveform = Constant(10)
    assert waveform.n_samples == 10
    assert list(waveform.sample()) == [1.0] * 10


def test_cosinerisefall():
    """Test creating a CosineRiseFall waveform.
    Furthermore, test whether first value is equal to zero and middle value is equal to one."""
    waveform = CosineRiseFall(10, full_width=0.45, rise_time=0.1, center_offset=0.05)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10

    assert waveform.sample()[0] == 0
    assert waveform.sample()[int(len(waveform.sample()) / 2)] == 1


def test_cosinerisefall_limits():
    waveform = CosineRiseFall(101, 1, 0.5, 0).sample()
    assert pytest.approx(waveform[0], abs=0.001) == 0
    assert pytest.approx(waveform[-1], abs=0.001) == 0
    assert pytest.approx(waveform[50], abs=0.001) == 1


def test_symmetric_cosinerisefall():
    """Test whether the waveform is indeed symmetric if center_offset = 0"""
    waveform = CosineRiseFall(100, full_width=0.45, rise_time=0.1, center_offset=0.0).sample()

    assert np.allclose(waveform, np.flip(waveform), atol=1e-10)


def test_continuous_limit_cosinerisefall():
    """Test whether the mean of the waveform converges to the FWHM for sufficiently many samples and independent
    of the rise time."""
    plateau_width = 0.35
    for rise_time in [0.1, 0.2, 0.3]:
        full_width = plateau_width + 2 * rise_time
        waveform = CosineRiseFall(1000, full_width=full_width, rise_time=rise_time, center_offset=0.04).sample()
        assert pytest.approx(np.mean(waveform), abs=0.002) == plateau_width + rise_time


def test_cosinerisefall_derivative():
    """Test creating a CosineRiseFallDerivate waveform.
    Furthermore, test whether first and middle values are equal to zero."""
    waveform = CosineRiseFallDerivative(10, full_width=0.45, rise_time=0.1, center_offset=0.05)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10

    assert waveform.sample()[0] == 0
    assert waveform.sample()[int(len(waveform.sample()) / 2)] == 0


def test_cosinerisefall_derivative_limits():
    waveform = CosineRiseFallDerivative(101, 1, 0.5, 0).sample()
    assert pytest.approx(waveform[0], abs=0.1) == 0
    assert pytest.approx(waveform[-1], abs=0.1) == 0
    assert pytest.approx(waveform[25], abs=0.001) == 1
    assert pytest.approx(waveform[50], abs=0.001) == 0
    assert pytest.approx(waveform[75], abs=0.001) == -1


def test_antisymmetric_cosinerisefall_derivative():
    """Test whether the waveform is indeed antisymmetric if center_offset = 0"""
    waveform = CosineRiseFallDerivative(100, full_width=0.45, rise_time=0.1, center_offset=0.0).sample()
    assert np.allclose(waveform, -np.flip(waveform), atol=1e-10)


def test_cosine():
    cosine = Cosine(101, -1, 0)
    waveform = cosine.sample()
    assert pytest.approx(waveform[0], abs=0.001) == -1
    assert pytest.approx(waveform[-1], abs=0.001) == -1
    assert pytest.approx(waveform[50], abs=0.001) == 1
    assert pytest.approx(waveform[25], abs=0.1) == 0
    assert pytest.approx(waveform[75], abs=0.1) == 0


def test_polynomial_cosine():
    poly_cosine = PolynomialCosine(301, 1, np.array([0, 0.9, 0.1]), np.pi / 2)
    waveform = poly_cosine.sample()
    assert pytest.approx(waveform[0], abs=0.001) == 0.0094  # should be 0 but half-interval sampling
    assert pytest.approx(waveform[-1], abs=0.001) == -0.0094
    assert pytest.approx(waveform[150], abs=0.001) == 0
    assert pytest.approx(waveform[75], abs=0.1) == 1.0
    assert pytest.approx(waveform[225], abs=0.1) == -0.8


def test_slepian():
    """Test creating a Slepian waveform.
    Furthermore, test whether first value is equal to zero and middle value is equal to one."""
    waveform = Slepian(
        n_samples=80,
        full_width=0.6,
        center_offset=0,
        lambda_1=-0.5,
        lambda_2=0.05,
        frequency_initial_normalized=0.7,
        frequency_to_minimize_normalized=0.9,
        coupling_strength_normalized=0.01,
        squid_asymmetry=0.4,
    )
    assert waveform.n_samples == 80
    samples = waveform.sample()
    assert len(samples) == 80

    assert samples[0] == 0
    assert samples[-1] == 0
    assert np.max(samples) == pytest.approx(1, abs=0.0001)
    assert samples[len(samples) // 2] == pytest.approx(1, abs=0.0001)


def test_modulatedcosinerisefall():
    """Test creating a CosineRiseFall waveform.
    Furthermore, test whether first value is equal to zero and middle value is equal to one."""
    waveform = ModulatedCosineRiseFall(10, full_width=0.9, rise_time=0.3, center_offset=0.0, modulation_frequency=1)
    assert waveform.n_samples == 10
    assert len(waveform.sample()) == 10

    assert waveform.sample()[0] == 0
    assert pytest.approx(waveform.sample()[int(len(waveform.sample()) / 2)], abs=0.1) == 1


def test_get_waveform_parameters():
    parameters = get_waveform_parameters(DummyWithManyAttributes)
    assert isinstance(parameters["float_arg"], Parameter)
    assert parameters["float_arg"].unit == "s"
    assert parameters["float_arg"].label == "float_arg of dwma"  # 'DummyWithManyAttributes' abbreviated
    assert parameters["float_arg"].data_type == DataType.FLOAT
    assert parameters["float_arg"].collection_type == CollectionType.SCALAR

    assert isinstance(parameters["str_arg"], Parameter)
    assert parameters["str_arg"].unit == ""
    assert parameters["str_arg"].label == "str_arg of dwma"
    assert parameters["str_arg"].data_type == DataType.STRING
    assert parameters["str_arg"].collection_type == CollectionType.SCALAR

    assert isinstance(parameters["bool_arg"], Parameter)
    assert parameters["bool_arg"].unit == ""
    assert parameters["bool_arg"].label == "bool_arg of dwma"
    assert parameters["bool_arg"].data_type == DataType.BOOLEAN
    assert parameters["bool_arg"].collection_type == CollectionType.SCALAR

    assert isinstance(parameters["complex_arg"], Parameter)
    assert parameters["complex_arg"].unit == "V"
    assert parameters["complex_arg"].label == "complex_arg of dwma"
    assert parameters["complex_arg"].data_type == DataType.COMPLEX
    assert parameters["complex_arg"].collection_type == CollectionType.SCALAR

    assert isinstance(parameters["float_list_arg"], Parameter)
    assert parameters["float_list_arg"].unit == "s"
    assert parameters["float_list_arg"].label == "float_list_arg of dwma"
    assert parameters["float_list_arg"].data_type == DataType.FLOAT
    assert parameters["float_list_arg"].collection_type == CollectionType.LIST

    assert isinstance(parameters["str_list_arg"], Parameter)
    assert parameters["str_list_arg"].unit == ""
    assert parameters["str_list_arg"].label == "str_list_arg of dwma"
    assert parameters["str_list_arg"].data_type == DataType.STRING
    assert parameters["str_list_arg"].collection_type == CollectionType.LIST

    assert isinstance(parameters["complex_array_arg"], Parameter)
    assert parameters["complex_array_arg"].unit == "V"
    assert parameters["complex_array_arg"].label == "complex_array_arg of dwma"
    assert parameters["complex_array_arg"].data_type == DataType.COMPLEX
    assert parameters["complex_array_arg"].collection_type == CollectionType.NDARRAY

    assert isinstance(parameters["default_arg1"], Setting)
    assert parameters["default_arg1"].unit == "s"
    assert parameters["default_arg1"].label == "default_arg1 of dwma"
    assert parameters["default_arg1"].parameter.data_type == DataType.FLOAT
    assert parameters["default_arg1"].parameter.collection_type == CollectionType.SCALAR
    assert parameters["default_arg1"].value == 0.0

    assert isinstance(parameters["default_arg2"], Setting)
    assert parameters["default_arg2"].unit == ""
    assert parameters["default_arg2"].label == "default_arg2 of dwma"
    assert parameters["default_arg2"].parameter.data_type == DataType.STRING
    assert parameters["default_arg2"].parameter.collection_type == CollectionType.SCALAR
    assert parameters["default_arg2"].value == "foo"

    assert isinstance(parameters["default_arg3"], Setting)
    assert parameters["default_arg3"].unit == "s"
    assert parameters["default_arg3"].label == "default_arg3 of dwma"
    assert parameters["default_arg3"].parameter.data_type == DataType.FLOAT
    assert parameters["default_arg3"].parameter.collection_type == CollectionType.LIST
    assert parameters["default_arg3"].value == [1.0, 2.0, 3.0]

    parameters_with_prefix = get_waveform_parameters(DummyWithManyAttributes, label_prefix="My cool prefix ")
    assert parameters_with_prefix["float_arg"].label == "My cool prefix float_arg of dwma"

    parameters_without_abbr = get_waveform_parameters(Dummywithoutcapitalisedwords)
    assert parameters_without_abbr["some_attribute"].label == "some_attribute of Dummywithoutcapitalisedwords"
