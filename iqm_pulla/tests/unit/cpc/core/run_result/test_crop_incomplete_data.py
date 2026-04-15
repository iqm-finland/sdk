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

import numpy as np
import pytest

from exa.common.data.parameter import DataType, Parameter, Sweep
from exa.common.sweep.util import NdSweep
from iqm.cpc.core.run_result import DataVariables, _crop_incomplete_data

RNG = np.random.default_rng()


def verify_sweep_and_data_variables(
    original_sweeps: NdSweep,
    sweeps: NdSweep,
    expected_sweep_sizes: list[int],
    original_data_variables: DataVariables,
    data_variables: DataVariables,
    expected_data_variable_sizes: int,
) -> None:
    assert len(expected_sweep_sizes) == len(sweeps), "Inconsistent expectations = incorrect test"
    assert len(sweeps) == len(original_sweeps)
    for expected_sweep_range, original_sweep_tuple, sweep_tuple in zip(expected_sweep_sizes, original_sweeps, sweeps):
        assert len(sweep_tuple) == len(original_sweep_tuple)
        for oswt, swt in zip(original_sweep_tuple, sweep_tuple):
            assert swt.parameter == oswt.parameter
            assert swt.data == oswt.data[:expected_sweep_range]

    assert len(data_variables) == len(original_data_variables)
    assert expected_data_variable_sizes == np.prod(expected_sweep_sizes), "Inconsistent expectations = incorrect test"
    for original_dv_tuple, dv_tuple in zip(original_data_variables, data_variables):
        # 0th of the tuple: parameter
        assert dv_tuple[0] == original_dv_tuple[0]


def verify_hard_sweeps(
    original_data_variables: DataVariables, data_variables: DataVariables, expected_sizes: list[int]
) -> None:
    for ind, variable in enumerate(data_variables):
        _, spots, shape = variable
        for ind2, sweep in enumerate(shape):
            assert len(sweep[0].data) == expected_sizes[ind2]
        for ind2, spot in enumerate(spots):
            assert len(spot) == np.prod(expected_sizes)
            assert list(spot) == list(original_data_variables[ind][1][ind2][: np.prod(expected_sizes)])


@pytest.fixture
def parameter1() -> Parameter:
    return Parameter(name="length", label="Length", unit="m", data_type=DataType.FLOAT)


@pytest.fixture
def parameter2() -> Parameter:
    return Parameter(name="volume", label="Volume", unit="L", data_type=DataType.FLOAT)


@pytest.fixture
def parameter3() -> Parameter:
    return Parameter(name="mass", label="Mass", unit="g", data_type=DataType.FLOAT)


@pytest.fixture
def parameter4() -> Parameter:
    return Parameter(name="frequency", label="Frequency", unit="Hz", data_type=DataType.FLOAT)


@pytest.fixture
def parameter5() -> Parameter:
    return Parameter(name="amplitude", label="Amplitude", unit="A", data_type=DataType.FLOAT)


def hard_parameter() -> Parameter:
    return Parameter(name="hard", label="Hard", unit="A", data_type=DataType.FLOAT)


def very_hard_parameter() -> Parameter:
    return Parameter(name="very_hard", label="Very Hard", unit="A", data_type=DataType.FLOAT)


@pytest.fixture
def sweeps_10(parameter1) -> NdSweep:
    return [(Sweep(parameter=parameter1, data=RNG.random(10).tolist()),)]


def construct_data_variables(parameters: list[Parameter], iterations: int) -> DataVariables:
    return [(parameter, RNG.random(iterations), []) for parameter in parameters]


def construct_data_variables_with_2_hard_sweeps(
    parameters: list[Parameter], iterations: int, lengths: list[int], cropped_length: int
) -> DataVariables:
    return [
        (
            parameter,
            np.array([RNG.random(cropped_length) for _ in range(iterations)]),
            [
                (Sweep(parameter=hard_parameter(), data=RNG.random(lengths[0]).tolist()),),
                (Sweep(parameter=very_hard_parameter(), data=RNG.random(lengths[1]).tolist()),),
            ],
        )
        for parameter in parameters
    ]


def test_1_dim_sweep_with_all_raw_results_present_nothing_is_cropped(parameter1, sweeps_10):
    original_data_variables = construct_data_variables([parameter1], 10)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10, original_data_variables)
    assert sweeps is sweeps_10
    assert data_variables is original_data_variables


def test_1_dim_sweep_with_incomplete_raw_results_will_be_cropped(parameter1, sweeps_10):
    original_data_variables = construct_data_variables([parameter1], 7)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10, original_data_variables)
    verify_sweep_and_data_variables(
        original_sweeps=sweeps_10,
        sweeps=sweeps,
        expected_sweep_sizes=[7],
        original_data_variables=original_data_variables,
        data_variables=data_variables,
        expected_data_variable_sizes=7,
    )


@pytest.fixture
def sweeps_10x3(parameter1, parameter2, parameter3) -> NdSweep:
    return [
        (Sweep(parameter=parameter1, data=RNG.random(10).tolist()),),
        (
            Sweep(parameter=parameter2, data=RNG.random(3).tolist()),
            Sweep(parameter=parameter3, data=RNG.random(3).tolist()),
        ),
    ]


def test_2_dim_sweep_with_all_raw_results_present_nothing_is_cropped(parameter1, parameter2, sweeps_10x3):
    original_data_variables = construct_data_variables([parameter1, parameter2], 30)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10x3, original_data_variables)
    assert sweeps is sweeps_10x3
    assert data_variables is original_data_variables


def test_2_dim_sweep_with_incomplete_raw_results_will_be_cropped(parameter1, sweeps_10x3):
    original_data_variables = construct_data_variables([parameter1, parameter2], 27)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10x3, original_data_variables)
    verify_sweep_and_data_variables(
        original_sweeps=sweeps_10x3,
        sweeps=sweeps,
        expected_sweep_sizes=[10, 2],
        original_data_variables=original_data_variables,
        data_variables=data_variables,
        expected_data_variable_sizes=20,
    )


@pytest.fixture
def sweeps_10x3x20x5(parameter1, parameter2, parameter3, parameter4, parameter5) -> NdSweep:
    return [
        (Sweep(parameter=parameter1, data=RNG.random(10).tolist()),),
        (
            Sweep(parameter=parameter2, data=RNG.random(3).tolist()),
            Sweep(parameter=parameter3, data=RNG.random(3).tolist()),
        ),
        (Sweep(parameter=parameter4, data=RNG.random(20).tolist()),),
        (Sweep(parameter=parameter5, data=RNG.random(5).tolist()),),
    ]


def test_5_dim_sweep_with_all_raw_results_present_nothing_is_cropped(
    parameter1, parameter2, parameter4, parameter5, sweeps_10x3x20x5
):
    original_data_variables = construct_data_variables([parameter1, parameter2, parameter4, parameter5], 3000)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10x3x20x5, original_data_variables)
    assert sweeps is sweeps_10x3x20x5
    assert data_variables is original_data_variables


def test_5_dim_sweep_with_incomplete_raw_results_will_be_cropped(
    parameter1, parameter2, parameter4, parameter5, sweeps_10x3x20x5
):
    original_data_variables = construct_data_variables([parameter1, parameter2, parameter4, parameter5], 100)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10x3x20x5, original_data_variables)
    verify_sweep_and_data_variables(
        original_sweeps=sweeps_10x3x20x5,
        sweeps=sweeps,
        expected_sweep_sizes=[10, 3, 3, 1],
        original_data_variables=original_data_variables,
        data_variables=data_variables,
        expected_data_variable_sizes=90,
    )


def test_2_hard_sweeps_and_no_soft_sweeps_will_be_cropped(parameter1):
    original_data_variables = construct_data_variables_with_2_hard_sweeps([parameter1], 1, [10, 10], 54)
    sweeps, data_variables = _crop_incomplete_data([], original_data_variables)
    verify_sweep_and_data_variables(
        original_sweeps=[],
        sweeps=sweeps,
        expected_sweep_sizes=[],
        original_data_variables=original_data_variables,
        data_variables=data_variables,
        expected_data_variable_sizes=1,
    )
    verify_hard_sweeps(original_data_variables, data_variables, [10, 5])


def test_hard_sweeps_wont_be_cropped_if_there_are_multiple_soft_sweep_spots(parameter1, sweeps_10):
    original_data_variables = construct_data_variables_with_2_hard_sweeps([parameter1], 7, [2, 2], 4)
    sweeps, data_variables = _crop_incomplete_data(sweeps_10, original_data_variables)
    verify_sweep_and_data_variables(
        original_sweeps=sweeps_10,
        sweeps=sweeps,
        expected_sweep_sizes=[7],
        original_data_variables=original_data_variables,
        data_variables=data_variables,
        expected_data_variable_sizes=7,
    )
    verify_hard_sweeps(original_data_variables, data_variables, [2, 2])
