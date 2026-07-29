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

from exa.common.control.sweep.option import StartStopOptions
from exa.common.data.parameter import CollectionType, DataType, Parameter, Sweep
from iqm.cpc.core.run_result import _convert_sweep_results_to_dataset

variable1 = Parameter("QB1__readout.frequency", "Frequency", "Hz", DataType.FLOAT)
variable2 = Parameter("flux_1.voltage", "Voltage", "V", DataType.FLOAT)
result_parameter = Parameter("QB1__readout.result", "Result", "V", DataType.COMPLEX, CollectionType.NDARRAY)
result_parameter2 = Parameter("QB1__readout.result_2", "Result", "V", DataType.COMPLEX, CollectionType.NDARRAY)


def test_dataset_has_data_parameters_as_data_arrays():
    sweeps = [(Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),)]
    data_variables = [
        (result_parameter, np.ones(10).tolist(), []),
        (result_parameter2, np.ones(10).tolist(), []),
    ]
    dataset = _convert_sweep_results_to_dataset(data_variables, sweeps)

    assert result_parameter.name in dataset
    assert result_parameter2.name in dataset


def test_dataset_has_sweeps_as_coordinates():
    sweeps = [
        (Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),),
        (Sweep(parameter=variable2, data=StartStopOptions(-1, 1, 3).data),),
        (Sweep(parameter=Parameter("var3"), data=StartStopOptions(-2, 2, 100).data),),
    ]
    data_variables = [
        (result_parameter, np.ones((100, 3, 10)).tolist(), []),
        (result_parameter2, np.ones((100, 3, 10)).tolist(), []),
    ]
    dataset = _convert_sweep_results_to_dataset(data_variables, sweeps)

    for tuple_of_sweeps in sweeps:
        assert tuple_of_sweeps[0].parameter.name in dataset.coords


def test_dataset_has_multi_parameter_dimensions_as_data_vars():
    minor_variable_name = "var4"
    sweeps = [
        (Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),),
        (
            Sweep(parameter=variable2, data=StartStopOptions(-1, 1, 3).data),
            Sweep(parameter=Parameter("var4"), data=StartStopOptions(-1, 1, 3).data),
        ),
        (Sweep(parameter=Parameter("var3"), data=StartStopOptions(-2, 2, 100).data),),
    ]
    data_variables = [
        (result_parameter, np.ones((100, 3, 10)).tolist(), []),
        (result_parameter2, np.ones((100, 3, 10)).tolist(), []),
    ]
    dataset = _convert_sweep_results_to_dataset(data_variables, sweeps)
    assert minor_variable_name in dataset


def test_uses_hardware_sweep_name_as_coordinate_if_given_as_datalength():
    sweeps = [
        (Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),),
        (Sweep(parameter=variable2, data=StartStopOptions(-1, 1, 3).data),),
    ]
    hardware_sweep = [(Sweep(parameter=Parameter("hard2"), data=StartStopOptions(0, 40, 40).data),)]
    data_variables = [
        (result_parameter2, np.ones((40, 3, 10)).tolist(), hardware_sweep),
    ]
    dataset = _convert_sweep_results_to_dataset(data_variables, sweeps)

    assert "hard2" in dataset[result_parameter2.name].coords
    assert "hard2" in dataset.coords

    for tuple_of_sweeps in sweeps:
        assert tuple_of_sweeps[0].parameter.name in dataset.coords


def test_dummy_index_is_generated_if_hard_sweep_is_none():
    sweeps = [(Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),)]
    hardware_sweep = None
    data_variables = [
        (result_parameter2, [np.ones((7,))] * 10, hardware_sweep),
    ]
    dataset = _convert_sweep_results_to_dataset(data_variables, sweeps)

    assert "QB1__readout.result_2_index" in dataset[result_parameter2.name].coords
    assert (dataset.coords["QB1__readout.result_2_index"] == list(range(7))).all()


def test_no_data():
    sweeps = [
        (Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),),
        (Sweep(parameter=variable2, data=StartStopOptions(-1, 1, 3).data),),
    ]
    dataset = _convert_sweep_results_to_dataset([], sweeps)

    for tuple_of_sweeps in sweeps:
        assert tuple_of_sweeps[0].parameter.name in dataset.coords


def test_no_sweeps():
    data_variables = [(result_parameter, [1], [])]
    dataset = _convert_sweep_results_to_dataset(data_variables, [])

    for array in data_variables:
        assert array[0].name in dataset


def test_stores_metadata():
    sweeps = [
        (Sweep(parameter=variable1, data=StartStopOptions(0, 10, 10).data),),
        (Sweep(parameter=variable2, data=StartStopOptions(-1, 1, 3).data),),
    ]
    data_variables = [(result_parameter, np.ones((3, 10)).tolist(), [])]

    dataset = _convert_sweep_results_to_dataset(data_variables, sweeps)
    assert dataset[variable1.name].attrs["parameter"] == variable1.model_copy(
        update={"label": "QB1  readout Frequency"}
    )
    assert dataset[variable2.name].attrs["parameter"] == variable2.model_copy(update={"label": "Flux 1 Voltage"})
    assert dataset[result_parameter.name].attrs["parameter"] == result_parameter
