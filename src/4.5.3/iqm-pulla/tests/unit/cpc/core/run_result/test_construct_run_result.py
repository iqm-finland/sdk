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

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest
import xarray as xr

from exa.common.data.parameter import Sweep
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import convert_sweeps_to_list_of_tuples
from iqm.cpc.core.run_result import RunResult, construct_run_result
from iqm.station_control.interface.models import JobExecutorStatus, RunData, SweepData

DUMMY_TIMESTAMP = datetime.now(timezone.utc)
QUBIT_3 = "QB3"
QUBIT_5 = "QB5"
DUT_LABEL = "M138_W36_A22_N05"


def mock_sweep_data(
    sweep_id: UUID = uuid4(),
    settings: SettingNode | None = SettingNode("root"),
) -> SweepData:
    """A dummy sweep data object."""
    return SweepData(
        sweep_id=sweep_id,
        dut_label=DUT_LABEL,
        settings=settings,  # type: ignore[arg-type]
        sweeps=[],
        return_parameters=[],
        created_timestamp=DUMMY_TIMESTAMP,
        modified_timestamp=DUMMY_TIMESTAMP,
        begin_timestamp=DUMMY_TIMESTAMP,
        end_timestamp=DUMMY_TIMESTAMP,
        job_status=JobExecutorStatus.READY,
    )


def mock_run_data(  # noqa: PLR0913
    run_id: UUID = uuid4(),
    sweep_id: UUID = uuid4(),
    experiment_name: str | None = "experiment_name",
    experiment_label: str | None = "experiment_label",
    begin_timestamp: datetime = DUMMY_TIMESTAMP - timedelta(minutes=5),
    end_timestamp: datetime = DUMMY_TIMESTAMP,
    options: dict | None = None,
    additional_run_properties: dict | None = None,
    components: list[str] | None = None,
    default_data_parameters: list[str] | None = None,
    default_sweep_parameters: list[str] | None = None,
    settings: SettingNode | None = None,
) -> RunData:
    """A dummy RunData object."""
    return RunData(
        run_id=run_id,
        username="username",
        experiment_name=experiment_name or "",
        experiment_label=experiment_label or "",
        options=options or {},
        additional_run_properties=additional_run_properties or {},
        software_version_set_id=1,
        components=components or [],
        default_data_parameters=default_data_parameters or [],
        default_sweep_parameters=default_sweep_parameters or [],
        sweep_data=mock_sweep_data(
            sweep_id=sweep_id,
            settings=settings or SettingNode("root"),
        ),
        created_timestamp=begin_timestamp,
        modified_timestamp=begin_timestamp,
        begin_timestamp=begin_timestamp,
        end_timestamp=end_timestamp,
    )


@pytest.mark.parametrize("post_processes", (None, []))
def test_returns_same_run_data_and_dataset_when_no_post_processes(post_processes):
    run_id = uuid4()
    sweep_id = uuid4()

    run_data = mock_run_data(
        run_id=run_id,
        sweep_id=sweep_id,
        experiment_name="experiment_name",
        begin_timestamp=DUMMY_TIMESTAMP - timedelta(minutes=5),
        components=[QUBIT_3, QUBIT_5],
        default_data_parameters=["data_parameter"],
        default_sweep_parameters=["sweep_parameter"],
    )
    sweep_results = {}
    run_result = construct_run_result(run_data, sweep_results, post_processes)
    expected_dataset = xr.Dataset()

    assert run_result.dataset.identical(expected_dataset)
    assert run_result.run_data.run_id == run_id
    assert run_result.run_data.components == run_data.components
    assert run_result.run_data.default_data_parameters == run_data.default_data_parameters
    assert run_result.run_data.default_sweep_parameters == run_data.default_sweep_parameters
    assert run_result.run_data.additional_run_properties == run_data.additional_run_properties
    assert run_result.run_data.username == run_data.username
    assert run_result.run_data.experiment_name == run_data.experiment_name
    assert run_result.run_data.experiment_label == run_data.experiment_label
    assert run_result.run_data.options == run_data.options


def test_returns_dataset_that_is_changed_due_to_post_processes(complete_run_data, complete_sweep_results):
    expected_dataset = construct_run_result(complete_run_data, complete_sweep_results).dataset
    expected_dataset = expected_dataset.assign_coords({"baz": ("baz", [2])})

    post_process_1 = lambda _run_result: replace(  # noqa: E731
        _run_result, dataset=_run_result.dataset.assign_coords({"baz": ("baz", [2])})
    )
    complete_run_data.default_data_parameters = ["updated_data_parameter"]
    post_process_2 = lambda _run_result: replace(_run_result, run_data=complete_run_data)  # noqa: E731
    run_result = construct_run_result(complete_run_data, complete_sweep_results, [post_process_1, post_process_2])

    assert run_result.run_data.run_id == complete_run_data.run_id
    assert run_result.run_data.components == complete_run_data.components
    assert run_result.run_data.default_data_parameters == complete_run_data.default_data_parameters
    assert run_result.run_data.default_sweep_parameters == complete_run_data.default_sweep_parameters
    assert run_result.run_data.additional_run_properties == complete_run_data.additional_run_properties
    assert run_result.run_data.username == complete_run_data.username
    assert run_result.run_data.experiment_name == complete_run_data.experiment_name
    assert run_result.run_data.experiment_label == complete_run_data.experiment_label
    assert run_result.run_data.options == complete_run_data.options

    assert run_result.dataset.identical(expected_dataset)


def test_array_of_one_item_is_turned_to_scalar(complete_run_data):
    sweep_results = {
        "QB1__readout.result": [
            np.array([1]),
        ]
    }
    run_result = construct_run_result(complete_run_data, sweep_results)

    assert isinstance(run_result.dataset["QB1__readout.result"], xr.DataArray)
    assert run_result.dataset["QB1__readout.result"][0] == 1


@pytest.fixture
def verify_run_result(sweeps, return_parameter, hard_sweeps, request_metadata, settings):
    def _verify_run_result(run_result: RunResult, updated_metadata: dict[str, Any] | None = None):
        _request_metadata = updated_metadata or request_metadata
        assert run_result.run_data.components == _request_metadata["components"]
        assert run_result.run_data.default_data_parameters == _request_metadata["default_data_parameters"]
        assert run_result.run_data.default_sweep_parameters == _request_metadata["default_sweep_parameters"]
        assert run_result.run_data.sweep_data.settings == settings

        assert run_result.run_data.sweep_data.return_parameters == [return_parameter.name]
        assert run_result.run_data.sweep_data.sweeps == convert_sweeps_to_list_of_tuples(sweeps)

    return _verify_run_result


def test_complete_run_result_has_expected_attributes(complete_run_data, complete_sweep_results, verify_run_result):
    run_result = construct_run_result(run_data=complete_run_data, sweep_results=complete_sweep_results)
    verify_run_result(run_result=run_result)


def verify_coords(expected_sweep_dimensions: list[tuple[Sweep, int]], coords: xr.core.coordinates.DatasetCoordinates):
    expected_coord_keys = {sw_tuple[0].parameter.name for sw_tuple in expected_sweep_dimensions}
    assert set(coords.keys()) == expected_coord_keys

    for sw_tuple in expected_sweep_dimensions:
        expected_sweep = sw_tuple[0]
        expected_size = sw_tuple[1]
        assert len(coords[expected_sweep.parameter.name]) == expected_size
        for idx, coord in enumerate(coords[expected_sweep.parameter.name]):
            assert coord.standard_name == expected_sweep.parameter.name == coord.parameter.name
            assert expected_sweep.parameter.label in coord.long_name
            assert coord.units == expected_sweep.parameter.unit == coord.parameter.unit
            assert coord.data.item() == expected_sweep.data[idx]


def test_dataset_from_complete_run_result_has_expected_coordinates(
    sweeps, return_parameter, complete_run_data, complete_sweep_results
):
    run_result = construct_run_result(run_data=complete_run_data, sweep_results=complete_sweep_results)
    dataset = run_result.dataset
    readout_count = len(dataset[return_parameter.name])

    verify_coords([(sweeps[0][0], readout_count)], dataset.coords)


def test_dataset_from_complete_run_result_has_expected_data_variables(
    return_parameter, complete_run_data, complete_sweep_results
):
    run_result = construct_run_result(run_data=complete_run_data, sweep_results=complete_sweep_results)
    dataset = run_result.dataset

    assert np.all(
        [d.data.item() for d in dataset[return_parameter.name]] == complete_sweep_results[return_parameter.name]
    )


def test_complete_run_result_has_expected_settings(settings, complete_run_data, complete_sweep_results):
    run_result = construct_run_result(run_data=complete_run_data, sweep_results=complete_sweep_results)
    assert run_result.run_data.sweep_data.settings == settings


def test_complete_run_result_has_expected_sweeps_reconstructed_from_control_instructions(
    sweeps, complete_run_data, complete_sweep_results
):
    run_result = construct_run_result(run_data=complete_run_data, sweep_results=complete_sweep_results)
    expected_sweeps = convert_sweeps_to_list_of_tuples(sweeps)
    assert run_result.run_data.sweep_data.sweeps == expected_sweeps


def test_interrupted_run_result_has_expected_attributes(
    interrupted_run_data, interrupted_sweep_results, verify_run_result
):
    run_result = construct_run_result(run_data=interrupted_run_data, sweep_results=interrupted_sweep_results)
    verify_run_result(run_result=run_result)


def test_dataset_from_interrupted_run_result_has_expected_coordinates(
    sweeps, return_parameter, interrupted_run_data, interrupted_sweep_results
):
    run_result = construct_run_result(run_data=interrupted_run_data, sweep_results=interrupted_sweep_results)
    dataset = run_result.dataset
    readout_count = len(dataset[return_parameter.name])

    # Coordinates of datasets of incomplete sweeps are
    # cropped to fit the number of successful of readouts.
    # See full test suite for BackendRequestDatasetLoader.crop_incomplete_data.
    verify_coords([(sweeps[0][0], readout_count)], dataset.coords)


def test_dataset_from_interrupted_run_result_has_expected_data_variables(
    return_parameter, interrupted_run_data, interrupted_sweep_results, complete_sweep_results
):
    run_result = construct_run_result(run_data=interrupted_run_data, sweep_results=interrupted_sweep_results)
    dataset = run_result.dataset

    # Data points of datasets of incomplete sweeps are
    # cropped to fit the number of successful readouts.
    # See full test suite for BackendRequestDatasetLoader.crop_incomplete_data.
    assert [d.data.item() for d in dataset[return_parameter.name]] == complete_sweep_results[return_parameter.name]
