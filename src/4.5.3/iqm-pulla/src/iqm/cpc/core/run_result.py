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
"""Data structures to represent run results.

Contains also logic for reconstructing run results from persisted sweep results (raw data) and run data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from exa.common.data.parameter import Parameter, Sweep
from exa.common.sweep.util import NdSweep, ParallelSweep, linear_index_sweep
from iqm.station_control.interface.models import RunData, SweepResults

DataVariables = list[tuple[Parameter, list[np.ndarray], NdSweep]]


@dataclass
class RunResult:
    """The definitive data structure to represent the execution results of an Experiment.

    The run result contains the full dataset and run data.
    """

    run_data: RunData
    """Information that defines the Experiment run, after it was executed."""

    dataset: xr.Dataset
    """Xarray dataset that has the raw and post-processed data of an execution."""

    def __repr__(self) -> str:
        return f"RunResult of {self.run_data!r} ({self.run_data.begin_timestamp})"


def construct_run_result(
    run_data: RunData,
    sweep_results: SweepResults,
    post_processes: list[Callable[[RunResult], RunResult]] | None = None,
) -> RunResult:
    """Construct the complete experiment results of a run.

    Loads data that was saved by Station Control, using reconstructing functionality for run result.
    Runs post-processing functionality against the dataset of the constructed run result and packages
     it to a single structure along with run data.

    Args:
        run_data: Data of the run for which a run result should be constructed.
        sweep_results: Sweep results of the run for which a run result should be constructed.
        post_processes: List of functions to apply for dataset of the reconstructed run result.
            Default to returning reconstructed run result without changes.

    Returns:
        Run result which contains the post-processed run data and the post-processed dataset.

    """
    run_result = _construct_run_result(run_data, sweep_results)

    if post_processes:
        for post_process in post_processes:
            run_result = post_process(run_result)

    return run_result


def _construct_run_result(run_data: RunData, sweep_results: SweepResults) -> RunResult:
    """Construct a RunResult from raw data entries of RunData and SweepResults.

    Returns:
        Run result which contains the run data and the dataset.

    """

    def _resolve_from_settings(name: str) -> Parameter:
        setting = run_data.sweep_data.settings.find_by_name(name)
        if setting is None:
            return Parameter(name, label=name)
        return setting.parameter

    # Turn into scalar if it was originally scalar
    for return_parameter, sweep_result in sweep_results.items():
        for index, data in enumerate(sweep_result):
            if data.shape == np.atleast_1d(1).shape:
                sweep_results[return_parameter][index] = data[0]

    incomplete_data_variables = [
        (
            _resolve_from_settings(return_parameter),
            sweep_results[return_parameter],
            run_data.hard_sweeps[return_parameter],
        )
        for return_parameter in run_data.sweep_data.return_parameters
    ]
    soft_sweeps, data_variables = _crop_incomplete_data(run_data.sweep_data.sweeps, incomplete_data_variables)

    dataset = _convert_sweep_results_to_dataset(data_variables, soft_sweeps)

    return RunResult(run_data=run_data, dataset=dataset)


def _crop_incomplete_data(
    original_sweeps: NdSweep, incomplete_data_variables: DataVariables
) -> tuple[NdSweep, DataVariables]:
    """Crop data.

    Crops an original sweep and an incomplete data variables
    into a consistent shape that can be used to construct some
    consistent dataset, salvaging whatever data variables are available.

    A sweep is either `complete` or `incomplete` (ongoing or interrupted).
    In case the sweep is incomplete, the amount of data in data variables
    does not fill up the shape prescribed by the sweep definitions.

    Padding missing data points with NaNs is not an option because there
    will be always some analysis and plotting functions that will either fail
    or will behave unexpectedly in the presence of NaNs.

    The alternative is to reshape the original sweep and
    existing data variables to make it a whole, i.e. cropping.

    Cropping includes only the complete sweeps of the highest dimension.
    For example: A sweep was supposed to be 10x3x20x5, but was interrupted
    after N=100 iterations. The resulting data will then be 10x3x3x1
    (because next full data cube would be at 10x3x4x1=120 datapoints).
    The returned tuple will be sweep definitions for 10x3x3x1
    and N=90 iteration's (=soft sweep's) results.

    .. image:: ../_static/images/cropping-incomplete-sweeps.png

    Args:
        original_sweeps: The original sweeps of an incomplete sweep definition.
        incomplete_data_variables: The data variables of an incomplete sweep definition.

    Returns:
        A tuple of cropped sweeps and data variables such
        that their shapes are consistent with each other and as much
        data is preserved as possible.

    """
    cropped_data_variables = []
    for parameter, data, hard_sweeps in incomplete_data_variables:
        # number of successful iterations
        # e.g. 100 for our example
        successful_iterations = sum(len(spot_data) if hasattr(spot_data, "__len__") else 1 for spot_data in data)
        hard_sweeps_list = hard_sweeps if isinstance(hard_sweeps, list) else []
        all_sweeps = hard_sweeps_list + original_sweeps

        # e.g. [10, 3, 20, 5] - for our 10x3x20x5 sweep
        sweep_dim_sizes = [len(tuples_of_sweeps[0].data) for tuples_of_sweeps in all_sweeps]

        # the number of (successful) iterations to make a sweep dimension complete
        # e.g. [10, 30, 600, 3000] - for our 10x3x20x5 sweep
        iterations_per_complete_dim = np.cumprod(sweep_dim_sizes)

        if not np.any(iterations_per_complete_dim) or successful_iterations == iterations_per_complete_dim[-1]:
            # we actually have all the raw results mandated by the sweep
            return original_sweeps, incomplete_data_variables

        # which (first) sweep dimension is incomplete?
        # e.g. the 3rd (at index 2) sweep dimension is incomplete, as 600 > 100
        # for our 10x3x20x5 sweep
        #              ^
        #              |
        #              +--- incomplete
        incomplete_dim = np.argmax(iterations_per_complete_dim > successful_iterations)

        # calculate how much of that sweep was completed, and how many points cannot be used
        # e.g. for the 3rd, incomplete sweep dimension, we have 100 // 30 = 3 rounds
        # and 100 % 30 = 10 raw results are left over
        last_full_dim_iters = iterations_per_complete_dim[incomplete_dim - 1]
        complete, leftover = divmod(successful_iterations, last_full_dim_iters)

        # crop raw results
        if incomplete_dim > 0:
            # We have enough raw results at least for a complete round of
            # the very first sweep dimension.
            # Discard the leftover raw results.

            # If the first incomplete dim is hard, there will be just one spot and we need to crop the spot data
            crop_soft_index = None
            crop_hard_index = last_full_dim_iters * complete
            if incomplete_dim >= len(hard_sweeps_list):
                # if the first incomplete dim is soft we don't crop the spot data but remove the leftover spots
                hard_sweep_iters = (
                    iterations_per_complete_dim[len(hard_sweeps_list) - 1] if len(hard_sweeps_list) > 0 else 1
                )
                crop_soft_index = int(last_full_dim_iters * complete / hard_sweep_iters)
                crop_hard_index = None
            cropped_data_variable = (
                parameter,
                [
                    spot_result[:crop_hard_index] if len(hard_sweeps_list) > 0 else spot_result
                    for spot_result in data[:crop_soft_index]
                ],
                [_crop_sweep(complete, index, sweep, incomplete_dim) for index, sweep in enumerate(hard_sweeps_list)]
                if len(hard_sweeps_list) > 0
                else [],
            )
        else:
            # Not even the first sweep dimension has a complete round.
            # Since the leftovers are all we have, do not discard anything.
            complete = leftover
            cropped_data_variable = (parameter, data, hard_sweeps)

        cropped_data_variables.append(cropped_data_variable)

    # crop the entire sweep(s) definition
    cropped_sweeps = [
        _crop_sweep(complete, index + len(hard_sweeps_list), sweep_tuple, incomplete_dim)
        for index, sweep_tuple in enumerate(original_sweeps)
    ]

    return cropped_sweeps, cropped_data_variables


def _crop_sweep(complete: int, index: int, sweep_tuple: ParallelSweep, incomplete_dim) -> ParallelSweep:  # noqa: ANN001
    if index < incomplete_dim:
        # this sweep dimension is okay needs no cropping
        return sweep_tuple

    # Crop the very first incomplete sweep dimension to the
    # number of complete rounds we know we have.
    # Crop subsequence (also incomplete) sweep dimensions to
    # a single data point.
    cropped_sweep_datas = [
        sweep.data[:complete] if index == incomplete_dim else sweep.data[:1] for sweep in sweep_tuple
    ]
    return tuple(
        Sweep(parameter=sweep.parameter, data=cropped_sweep_data)
        for sweep, cropped_sweep_data in zip(sweep_tuple, cropped_sweep_datas)
    )


def _convert_sweep_results_to_dataset(
    data_variables: DataVariables,
    soft_sweeps: NdSweep,
) -> xr.Dataset:
    """Produce an xarray Dataset from raw results, using recorded sweep specifications for reshaping.

    Adds each variable in `data_variables` as an xarray DataArray, whose coordinates are a combination of
    `soft_sweeps` and the hardware sweeps specific to each data variable.

    Args:
        data_variables: Data to format.
        soft_sweeps: List of parallel sweeps that represent the coordinates common to all data variables.
            This follows the convention that the first sweep corresponds to the fastest-changing index.
            Sweep major dimension is the first element of parallel sweep in the list of sweeps.
            Sweep minor dimensions are elements [1:] in a parallel sweep,
            and are added as additional coordinates, indexed by the major dimension.

    """
    final_data_variables = []
    for parameter, data, hard_sweep in data_variables:
        # if hard sweep has not been specified, we create a dummy index based on the data.
        if hard_sweep is None:
            if len(data) > 0:
                hard_sweep = linear_index_sweep(parameter, np.atleast_1d(data[0]).size)  # noqa: PLW2901
            else:  # data is empty, reshaping won't do anything so this value won't matter.
                hard_sweep = []  # noqa: PLW2901
        final_data_variables.append((parameter, data, hard_sweep))

    # Create initial dataset with all the coordinates, but no data yet.
    dataset = xr.Dataset()
    # Gather known coordinates of each sweep.
    soft_and_hard_sweeps = soft_sweeps[:]
    for data_variables_ in final_data_variables:
        soft_and_hard_sweeps.extend(data_variables_[2])

    coordinate_assignments: dict[str, Any] = {}
    for parallel_sweep in soft_and_hard_sweeps:
        # Add first sweep of a parallel sweep as a major dimension for the result data.
        sweep = parallel_sweep[0]
        coordinate_assignments[sweep.parameter.name] = sweep.data

        # Add tail of a parallel sweep as additional coordinates, and index them using the major dimension.
        major_dimension_name = parallel_sweep[0].parameter.name
        for sweep in parallel_sweep[1:]:
            coordinate_assignments[sweep.parameter.name] = (major_dimension_name, sweep.data)

    dataset = dataset.assign_coords(coordinate_assignments)
    for parallel_sweep in soft_and_hard_sweeps:
        for sweep in parallel_sweep:
            _annotate(dataset.variables[sweep.parameter.name], sweep.parameter)

    return dataset.assign(
        {
            parameter.name: _reshape_raw_data_into_data_array(parameter, data, hard_sweeps + soft_sweeps)
            for parameter, data, hard_sweeps in final_data_variables
        }
    )


def _reshape_raw_data_into_data_array(
    parameter: Parameter,
    data: list[np.ndarray],
    sweeps: NdSweep,
) -> xr.DataArray:
    """Build an xarray DataArray out of given `parameter` and `data`.
    The coordinates are given by `sweeps`. The data is reshaped according to the sizes of the first
    elements of the tuples (i.e. sizes of the major dimensions of the parallel sweeps).
    For example, ``len(soft_sweeps[0][0].data) = 3; len(soft_sweeps[1][0].data) = 2`` will shape the data
    into a 3-by-2 array.
    This follows the convention that the first sweep corresponds to the fastest-changing index.

    The total number of elements in `data` must equal to the product of the lengths of the Sweep data in `sweeps`.

    Args:
        parameter: Parameter that this data represents.
        data: Numerical data that can be converted into a numpy array.
           Each item in the sequence represents raw results obtained for a given parameter spot.
           A raw result can be a scalar, or in case of a hard sweep, a sequence.
        sweeps: list of tuples of Sweeps that represent the n-dimensional combination of hard and soft sweeps this
           data is gathered from.
           Only the length of the first item in each tuple will be used.

    Raises:
         ValueError: if `data` cannot be reshaped according to `sweeps`.

    """
    dimensions = [sweep_tuple[0].parameter.name for sweep_tuple in sweeps]
    dimension_sizes = [len(sweep_tuple[0].data) for sweep_tuple in sweeps]

    try:
        # By convention, the sweep dimensions are ordered such that first dim corresponds to the
        # fastest changing index. Numpy reshape works the other way, so we shape in reverse order,
        # then transpose the array back.
        reshaped = np.asarray(data).reshape(tuple(reversed(dimension_sizes))).T
    except ValueError as err:
        msg = (
            f"Cannot reshape {parameter.name} (shape {np.asarray(data).shape}) into shape {dimension_sizes}. "
            f"The sweeps are "
        )
        sweep_msg = ", ".join(f"{name}, ({size})" for name, size in zip(dimensions, dimension_sizes))
        raise ValueError(msg + sweep_msg) from err
    attrs = {
        "parameter": parameter,
        "standard_name": parameter.name,
        "long_name": parameter.label,
        "units": parameter.unit,
    }
    return xr.DataArray(name=parameter.name, data=reshaped, attrs=attrs, dims=dimensions)


def _annotate(array: xr.Variable, parameter: Parameter) -> None:
    parts = parameter.name.split(".")
    if parts[0] == "gates" and len(parts) > 3:
        # Add the gate with locus to the parameter label.
        locus = parts[3].replace("__", ", ")
        annotation = parameter.model_copy(update={"label": f"{parts[1].upper()}({locus}) {parameter.label.lower()}"})
    elif len(parts) > 1:
        # Add the controller name, prettified, to the parameter label.
        # e.g. "drive_3.frequency" becomes "Drive 3 Frequency".
        parent_name = parts[0].replace("_", " ")
        parent_name = parent_name[0].capitalize() + parent_name[1:]
        annotation = parameter.model_copy(update={"label": f"{parent_name} {parameter.label}"})
    else:
        annotation = parameter

    array.attrs.update(
        {
            "parameter": annotation,
            "units": annotation.unit,
            "standard_name": annotation.name,
            "long_name": annotation.label,
        }
    )
