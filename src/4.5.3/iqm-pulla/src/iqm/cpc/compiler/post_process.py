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
"""Post-processing utility functions for MultiQubitExperiment."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
import logging
import re
from typing import Any

import numpy as np
from pandas import MultiIndex
import xarray as xr

from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import CollectionType, DataType, Parameter
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import NdSweep
from iqm.cpc.compiler._utils.post_process import (
    AverageResponse,
    _get_gate_and_impl_name,
    _map_acquisitions,
    compute_excitation_probability,
    compute_excitation_probability_threshold,
    compute_single_qubit_probability_from_counter_results,
    convert_single_shot_to_counter,
    principal_component_analysis,
)
from iqm.cpc.compiler.compiler import CompilationStage
from iqm.cpc.core.config import ComponentGroupingMode
from iqm.cpc.core.dataset import (
    FitResults,
    annotate,
    apply_along_coordinate,
)
from iqm.cpc.core.observation.observation_parameters import SINGLE_SHOT_READOUT_01_ERROR, SINGLE_SHOT_READOUT_10_ERROR
from iqm.cpc.core.run_result import RunResult, _construct_run_result
from iqm.cpc.interface.circuit_execution import ReadoutMappingBatch
from iqm.pulla.interface import HERALDING_KEY
from iqm.pulla.utils import convert_sweep_spot_to_arrays, convert_sweep_spot_to_arrays_with_heralding_mode_zero
from iqm.pulse.gates.measure import DEFAULT_INTEGRATION_KEY, DEFAULT_TIME_TRACE_KEY
from iqm.pulse.playlist.instructions import ReadoutMetrics
from iqm.station_control.interface.models import CircuitMeasurementResultsBatch, RunData, SweepResults

logger = logging.getLogger(__name__)


RAGGED_DATA_INDEX = Parameter("ragged_data_index", "Ragged data index")

TRIGGER_INDEX = Parameter("trigger_index", "Absolute index of the readout trigger")


@dataclass
class PullaData:
    """Raw Pulla return data."""

    sweep_results: SweepResults
    """Sweep results from the Station Control."""
    run_data: RunData | None
    """Run data (i.e. metadata)."""


def _unmap_label(label: str, mapped_readout_keys: dict[str, str]) -> str:
    """Unmap a readout label."""
    if "__" not in label:
        return label
    qubit, mapped_key = re.split("__[__]*", label)
    if mapped_key not in mapped_readout_keys:
        return label
    separator = label.count("__") * "__"
    key = mapped_readout_keys[mapped_key]
    return f"{qubit}{separator}{key}"


def _unmap_additional_run_properties(run_data: RunData, mapped_readout_keys: dict[str, str]) -> None:
    """Unmap the readout keys in run_data.additional_run_properties."""
    additional_run_properties = run_data.additional_run_properties
    applicable_fields = [
        "target_data_parameters",
        "integration_data_parameters",
        "ragged_data_labels",
        "readout_label_to_impl",
        "readout_group_names",
    ]
    for field in applicable_fields:
        if field in additional_run_properties:
            if isinstance(additional_run_properties[field], dict):  # type:ignore[index]
                unmapped_contents = {
                    _unmap_label(label, mapped_readout_keys): value
                    for label, value in additional_run_properties[field].items()  # type:ignore[index]
                }
            else:
                unmapped_contents = [
                    _unmap_label(label, mapped_readout_keys) for label in additional_run_properties[field]
                ]  # type:ignore[assignment,index]
            additional_run_properties[field] = unmapped_contents  # type:ignore[index]


def _unmap_sweeps(sweeps: NdSweep, mapped_readout_keys: dict[str, str]) -> NdSweep:
    """Unmap sweep parameters"""
    unmapped_sweeps = []
    for sweep in sweeps:
        unmapped_sweep = []
        for parallel_sweep in sweep:
            if "_ragged_data_index" in parallel_sweep.parameter.name:
                unmapped_label = _unmap_label(
                    parallel_sweep.parameter.name.replace("_ragged_data_index", ""), mapped_readout_keys
                )
                unmapped_name = f"{unmapped_label}_ragged_data_index"
                unmapped_sweep.append(
                    Sweep(
                        parameter=parallel_sweep.parameter.model_copy(
                            update={"name": unmapped_name, "label": unmapped_name}
                        ),
                        data=parallel_sweep.data,
                    )
                )
            else:
                unmapped_sweep.append(parallel_sweep)
        unmapped_sweeps.append(tuple(unmapped_sweep))
    return unmapped_sweeps


def _unmap_readout_keys(
    run_data: RunData, sweep_results: SweepResults, mapped_readout_keys: dict[str, str]
) -> tuple[RunData, SweepResults]:
    """Unmap the hashed readout keys with the mapping provided in the context."""
    run_data = replace(run_data)
    sweep_results = sweep_results.copy()
    sweep_data = run_data.sweep_data
    _unmap_additional_run_properties(run_data, mapped_readout_keys)
    run_data.hard_sweeps = {
        _unmap_label(p, mapped_readout_keys): _unmap_sweeps(sweeps, mapped_readout_keys)
        for p, sweeps in run_data.hard_sweeps.items()
    }
    run_data.default_data_parameters = [_unmap_label(p, mapped_readout_keys) for p in run_data.default_data_parameters]
    sweep_data.return_parameters = [_unmap_label(p, mapped_readout_keys) for p in sweep_data.return_parameters]
    sweep_results = {_unmap_label(p, mapped_readout_keys): data for p, data in sweep_results.items()}
    return run_data, sweep_results


def construct_run_result(pulla_data: PullaData, context: dict[str, Any]) -> RunResult:
    """Construct the dataset and attach it to a RunResult object.

    Unmaps the hashed readout keys.

    Args:
        pulla_data: The Pulla data (run data and sweep data).
        context: The Compiler context.

    Returns:
        The RunResult.

    """
    run_data = pulla_data.run_data
    sweep_results = pulla_data.sweep_results
    if mapped_readout_keys := context.get("mapped_readout_keys", {}):
        run_data, sweep_results = _unmap_readout_keys(run_data, sweep_results, mapped_readout_keys)  # type:ignore[arg-type]
    return _construct_run_result(run_data, sweep_results)  # type:ignore[arg-type]


def process_metadata(run: RunResult) -> RunResult:
    """Process the metadata in the RunResult.

    - Set the target data variables
    - Process tuples serialised into lists back into tuples.

    Args:
        run: The RunResult.

    Returns:
        The RunResult with the metadata processed.

    """
    # set target data variable
    additional_run_properties = run.run_data.additional_run_properties
    if additional_run_properties["target_data_parameters"]:  # type:ignore[index]
        run.default_data_parameters = additional_run_properties["target_data_parameters"]  # type:ignore[attr-defined,index]
    else:
        excited_state_parameters = [
            f"{c}_excited_state_probability"
            for c in additional_run_properties["readout_components"]  # type: ignore[index]
        ]
        run.run_data.default_data_parameters = excited_state_parameters
    # fix tuple serialization for grouped and colour grouped run components
    run_components = additional_run_properties["components"]  # type: ignore[index]
    grouping_mode = additional_run_properties["components_grouping_mode"]  # type: ignore[index]
    if grouping_mode in [ComponentGroupingMode.GROUP.value, ComponentGroupingMode.COLOUR_GROUP.value]:
        if grouping_mode == ComponentGroupingMode.GROUP.value:
            additional_run_properties["components"] = [tuple(group) for group in run_components]  # type: ignore[index]
        else:
            colour_groups = []
            for colour in run_components:
                colour_groups.append([tuple(group) for group in colour])
            additional_run_properties["components"] = colour_groups  # type: ignore[index]
    return run


def create_ragged_data_index(
    run: RunResult,
) -> RunResult:
    """Adds ``MultiIndex`` to the trigger index dimension for accessing ragged data via the hard sweep values.

    If there is a non-uniform number of readout triggers for n RO label across the segments of the playlist, the data
    cannot be reshaped into an N-dimensional box with the dims
    ``(<hard sweep dimension 0>, <hard sweep dimension 1>, ...)``. The data is then "ragged" and the only meaningful
    way to index the data is along the readout trigger index dimension. In order to be able to use this ragged data
    meaningfully, the user often wants access it via the known dimensions of the experiment run, i.e. the sweeps.
    For this reason we attach the values the hard sweeps took in each segment where a trigger happened to the
    trigger index dimension.

    An example: the playlist was generated from hard sweeps (there are no soft sweeps here):
    ``[Sweep(parameter=param1, data=[0, 1]), Sweep(parameter=param2, data=[0.1, 0.2, 0.3])]``,
    which means there are 2 * 3 = 6 segments. Let's say for a given RO label ``"QB1__key"``
    we have 1 RO trigger in the first segment, 1 trigger in the third, and 2 in the final one
    (the rest have no triggers for this label). Thus, the returned data has the length 4, i.e.
    the trigger index goes from 0 to 3. Taking into account the convention in ordering sweep dimensions,
    the values of the sweeps for these trigger indices are then:

    0 => ``(param2 = 0.1, param1 = 0)`` (in the 1st segment, both params assume their first values)
    1 => ``(param2 = 0.3, param1 = 0)`` (in the 3rd segment param2 has its 3rd value, while param1 is still in its 1st)
    2 => ``(param2 = 0.3, param1 = 1)`` (in the last segment, both params have their final values)
    3 => ``(param2 = 0.3, param1 = 1)`` (there were 2 triggers in the last segment, so this has the same values as 2.)

    The ``MultiIndex`` allows then the data via the hard sweep values:
    ``exp.last_run["<my label>_readout].sel({QB1::param1: 0})`` gives the slice where ``param1 == 0``

    Args:
        run: The run result.

    Returns:
        The RunResult with the ragged data variables now indexed.

    """
    additional_run_properties = run.run_data.additional_run_properties
    if "ragged_data_labels" not in additional_run_properties:  # type:ignore[operator]
        return run

    ragged_label_counts = additional_run_properties["ragged_data_labels"]  # type:ignore[index]
    hard_sweeps = dict(
        zip(
            additional_run_properties["ragged_hard_sweeps"],  # type: ignore[index]
            additional_run_properties["ragged_hard_sweeps_data"],  # type: ignore[index]
        )
    )
    parallel_sweep_info = additional_run_properties["ragged_parallel_sweeps"]  # type:ignore[index]

    def _generate_parallel_spot(param: str):  # noqa: ANN202
        parallel_datas = [data for p, data in reversed(hard_sweeps.items()) if p in parallel_sweep_info[param]]
        for val in zip(*parallel_datas):
            yield val

    all_parallel_dims = [d for dims in parallel_sweep_info.values() for d in dims]

    # modify the hard sweeps data so that
    # parallel sweeps are represented by their major dimension and have tuple values
    cart_sweeps: dict[str, Any] = {}
    for param, data in hard_sweeps.items():
        if param in parallel_sweep_info:
            cart_sweeps[param] = list(_generate_parallel_spot(param))
        elif param not in all_parallel_dims:
            cart_sweeps[param] = data

    # the hard sweep values in each segment as tuples with parallel sweeps represented by nested tuples
    # with the sweep spots ordered by the convention
    sweeps_datas = list(cart_sweeps.values())
    sweeps_datas.reverse()
    raw_tuples = list(product(*sweeps_datas))

    def _gen_ragged_index(run: RunResult) -> RunResult:
        coords_assignment = {}
        for label, counts in ragged_label_counts.items():
            # now unpack the nested tuples (parallel sweeps) to make flat tuples (for this acq label)
            tuples = []
            found_triggers = 0
            for tup_idx, tup in enumerate(raw_tuples):
                new_as_list = []
                for elem in tup:
                    if isinstance(elem, tuple):
                        new_as_list.extend(list(elem))
                    else:
                        new_as_list.append(elem)
                new_tup = tuple(new_as_list)
                for trig_idx_in_seg in range(counts[tup_idx]):
                    # finally, add the tuple as many times as there were triggers in this segment
                    # and advance the "trigger_index" dimension accordingly
                    tuples.append(new_tup + (found_triggers + trig_idx_in_seg,))
                found_triggers += counts[tup_idx]

            component, key = re.split("__[__]*", label)
            prefix = component if key in (DEFAULT_INTEGRATION_KEY, DEFAULT_TIME_TRACE_KEY) else label
            multi_dim_names = [f"{prefix}::{d}" for d in list(hard_sweeps.keys())[::-1]] + [
                f"{prefix}::{TRIGGER_INDEX.name}"
            ]
            ragged_multi_index = MultiIndex.from_tuples(tuples, names=multi_dim_names)
            ragged_array = run.dataset[label]
            ragged_dim_name = next(d for d in ragged_array.dims if RAGGED_DATA_INDEX.name in d)  # type: ignore[operator]
            coords_assignment[ragged_dim_name] = (ragged_dim_name, ragged_multi_index)
        return replace(run, dataset=run.dataset.assign_coords(coords_assignment))

    return _gen_ragged_index(run)


def _map_default_label(label: str) -> str:
    """Map default measurement labels such that the backwards compatibility is maintained."""
    label_split = re.split("__[__]*", label)
    if label_split[1] in (DEFAULT_INTEGRATION_KEY, DEFAULT_TIME_TRACE_KEY):
        return label_split[0]
    return label


def _is_single_shot(settings: SettingNode) -> bool:
    """Whether the circuit was run in single shot mode."""
    avg_bins = settings.controllers.options.averaging_bins.value
    if avg_bins is None:
        return False
    return avg_bins > 1


def rename_data_variables(run: RunResult, context: dict[str, Any]) -> RunResult:
    """Rename the data variables in the dataset and optionally change their data types.

    Args:
        run: The RunResult object.
        context: Compiler context.

    Returns:
        The post-processed run results with the data variables renamed.

    """
    additional_run_properties = run.run_data.additional_run_properties
    result_labels = additional_run_properties["integration_data_parameters"]  # type:ignore[index]
    settings = run.run_data.sweep_data.settings
    if run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value:
        return run  # no renaming needed when doing counter readout
    data_prefixes = [_map_default_label(label) for label in result_labels]
    acquisition_map = _map_acquisitions(
        run.run_data.sweep_data.settings, data_prefixes, additional_run_properties.get("readout_label_to_impl")
    )
    all_result_labels = result_labels + additional_run_properties["time_trace_data_parameters"]
    all_data_prefixes = data_prefixes + [
        _map_default_label(label) for label in additional_run_properties["time_trace_data_parameters"]
    ]

    def _get_suffix(result_label: str, prefix: str) -> str:
        if "time_trace" in result_label:
            return "_time_trace"
        if _is_single_shot(settings):
            return "_readout_single_shot" if acquisition_map.get(prefix) == "complex" else "_state_single_shot"
        return "_readout_single_shot" if _is_single_shot(settings) else "_readout"

    rename_dict = {
        label: Parameter(
            f"{prefix}{_get_suffix(label, prefix)}",
            f"{prefix}{_get_suffix(label, prefix).replace('_', '')}",
            data_type=DataType.COMPLEX if acquisition_map.get(prefix) == "complex" else DataType.FLOAT,
        )
        for label, prefix in zip(all_result_labels, all_data_prefixes)
    }

    context["rename_dict"] = rename_dict

    data_types_dict: dict[str, DataType] = {}

    def _rename(run_result: RunResult) -> RunResult:
        dataset = run_result.dataset
        new_names = {}
        default_data_parameters = run_result.run_data.default_data_parameters
        renamed_default_data_parameters = default_data_parameters.copy()
        for old_name, new in rename_dict.items():
            if isinstance(new, str):
                data_parameter = dataset[old_name].parameter
                renamed_parameter = data_parameter.model_copy(
                    update={
                        "name": new,
                        "data_type": data_types_dict.get(old_name, data_parameter.data_type),
                    }
                )
            else:
                renamed_parameter = new
            annotate(dataset[old_name], renamed_parameter)
            new_names[old_name] = renamed_parameter.name
            if old_name in default_data_parameters:
                renamed_default_data_parameters.remove(old_name)
                renamed_default_data_parameters.append(renamed_parameter.name)

        renamed_dataset = dataset.rename(new_names)

        run_data = run_result.run_data
        run_data.default_data_parameters = renamed_default_data_parameters
        return replace(run_result, dataset=renamed_dataset, run_data=run_data)

    return _rename(run)


def average_single_shot_data(
    run: RunResult,
) -> RunResult:
    """Average the shots (or more generally the average bins) into a new data variable in the dataset.

    If the data is already averaged, it is returned unchanged.

    Args:
        run: The RunResult object.

    Returns:
        The RunResult with the single shot data variables averaged.

    """
    settings = run.run_data.sweep_data.settings
    if _is_single_shot(settings):
        additional_run_properties = run.run_data.additional_run_properties
        result_labels = additional_run_properties["integration_data_parameters"]  # type:ignore[index]
        data_prefixes = [_map_default_label(label) for label in result_labels]
        dataset = run.dataset
        new_arrays = {}
        for prefix in data_prefixes:
            parameter = _average_single_shot_data(prefix, dataset)
            new_arrays[parameter.name] = parameter

        return replace(run, dataset=run.dataset.assign(new_arrays))
    return run


def _average_single_shot_data(prefix: str, dataset: xr.Dataset) -> xr.DataArray:
    """Create the averaged data array."""
    ss_data_name = (
        f"{prefix}_readout_single_shot" if f"{prefix}_readout_single_shot" in dataset else f"{prefix}_state_single_shot"
    )
    array = dataset[ss_data_name]
    avg_array = array.mean(dim="repetitions").rename(f"{prefix}_readout")
    avg_array.attrs = {
        "parameter": Parameter(f"{prefix}_readout", "Averaged readout data", "", DataType.FLOAT),
        "standard_name": f"{prefix}_readout",
        "long_name": f"{prefix} Averaged readout data",
        "units": "V",
    }
    return avg_array


def contrast_data(
    run: RunResult,
) -> RunResult:
    """Add contrast for several components/readouts.

    Contrast is calculated from complex integrated data with PCA. If a data variable is not complex (i.e. it is already
    discriminated) it is retained as it is.

    Args:
        run: The RunResult object.

    Returns:
        The RunResult with the contrast data variable added for all complex variables.

    """
    additional_run_properties = run.run_data.additional_run_properties
    if run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value:
        return run  # no contrast calculation needed when doing counter readout
    result_labels = additional_run_properties["integration_data_parameters"]  # type:ignore[index]
    data_prefixes = [_map_default_label(label) for label in result_labels]
    new_arrays: dict[str, xr.DataArray] = {}
    acquisition_map = _map_acquisitions(
        run.run_data.sweep_data.settings, data_prefixes, additional_run_properties.get("readout_label_to_impl")
    )
    for prefix in data_prefixes:
        if acquisition_map[prefix] == "complex":
            contrast_dict = _add_contrast(prefix, run)
            if contrast_dict is not None:
                new_arrays.update(contrast_dict)

    return replace(run, dataset=run.dataset.assign(new_arrays))


def _add_contrast(prefix: str, run_result: RunResult) -> dict[str, xr.DataArray] | None:
    """Add contrast for a complex data variable."""
    readout = run_result.dataset[f"{prefix}_readout"]
    if not readout.dims or len(readout[readout.dims[0]]) < 2:
        return None
    contrast_param = Parameter(f"{prefix}_contrast", f"{prefix} readout contrast")
    poq_param = Parameter(f"{prefix}_phase_optimal_quad", f"Phase of the first principal component for {prefix}")

    # Contrast calculation does not care about coordinate labels, so we drop them to speed up
    readout = readout.drop_vars(readout.coords.keys())

    contrast_array, phase_optimal_quad_array = apply_along_coordinate(  # type: ignore[call-overload]
        data_array=readout,
        coord=readout.dims[0],
        func=_get_pca_analyzer(poq_param.name),
        result_parameter=contrast_param,
        returns=[poq_param.name],
    )
    return {contrast_param.name: contrast_array, poq_param.name: annotate(phase_optimal_quad_array, poq_param)}


def _get_pca_analyzer(poq_param_name: str) -> Callable[[np.ndarray, np.ndarray], FitResults]:
    def pca(x, y):  # noqa: ANN001, ANN202
        return principal_component_analysis(y, poq_param_name=poq_param_name)

    return pca


def compute_excitation_probability_from_data(run: RunResult) -> RunResult:
    """Compute the excited state probability for several components. For complex readout data, this is done using the
    averaged threshold readout observations and for threshold discriminated data using the 01 and 10 assignment error
    observations. If the aforementioned observations are not available, a warning is thrown.

    Args:
        run: The RunResult object.

    Returns:
        The RunResult with the excited state probability added into the dataset.

    """
    additional_run_properties = run.run_data.additional_run_properties
    result_labels = additional_run_properties["integration_data_parameters"]  # type:ignore[index]
    if run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value:
        subscribed_components = _reconstruct_counter_subscribed_components(additional_run_properties)
        data_prefixes = _get_qubit_and_prefix(
            subscribed_components, additional_run_properties["readout_groups_without_label"]
        )
    else:
        data_prefixes = [_map_default_label(label) for label in result_labels]
    label_to_impl = additional_run_properties.get("readout_label_to_impl")
    acquisition_map = _map_acquisitions(
        run.run_data.sweep_data.settings, data_prefixes, additional_run_properties.get("readout_label_to_impl")
    )

    def _compute_excitation_probability(run_result: RunResult) -> RunResult:
        """Compute excitation probability and add it to dataset."""
        dataset = run_result.dataset
        settings = run_result.run_data.sweep_data.settings
        new_arrays = {}
        for prefix in data_prefixes:
            component = prefix.split("_")[0]  # FIXME
            measure_gate_name, implementation_name = _get_gate_and_impl_name(prefix, label_to_impl)
            if acquisition_map[prefix] == "complex":
                parameter = _compute_excitation_probability_complex(
                    prefix,
                    measure_gate_name,  # type:ignore[arg-type]
                    implementation_name,  # type:ignore[arg-type]
                    settings,
                    component,
                    dataset,  # type:ignore[arg-type]
                )
            elif acquisition_map[prefix] == "threshold":
                parameter = _compute_excitation_probability_threshold(
                    prefix,
                    measure_gate_name,  # type:ignore[arg-type]
                    implementation_name,  # type:ignore[arg-type]
                    settings,
                    component,
                    dataset,  # type:ignore[arg-type]
                )
            if parameter is not None:
                new_arrays[parameter.name] = parameter

        return replace(run_result, dataset=run_result.dataset.assign(new_arrays))

    return _compute_excitation_probability(run)


def _compute_excitation_probability_complex(
    prefix: str,
    measure_gate_name: str,
    implementation_name: str,
    settings: SettingNode,
    component: str,
    dataset: xr.Dataset,
) -> xr.DataArray | None:
    """Compute excitation probability from complex data."""
    if measure_gate_name is None:
        return None

    gate_properties = settings.get_gate_properties_for_locus(
        measure_gate_name, component, implementation=implementation_name
    )
    if (
        gate_properties.average_response_phase.value is None
        or gate_properties.average_response_g.value is None
        or gate_properties.average_response_e.value is None
    ):
        warning_template = (
            "Average response observations are not found for the data prefixe %s, "
            "or these data prefixes are associated with multiple measure gate implementations."
            " Cannot compute the excitation probability for these components."
        )
        logger.warning(warning_template, prefix)
        return None

    excited_state_parameter = Parameter(
        f"{prefix}_excited_state_probability", f"{prefix.replace('_', ' ')} excited state probability"
    )
    pulse_phase = settings.get_gate_node_for_locus(
        measure_gate_name, component, implementation=implementation_name
    ).phase.value
    probabilities = compute_excitation_probability(
        data=dataset[f"{prefix}_readout"],
        average_response=AverageResponse(
            phase=pulse_phase - gate_properties.average_response_phase.value,
            g=gate_properties.average_response_g.value,
            e=gate_properties.average_response_e.value,
        ),
        excited_state_probability=excited_state_parameter,
    )
    return probabilities


def _compute_excitation_probability_threshold(
    prefix: str,
    measure_gate_name: str,
    implementation_name: str,
    settings: SettingNode,
    component: str,
    dataset: xr.Dataset,
) -> xr.DataArray:
    """Compute excitation probability from threshold discriminated data."""
    if measure_gate_name is not None:
        gate_properties = settings.get_gate_properties_for_locus(
            measure_gate_name, component, implementation=implementation_name
        )
        single_shot_01_error = gate_properties.single_shot_01_error.value
        single_shot_10_error = gate_properties.single_shot_10_error.value
    else:
        single_shot_01_error = None
        single_shot_10_error = None
    excited_state_parameter = Parameter(f"{prefix}_excited_state_probability", f"{prefix} excited state probability")
    if single_shot_01_error is None or single_shot_10_error is None:
        probabilities = annotate(dataset[f"{prefix}_readout"], excited_state_parameter)
    else:
        probabilities = compute_excitation_probability_threshold(
            data_array=dataset[f"{prefix}_readout"],
            single_shot_01_error=gate_properties[SINGLE_SHOT_READOUT_01_ERROR.name].value,  # type: ignore[arg-type]
            single_shot_10_error=gate_properties[SINGLE_SHOT_READOUT_10_ERROR.name].value,  # type: ignore[arg-type]
            excited_state_probability_parameter=excited_state_parameter,
        )
    return probabilities


def _natural_keys(text: str) -> list[str | int]:
    """Sorts a list in natural human order.

    Implementation from http://nedbatchelder.com/blog/200712/human_sorting.html

    Args:
        text: text in the key

    """

    def _atoi(text: str) -> str | int:
        return int(text) if text.isdigit() else text

    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def _get_qubit_and_prefix(
    subscribed_components: dict[str, list[str]], readout_groups_without_label: list[str]
) -> list[str]:
    """Get qubit and prefix from counter readout groups."""
    all_data_prefixes = set()
    for name, group in subscribed_components.items():
        if name in readout_groups_without_label:
            prefix = None
        else:
            raw_group_name = next(g for g in readout_groups_without_label if g in name)
            prefix = name.replace(f"{raw_group_name}__", "")
        for qubit in group:
            all_data_prefixes.add(qubit if not prefix else f"{qubit}__{prefix}")
    data_prefixes = sorted(list(all_data_prefixes), key=_natural_keys)
    return data_prefixes


COUNTER_INDEX = Parameter("counter_index", "Counter state index")
"""MultiQubitCounter state index."""


def _reconstruct_counter_subscribed_components(additional_run_properties: dict[str, Any]) -> dict[str, list[str]]:
    """Reconstruct counter readout groups from serialised metadata."""
    readout_groups = additional_run_properties["readout_groups"]  # type: ignore[index]
    readout_group_names = additional_run_properties["readout_group_names"]  # type: ignore[index]
    return dict(zip(readout_group_names, readout_groups))


def extract_counter_group_data(run: RunResult) -> RunResult:
    """Extracts the data for all counter readout groups into their own data variables in the dataset.

    Args:
        run: the run result.

    Returns:
        The run result with new data variables for each counter readout group added.

    """
    if not run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value:
        return run
    additional_run_properties = run.run_data.additional_run_properties
    subscribed_components = _reconstruct_counter_subscribed_components(additional_run_properties)

    def _extract_data(run_result: RunResult) -> RunResult:
        grouped_components = subscribed_components.copy()
        dataset = run_result.dataset
        data_parameter_name = run_result.run_data.default_data_parameters[0]
        group_offset = 0
        group_arrays = {}
        for group_name, group in grouped_components.items():
            basis = Parameter(f"{group_name}_multi_qubit_basis", f"{group_name} multi-qubit basis")

            states_to_pick = [group_offset + state for state in range(2 ** len(group))]
            group_variable = (
                dataset[data_parameter_name]
                .sel({COUNTER_INDEX.name: states_to_pick})
                .rename({COUNTER_INDEX.name: basis.name})
                .assign_coords({basis.name: list(range(2 ** len(group)))})
            )
            group_variable[basis.name].attrs = {
                "parameter": basis,
                "standard_name": basis.name,
                "index_variable": basis.label,
                "units": "",
            }
            group_parameter = Parameter("counter_readout", "Counter Readout")
            group_variable = annotate(group_variable, group_parameter, prefix=group_name)
            group_arrays[group_variable.name] = group_variable

            group_offset += 2 ** len(group)

        return replace(run_result, dataset=run_result.dataset.assign(group_arrays))

    return _extract_data(run)


def add_counter_averaged_readout(run: RunResult) -> RunResult:
    """Post processor to compute and add the single-qubit excited state probabilities from the multi qubit counter
    results to the dataset.

    Args:
        run: The run result.

    Returns:
        The run result with new data variables for averaged readout for each component added.

    """
    if not run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value:
        return run

    additional_run_properties = run.run_data.additional_run_properties
    readout_groups_without_label = additional_run_properties["readout_groups_without_label"]
    subscribed_components = _reconstruct_counter_subscribed_components(additional_run_properties)

    def _add_probabilities(run_result: RunResult) -> RunResult:
        component_groups = subscribed_components
        dataset = run_result.dataset

        groups_for_qubits_and_prefix: dict[tuple[str, str | None], tuple[str, tuple[str, ...]]] = {}

        for name, group in component_groups.items():
            # for each component and readout label prefix, find the group with the fewest elements
            # for optimal performance
            if name in readout_groups_without_label:
                prefix = None
            else:
                raw_group_name = next(g for g in readout_groups_without_label if g in name)
                prefix = name.replace(f"{raw_group_name}__", "")
            for qubit in group:
                if (qubit, prefix) not in groups_for_qubits_and_prefix or len(
                    groups_for_qubits_and_prefix[(qubit, prefix)][1]
                ) > len(group):
                    groups_for_qubits_and_prefix[(qubit, prefix)] = (name, tuple(group))

        new_arrays = {}
        for qubit_and_prefix, name_and_group in groups_for_qubits_and_prefix.items():
            qubit, prefix = qubit_and_prefix
            group_name = name_and_group[0]
            averaged_readout = compute_single_qubit_probability_from_counter_results(
                dataset[f"{group_name}_counter_readout"],
                f"{group_name}_multi_qubit_basis",
                component_groups[group_name][::-1],
                qubit,
            )
            new_arrays[averaged_readout.name] = annotate(
                averaged_readout, "readout", prefix=qubit if not prefix else f"{qubit}__{prefix}"
            )

        return replace(run_result, dataset=run_result.dataset.assign(new_arrays))

    return _add_probabilities(run)


def add_counts_for(
    run: RunResult,
) -> RunResult:
    """Post processor to compute and add the single-qubit excited state probabilities from the multi qubit counter
    results to the dataset.

    This is done only if the dimensions of the single qubit probabilities are equal.

    Args:
        run: The run result.

    Returns:
        The run result with the counter data added.

    """
    if run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value:
        return run

    settings = run.run_data.sweep_data.settings
    if not _is_single_shot(settings):
        return run

    additional_run_properties = run.run_data.additional_run_properties
    result_labels = additional_run_properties["integration_data_parameters"]  # type:ignore[index]
    data_prefixes = [_map_default_label(label) for label in result_labels]
    acquisition_map = _map_acquisitions(
        run.run_data.sweep_data.settings, data_prefixes, additional_run_properties.get("readout_label_to_impl")
    )

    def _add_counts(run_result: RunResult) -> RunResult:
        dataset = run_result.dataset
        single_shot_arrays = [
            dataset[f"{pref}_state_single_shot"] for pref in data_prefixes if acquisition_map[pref] == "threshold"
        ]
        single_shot_prefixes = [pref for pref in data_prefixes if acquisition_map[pref] == "threshold"]
        if not single_shot_arrays:
            return run_result
        if any(arr.dims != single_shot_arrays[0].dims for arr in single_shot_arrays):
            return run_result
        counter_parameter = Parameter(
            "counter.result",
            f"Multi-qubit state probabilities for {single_shot_prefixes}",
            data_type=DataType.FLOAT,
            collection_type=CollectionType.NDARRAY,
        )
        counter_array = convert_single_shot_to_counter(
            single_shot_arrays, COUNTER_INDEX, counter_parameter, "repetitions"
        )
        return replace(run_result, dataset=run_result.dataset.assign({counter_array.name: counter_array}))

    return _add_counts(run)


def classify_two_states(
    run: RunResult,
) -> RunResult:
    """Classify the complex data using the calibrated threshold.

    If the observations are not available, a warning is logged.

    Args:
        run: The run result.

    Returns:
        The run result with the discriminated shots added.

    """
    settings = run.run_data.sweep_data.settings
    if run.run_data.sweep_data.settings.controllers.counter.subscribed_channels.value or not _is_single_shot(settings):
        return run

    additional_run_properties = run.run_data.additional_run_properties
    result_labels = additional_run_properties["integration_data_parameters"]  # type:ignore[index]
    data_prefixes = [_map_default_label(label) for label in result_labels]
    label_to_impl = additional_run_properties.get("readout_label_to_impl")
    acquisition_map = _map_acquisitions(run.run_data.sweep_data.settings, data_prefixes, label_to_impl)

    def _classify_two_states(run_result: RunResult) -> RunResult:
        dataset = run_result.dataset
        settings = run_result.run_data.sweep_data.settings
        prefixes_without_threshold = []
        new_arrays = {}
        for prefix in data_prefixes:
            if acquisition_map.get(prefix) != "complex":
                continue

            component = prefix.split("_")[0]
            measure_gate_name, implementation_name = _get_gate_and_impl_name(prefix, label_to_impl)

            if measure_gate_name is None:
                continue

            gate_node = settings.get_gate_node_for_locus(
                measure_gate_name, component, implementation=implementation_name
            )
            if gate_node.integration_threshold.value is None:
                prefixes_without_threshold.append(prefix)
                continue

            state_parameter = Parameter(f"{prefix}_state_single_shot", f"{prefix.replace('_', ' ')} state single shot")
            state_array = (
                dataset[f"{prefix}_readout_single_shot"].real > gate_node.integration_threshold.value
            ).astype(float)
            annotate(state_array, state_parameter)
            new_arrays[state_array.name] = state_array

        if prefixes_without_threshold:
            warning_template = (
                "No value for integration_threshold was found for the data prefixes %s, "
                " cannot perform state discrimination for them."
            )
            logger.warning(warning_template, ", ".join(prefixes_without_threshold))
        return replace(run_result, dataset=run_result.dataset.assign(new_arrays))

    return _classify_two_states(run)


construct_run_result_stage = CompilationStage(
    name="construct_run_result",
    info="Aggregate raw hardware data into a structured run result object.",
)
construct_run_result_stage.add_passes(construct_run_result)
construct_data_variables_stage = CompilationStage(
    name="construct_data_variables",
    info="Map measured counts and states to high-level user variables.",
)
construct_data_variables_stage.add_passes(
    create_ragged_data_index,
    extract_counter_group_data,
    rename_data_variables,
    classify_two_states,
    average_single_shot_data,
    contrast_data,
    add_counter_averaged_readout,
    compute_excitation_probability_from_data,
    add_counts_for,
    process_metadata,
)
_STANDARD_POST_PROCESSING_STAGES = [construct_run_result_stage, construct_data_variables_stage]


# OLD PULLA-STYLE CIRCUIT POST PROCESSING


@dataclass
class CircuitExecutionResults:
    """Old Pulla style circuit execution results."""

    circuit_measurement_results: CircuitMeasurementResultsBatch
    """Circuit measurement results."""
    readout_mappings: ReadoutMappingBatch
    """Readout mappings."""
    sweep_results: SweepResults
    """Raw sweep results."""
    run_data: RunData | None
    """Run data."""


def _process_readout_metrics(metrics: ReadoutMetrics, mapped_readout_keys: dict[str, str]) -> ReadoutMappingBatch:
    """Process a :class:`ReadoutMetrics` into a :class:`ReadoutMappingBatch` instance"""
    readout_mapping_batch = []
    for seg_idx in range(metrics.num_segments):
        mapping = defaultdict(list)
        for readout_label, occurrences in metrics.integration_occurrences.items():
            unmapped_label = _unmap_label(readout_label, mapped_readout_keys)
            readout_key = re.split("__[__]*", unmapped_label)[1]  # FIXME: this is potentially fragile
            if occurrences[seg_idx] == 1:
                mapping[readout_key].append(unmapped_label)
            elif occurrences[seg_idx] > 1:
                raise ValueError(
                    "Circuit style post-processing currently not available when there are multiple"
                    " measure calls with the same readout label in a single segment."
                )
        readout_mapping_batch.append({k: tuple(v) for k, v in mapping.items()})
    return tuple(readout_mapping_batch)


def construct_circuit_execution_results(pulla_data: PullaData, context: dict[str, Any]) -> CircuitExecutionResults:
    """Construct a :class:`CircuitExecutionResults` instance from a raw :class:`PullaData` instance.

    The construction is not possible if there are multiple readout triggers per segment for a given readout label

    Args:
        pulla_data: The raw Pulla data returned by the SC.
        context: The compiler context.

    Returns:
        The constructed :class:`CircuitExecutionResults` instance.

    """
    run_data = pulla_data.run_data
    sweep_results = pulla_data.sweep_results
    if mapped_readout_keys := context.get("mapped_readout_keys", {}):
        run_data, sweep_results = _unmap_readout_keys(run_data, sweep_results, mapped_readout_keys)  # type:ignore[arg-type]
    readout_mapping_batch = _process_readout_metrics(context["readout_metrics"], context.get("mapped_readout_keys", {}))
    # NOTE: if heralding is used, it is assumed to be used in each circuit of the batch
    heralding_used = HERALDING_KEY in readout_mapping_batch[0]
    conversion_func = (
        convert_sweep_spot_to_arrays_with_heralding_mode_zero if heralding_used else convert_sweep_spot_to_arrays
    )
    # NOTE: always the same number of soft sweep spots for each readout label
    num_soft_sweep_spots = len(next(iter(sweep_results.values())))
    results: list[dict[str, np.ndarray]] = []
    for spot_idx in range(num_soft_sweep_spots):
        results.extend(
            conversion_func(
                {label: spots[spot_idx] for label, spots in sweep_results.items()},
                readout_mapping_batch,
            )
        )
    batch_results = [{mk: array.tolist() for mk, array in circuit_res.items()} for circuit_res in results]
    return CircuitExecutionResults(
        circuit_measurement_results=batch_results,
        readout_mappings=readout_mapping_batch,
        sweep_results=sweep_results,
        run_data=run_data,
    )


construct_circuit_execution_results_stage = CompilationStage(
    name="construct_circuit_execution_results",
    info="Finalizes the mapping of execution data back to the original input circuits.",
)
construct_circuit_execution_results_stage.add_passes(construct_circuit_execution_results)

_STANDARD_CIRCUIT_POST_PROCESSING_STAGES = [construct_circuit_execution_results_stage]
