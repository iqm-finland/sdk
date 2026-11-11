#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Helpers to write analysis functions."""

from collections.abc import Callable, Sequence
import inspect
import logging

import numpy as np
import xarray

from exa.common.data import SettingNode
from exa.core.analyze_result import AnalyzeResult
from exa.core.run_result import RunResult
from exa.core.util.dataset import annotate
from iqm.cpc.core.observation.observation_parameters import SINGLE_SHOT_READOUT_01_ERROR, SINGLE_SHOT_READOUT_10_ERROR

from .setting_utils import COUNTER_RESULT
from .sweep_utils import COUNTER_INDEX

logger = logging.getLogger(__name__)


def analyze_singles(
    run_result: RunResult,
    analyzer: Callable[..., None],
    qubits: Sequence[str] | None = None,
    **analysis_kwargs,
) -> AnalyzeResult:
    """A helper tool to analyze data of independent qubits.

    This function repeats the function `analyzer` on all single qubits in the experiment defined by `run_result`.
    The purpose of the `analyzer` is to modify the :class:`.AnalyzeResult` instance that is passed to it.

    This assumes that the `run_result.definition.additional_run_properties` has the keys ``"qubits"`` and
    ``"readout_components"``, and the analysis is applied to qubits that appear in both lists.

    `analyzer` will be called with arguments that match the signature in the following way:

    - The first argument ``result`` is a template :class:`.AnalyzeResult` which should be modified by `analyzer`.
      Most notably, it contains the full dataset of the run, where more data arrays can be added. It also contains the
      `run_result` so that the analysis function can access all the metadata.
    - The second argument is always ``qubit`` (str) which specifies the qubit.
    - ``target_name`` (str) is the key to the "target data" for that qubit, as defined by Experiment.
      This is to access the target array as ``result.run_result.dataset[target_name]``.
      It is passed to `analyzer` only if ``target_name"`` is in the signature.
      Assumes that "target_data_parameters" come in the same order as flattened "subscribed components".
      If "target_data_parameters" is not defined, uses "default_data_parameters" instead.
    - Other arguments can be passed with `analysis_kwargs`.

    Args:
        run_result: The run result containing the full dataset and run data.
        analyzer: Function that analyzes a single qubit.
        qubits: Optionally run analysis only for these qubits. By default, runs on all qubits.
        analysis_kwargs: Additional arguments for ``analyzer``.

    Returns:
          A new instance of AnalyzeResult, where repeated calls to `analyzer` have added content.

    """
    analyze_result = AnalyzeResult(run_result=run_result)

    sig = inspect.signature(analyzer)
    additional_run_properties = run_result.run_definition.additional_run_properties
    targets = (
        additional_run_properties.get("target_data_parameters") or run_result.run_definition.default_data_parameters
    )

    for readout_component in additional_run_properties["readout_components"]:
        if qubits is not None and readout_component not in qubits:
            continue

        if "target_name" in sig.parameters:
            index = additional_run_properties["readout_components"].index(readout_component)
            analyzer(analyze_result, readout_component, target_name=targets[index], **analysis_kwargs)
        else:
            analyzer(analyze_result, readout_component, **analysis_kwargs)

    return analyze_result


def get_inverse_assignment_matrix(error_10: float, error_01: float) -> np.ndarray:
    """Recover an inverse assignment matrix for one qubit from individual errors.

    Args:
        error_10: Probability of measuring 0 if the 1 state was prepared.
        error_01: Probability of measuring 1 if the 0 state was prepared.

    Returns:
        Numpy array, inverse assignment matrix for one qubit.

    """
    return 1 / (1 - error_10 - error_01) * np.array([[1 - error_10, -error_10], [-error_01, 1 - error_01]])


def multi_kron(*args: np.ndarray) -> np.ndarray:
    """Cascade Kronecker product of arguments in order.

    Args:
        args: NumPy arrays of arbitrary sizes representing matrices, whose elements can be
            multiplied and added together.

    Returns:
        Kronecker product of matrices in order they appear as arguments.

    """
    args_length = len(args)
    if args_length == 1:
        return args[0]
    if args_length == 2:
        return np.kron(*args)
    idx = args_length // 2
    return np.kron(multi_kron(*args[:idx]), multi_kron(*args[idx:]))


def add_mitigated_results_tensor_product_assignment(
    run_result: RunResult, settings: SettingNode, readout_group_name: str | None = None
) -> xarray.DataArray:
    """Add mitigated counter (group) readout results to dataset. Should be used in analysis functions.

    Args:
        run_result: The run result to add readout results to.
        settings: Settings used in the run.
        readout_group_name: name of the readout group to add readout results to. If None, add to the entire counter.

    Returns:
        DataArray of mitigated readout results.

    """
    dataset = run_result.dataset
    if readout_group_name is None:
        results = dataset[COUNTER_RESULT.name]
        index = dataset[COUNTER_INDEX.name]
        readout_components = run_result.run_definition.additional_run_properties["readout_components"]
    else:
        results = dataset[f"{readout_group_name}_counter_readout"]
        index = dataset[f"{readout_group_name}_multi_qubit_basis"]
        names = run_result.run_definition.additional_run_properties["readout_group_names"]
        readout_groups = run_result.run_definition.additional_run_properties["readout_groups"]
        readout_components = readout_groups[names.index(readout_group_name)]
    if len(index.values) < 2 ** (len(readout_components)):
        logger.warning(
            "Some possible outcomes are missing from the index array. This method is really basic and"
            " will not work in this case. Returning the original array."
        )
        return results

    matrices = [
        get_inverse_assignment_matrix(
            settings.get_gate_properties_for_locus("measure", component)[SINGLE_SHOT_READOUT_10_ERROR.name].value,  # type: ignore[arg-type]
            settings.get_gate_properties_for_locus("measure", component)[SINGLE_SHOT_READOUT_01_ERROR.name].value,  # type: ignore[arg-type]
        )
        for component in readout_components
    ]
    full_matrix = multi_kron(*matrices)

    # This is a trick which lets xarray keep track of which dimension of the dataset is the correct one to perform the
    # matrix multiplication against. xarray.dot sums over identical dimensions, so one dimension of our inverse
    # assignment matrix will be the same as the data dimension, but the other one needs to be something else.
    # That new dimension replaces the original one in the final dataset, so we need to rename it back at the end.
    mitigated_index = index.rename({index.name: f"{index.name}_mitigated"}).rename(f"{index.name}_mitigated")
    assignment_matrix_inverse_array = xarray.DataArray(
        full_matrix,
        dims=(index.name, mitigated_index.name),
        coords={index.name: index, mitigated_index.name: mitigated_index},
    )
    mitigated_results = assignment_matrix_inverse_array.dot(results).rename({mitigated_index.name: index.name})
    mitigated_results = annotate(mitigated_results, f"{results.name}_mitigated")
    dataset.update({mitigated_results.name: mitigated_results})
    return mitigated_results
