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
"""Utility functions for the circuit-to-pulse compiler."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sequence
from itertools import chain
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from iqm.cpc.compiler.standard_stages import HERALDING_KEY
from iqm.cpc.quantum_architecture import CalibrationSetValues, ObservationValue
from iqm.station_control.client.qon import QON, QONGateParam
from iqm.station_control.interface.models import HeraldingMode
from iqm.station_control.interface.models.circuit import _CircuitMeasurementResultsNew

if TYPE_CHECKING:
    from iqm.cpc.interface.circuit_execution import ReadoutMappingBatch
    from iqm.pulse.gate_implementation import OpCalibrationDataTree
    from iqm.station_control.interface.models import (
        CircuitMeasurementResultsBatch,
        ObservationBase,
    )


logger = logging.getLogger(__name__)


class PostSelectionError(Exception):
    """Error in performing post selection via heralded results."""


def _get_trigger_indexing_for(readout_mappings: ReadoutMappingBatch) -> tuple[dict[str, int], list[list[str]]]:
    """Information for deciphering the circuit batch RO results returned by Station Control.

    Args:
        readout_mappings: RO mappings for the circuit batch

    Returns:
        mapping from acquisition label to total number of times it appears in the batch (at most once per circuit),
        acquisition labels present in each circuit in the batch

    """
    num_triggers_for_label: dict[str, int] = {}  # how many triggers for each RO acq label
    labels_for_circuit: list[list[str]] = []  # which RO acquisition labels were present in each circuit
    for readout_mapping in readout_mappings:
        circuit_labels: list[str] = []
        for labels in readout_mapping.values():
            circuit_labels.extend(labels)
            for label in labels:
                if label not in num_triggers_for_label:
                    num_triggers_for_label[label] = 1
                else:
                    num_triggers_for_label[label] += 1
        labels_for_circuit.append(circuit_labels)

    return num_triggers_for_label, labels_for_circuit


def _result_idx(label: str, circuit_idx: int, labels_for_circuit: list[list[str]]) -> int:
    """Index of the RO result that corresponds to the given acquisition label in the given circuit."""
    return len([circuit_labels for circuit_labels in labels_for_circuit[:circuit_idx] if label in circuit_labels])


def convert_sweep_spot_to_arrays(
    results: dict[str, np.ndarray], readout_mappings: ReadoutMappingBatch
) -> Iterator[_CircuitMeasurementResultsNew]:
    """Convert the sweep measurement results from Station Control into circuit batch measurement results.

    Args:
        results: mapping of acquisition labels to 1D arrays of readout results with the length
            ``num_shots * num_triggers_for_label_in_batch``
        readout_mappings: for each circuit in the batch, a mapping of measurement keys to corresponding
            tuples of acquisition labels

    Yields:
        Converted measurement results for each circuit in the batch.

    """

    def _fix_typing(arr: np.ndarray) -> np.ndarray:
        """Cast thresholded (int) results into np.uint8."""
        if np.iscomplex(arr).any() or np.modf(arr)[0].any():
            return arr
        return arr.astype(np.uint8)

    num_triggers_for_label, labels_for_circuit = _get_trigger_indexing_for(readout_mappings)
    first_key = next(iter(results))
    num_shots = len(results[first_key]) // num_triggers_for_label[first_key]  # num shots equal for all labels
    results = {
        label: _fix_typing(measurements.reshape((num_shots, num_triggers_for_label[label])))
        for label, measurements in results.items()
    }
    for circuit_idx, readout_mapping in enumerate(readout_mappings):
        yield {
            mk: np.stack(
                [results[label][:, _result_idx(label, circuit_idx, labels_for_circuit)] for label in result_labels],
                axis=1,
            )
            for mk, result_labels in readout_mapping.items()
        }


def convert_sweep_spot_to_arrays_with_heralding_mode_zero(
    results: dict[str, np.ndarray], readout_mappings: ReadoutMappingBatch
) -> Iterator[_CircuitMeasurementResultsNew]:
    """Like :func:`convert_sweep_spot_to_arrays`, but for results that contain heralding measurements.

    * For each circuit we only keep the shots for which the heralding result is zero for all the
      qubits used in the circuit.

    Args:
        results: Mapping of acquisition labels to 1D arrays of readout results with the length
            ``num_shots * num_triggers_for_label_in_batch``. The herald
            results are found under ``HERALDING_KEY``.
        readout_mappings: For each circuit in the batch, a mapping of measurement keys to corresponding
            tuples of acquisition labels.

    Yields:
        Converted measurement results for each circuit in the batch, filtered based on the result of
            the heralding measurement, , with the heralding measurement data removed.

    """
    for data in results.values():
        if np.iscomplex(data).any():
            raise PostSelectionError("Complex readout results are not supported in post-selection.")
    num_triggers_for_label, labels_for_circuit = _get_trigger_indexing_for(readout_mappings)
    first_key = next(iter(results.keys()))
    num_shots = len(results[first_key]) // num_triggers_for_label[first_key]  # num shots equal for all labels
    results = {
        label: measurements.reshape((num_shots, num_triggers_for_label[label])).astype(np.uint8)
        for label, measurements in results.items()
    }
    for circuit_idx, readout_mapping in enumerate(readout_mappings):
        # only use the wanted data for each circuit
        herald_readout_labels = readout_mapping[HERALDING_KEY]
        # For each circuit, we only keep those shots for which the heralding result is zero for
        # all the active qubits used in that circuit.
        herald_results = np.stack(
            [results[label][:, _result_idx(label, circuit_idx, labels_for_circuit)] for label in herald_readout_labels],
            axis=0,
        )
        mask = np.all(herald_results == 0, axis=0)
        if not np.any(mask):
            # TODO this is the best we can do right now, since the current iqm-client transfer format
            # cannot handle returning zero shots
            raise PostSelectionError(f"Execution of circuit {circuit_idx} with heralding discarded all the shots.")
        yield {
            mk: np.stack(
                [results[label][mask, _result_idx(label, circuit_idx, labels_for_circuit)] for label in labels],
                axis=1,
            )
            for mk, labels in readout_mapping.items()
            if mk != HERALDING_KEY
        }


# TODO (Ville): unused in code, should we remove it?
def map_sweep_results_to_logical_qubits(
    sweep_results: dict[str, list[np.ndarray]], readout_mappings: ReadoutMappingBatch, heralding_mode: HeraldingMode
) -> CircuitMeasurementResultsBatch:
    """Convert sweep results returned by Station Control to the circuit measurement results the client expects.

    Args:
        sweep_results: mapping of acquisition labels to a list of soft sweep spots, each represented by a 1D
            array of readout results, with ``shots * num_triggers_for_label`` elements.
        readout_mappings: for each circuit in the batch, a mapping of measurement keys to corresponding
            tuples of result parameter names.
        heralding_mode: Whether we use heralded readout or not.

    Returns:
        Converted, filtered measurement results, with the heralding measurement data removed.

    """
    # TODO the SC return data format should be rationalized, for example list[dict[str, np.ndarray]]
    # where the list has soft sweep spots, dict keys are result labels, and the array has shape
    # (shots, num_triggers_for_label) where the latter represents the hard sweep/circuit batch.

    # circuit execution uses just one soft sweep spot
    results = {k: v[0] for k, v in sweep_results.items()}
    if heralding_mode != HeraldingMode.NONE:
        try:
            # TODO 99% of the time in this function is spent in the tolist() converting the arrays to ints.
            # For large datasets, it is several seconds.
            # To rectify this, we should change CircuitMeasurementResultsBatch of the public API.
            circuit_results_iter = convert_sweep_spot_to_arrays_with_heralding_mode_zero(results, readout_mappings)
            return [{mk: array.tolist() for mk, array in circuit_res.items()} for circuit_res in circuit_results_iter]
        except PostSelectionError:
            logger.warning(
                "Cannot perform post-selection based on herald measurement results in complex readout mode. "
                "Returning all results."
            )
    circuit_results_iter = convert_sweep_spot_to_arrays(results, readout_mappings)
    return [{mk: array.tolist() for mk, array in circuit_res.items()} for circuit_res in circuit_results_iter]


def _extract_readout_controller_result_names(readout_mappings: ReadoutMappingBatch) -> set[str]:
    """Prepare readout controller names for a sweep request."""
    return {item for mapping in readout_mappings for item in chain(*mapping.values())}


def calset_to_cal_data_tree(calibration_set_values: CalibrationSetValues) -> OpCalibrationDataTree:
    """Build an iqm-pulse QuantumOp calibration data tree from a calibration set.

    Splits the dotted observation names that are prefixed with "gates." into the corresponding
    calibration data tree paths.
    """

    def set_path(node: dict[Hashable, Any], path: Sequence[Hashable], value: ObservationValue) -> None:
        """Insert ``value`` into the tree ``node``, at the location given by ``path``.

        Modifies ``node``.
        """
        if len(path) == 1:
            node[path[0]] = value
            return
        # recurse into a subnode
        set_path(node.setdefault(path[0], {}), path[1:], value)

    tree: OpCalibrationDataTree = {}
    for key, value in calibration_set_values.items():
        if not key.startswith("gates."):
            # shortcut
            continue
        qon = QON.from_str(key)
        if not isinstance(qon, QONGateParam):
            continue
        # empty locus? TODO should this be in QONGateParam?
        locus = () if qon.locus == ("",) else qon.locus
        path = qon.parameter.split(".")
        # mypy likes this
        set_path(tree.setdefault(qon.gate, {}).setdefault(qon.implementation, {}).setdefault(locus, {}), path, value)  # type: ignore[arg-type]
    return tree


def calset_from_observations(calset_observations: Iterable[ObservationBase]) -> CalibrationSetValues:
    """Create a calibration set from the given observations.

    Args:
        calset_observations: observations that form a calibration set

    Returns:
        calibration set

    """
    return {obs.dut_field: obs.value for obs in calset_observations}
