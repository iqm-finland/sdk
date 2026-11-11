# Copyright 2022-2026 IQM
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

"""Entry points that rank candidate qubit layouts for a circuit."""

import logging

from iqm.iqm_client import IQMClient
from iqm.iqm_client.transpile import transpile_remove_moves
from iqm.qubit_selector.core.cost import FastEdgeCost, LayoutScore
from iqm.qubit_selector.core.enumerator import Layout, make_enumerator
from iqm.qubit_selector.core.graph_builder import build_coupling_graph, remove_qubits_from_graph
from iqm.qubit_selector.core.ir import (
    InteractionGraph,
    active_qubits,
    interactions_from_circuit,
    measured_qubits,
    single_qubit_gate_counts,
)
from iqm.qubit_selector.core.metrics_data import MetricsData, MetricsDataManager, ReadoutMode

from iqm.pulse import Circuit
from iqm.station_control.interface.models import DynamicQuantumArchitecture

logger = logging.getLogger(__name__)

_INPUTS_ERROR = "Give either 'client' alone, or 'arch' and 'metrics' without a client."
"""Raised when the QPU is not given in one of the two accepted ways."""


def score_layouts(
    circuit: Circuit,
    arch: DynamicQuantumArchitecture | None = None,
    metrics: MetricsData | None = None,
    *,
    client: IQMClient | None = None,
    readout_mode: ReadoutMode = ReadoutMode.NONE,
    remove_qubits_list: list[str] | None = None,
    max_num_layouts: int = 20000,
) -> list[LayoutScore]:
    """Score every candidate layout for ``circuit``, best first, with the error split by source.

    Prefer this over :func:`select_layout` for deep circuits, where every candidate's error rounds to
    100% but :attr:`~iqm.qubit_selector.core.cost.LayoutCost.total` still tells them apart.

    On a Star architecture the circuit's ``MOVE`` gates are removed and the QPU is collapsed to
    qubit-qubit couplings, so ``metrics`` must hold the matching fictional CZ fidelities, as
    :meth:`~iqm.qubit_selector.core.metrics_data.MetricsDataManager.get_metrics_data` returns them.

    Give the QPU either as ``client``, from which the architecture and metrics are fetched on every
    call, or as ``arch`` and ``metrics`` without a client.

    Args:
        circuit: The circuit to place.
        arch: Architecture of the target QPU. Leave out when ``client`` is given.
        metrics: QPU metrics used for scoring. Leave out when ``client`` is given.
        client: Client to fetch the architecture and metrics from.
        readout_mode: Which readout metric to include, if any.
        remove_qubits_list: Physical qubits to keep out of every layout, e.g. known-bad ones.
        max_num_layouts: Maximum number of candidate layouts to score. On a sparse coupling graph it
            is also the VF2 state limit of each search; see
            :class:`~iqm.qubit_selector.core.enumerator.VF2Enumerator` for how to size it.

    Returns:
        All scored layouts, lowest :attr:`~iqm.qubit_selector.core.cost.LayoutCost.total` first.

    Raises:
        ValueError: If the QPU is not given as either ``client`` alone, or ``arch`` and ``metrics``
            without a client.
        ValueError: If the circuit acts on more qubits than the QPU has left after ``remove_qubits_list``.

    """
    arch, metrics = _resolve_inputs(arch, metrics, client)

    # Step 0: Collapse a Star circuit onto its equivalent resonator-free form, so that a
    # resonator-mediated cz(QB, COMPR) reads as the qubit-qubit interaction it actually encodes.
    if arch.computational_resonators:
        circuit = transpile_remove_moves(circuit)

    # Step 1: Extract circuit properties from the IR
    interaction_graph = interactions_from_circuit(circuit)
    sqg_counts = single_qubit_gate_counts(circuit)
    measured = measured_qubits(circuit)
    active = active_qubits(circuit)

    # Step 2: Build coupling graph
    coupling_graph = build_coupling_graph(arch)
    if remove_qubits_list:
        coupling_graph = remove_qubits_from_graph(coupling_graph, remove_qubits_list)

    if not active:
        logger.warning("Circuit has no active qubits")
        return []

    # Step 3: Enumerate layouts with the enumerator suited to the coupling graph
    layouts = make_enumerator(coupling_graph, max_num_layouts=max_num_layouts).enumerate(interaction_graph, len(active))

    if not layouts:
        logger.warning("No layouts found. Consider increasing 'max_num_layouts' or checking the circuit/architecture.")
        return []

    logger.info("Evaluating %d candidate layouts", len(layouts))

    # Step 4: Fast cost ranking
    fast_cost = FastEdgeCost(metrics, readout_mode)
    ranked = fast_cost.sort_by_scores(
        layouts, interaction_graph, sqg_counts, measured, _logical_order(interaction_graph, active)
    )

    best = ranked[0]
    logger.info(
        "Best layout error budget: %.4f nats (2Q %.4f, 1Q %.4f, readout %.4f), error %.4f%%",
        best.total,
        best.two_qubit,
        best.single_qubit,
        best.readout,
        100.0 * best.error,
    )
    return ranked


def select_layout(
    circuit: Circuit,
    arch: DynamicQuantumArchitecture | None = None,
    metrics: MetricsData | None = None,
    *,
    client: IQMClient | None = None,
    num_layouts: int = 1,
    readout_mode: ReadoutMode = ReadoutMode.NONE,
    remove_qubits_list: list[str] | None = None,
    max_num_layouts: int = 20000,
) -> list[tuple[Layout, float]]:
    """Return the best ``num_layouts`` layouts for ``circuit`` with their total errors.

    The errors are ``1 - exp(-total)`` and saturate at 1.0 for deep circuits, where every candidate
    then looks equally bad; use :func:`score_layouts` in that case.

    Give the QPU as in :func:`score_layouts`.

    Args:
        circuit: The circuit to place.
        arch: Architecture of the target QPU. Leave out when ``client`` is given.
        metrics: QPU metrics used for scoring. Leave out when ``client`` is given.
        client: Client to fetch the architecture and metrics from.
        num_layouts: Number of layouts to return.
        readout_mode: Which readout metric to include, if any.
        remove_qubits_list: Physical qubits to keep out of every layout, e.g. known-bad ones.
        max_num_layouts: Maximum number of candidate layouts to score; see :func:`score_layouts`.

    Returns:
        ``(layout, error)`` pairs, best first, so ``result[0][0]`` is the best layout.

    Raises:
        ValueError: If the QPU is not given as either ``client`` alone, or ``arch`` and ``metrics``
            without a client.
        ValueError: If the circuit acts on more qubits than the QPU has left after ``remove_qubits_list``.

    """
    top = score_layouts(
        circuit,
        arch,
        metrics,
        client=client,
        readout_mode=readout_mode,
        remove_qubits_list=remove_qubits_list,
        max_num_layouts=max_num_layouts,
    )[:num_layouts]

    return [(result.layout, result.error) for result in top]


def _resolve_inputs(
    arch: DynamicQuantumArchitecture | None,
    metrics: MetricsData | None,
    client: IQMClient | None,
) -> tuple[DynamicQuantumArchitecture, MetricsData]:
    """Return the architecture and metrics to score with, fetching both from ``client`` if given."""
    if client is None:
        if arch is None or metrics is None:
            raise ValueError(_INPUTS_ERROR)
        return arch, metrics

    if arch is not None or metrics is not None:
        raise ValueError(_INPUTS_ERROR)
    arch = client.get_dynamic_quantum_architecture()
    logger.info("Fetching the metrics of calibration set %s from the server", arch.calibration_set_id)
    return arch, MetricsDataManager().get_metrics_data(client, arch.calibration_set_id)


def _logical_order(interaction_graph: InteractionGraph, active: set[str]) -> list[str]:
    """Return the logical qubit of each layout position, ``sorted(core) + sorted(extras)``.

    This has to match the order the enumerators fill layouts in
    (:meth:`~iqm.qubit_selector.core.enumerator.BaseEnumerator.enumerate`). Sorting all qubits
    together differs as soon as a single-qubit-only qubit sorts before a core one, and the cost model
    would then read each qubit's metrics from the wrong physical qubit.

    Args:
        interaction_graph: Circuit's 2-qubit interaction graph.
        active: All qubits the circuit acts on.

    Returns:
        Logical qubit names, one per layout position.

    """
    core_qubits = sorted({qubit for edge in interaction_graph for qubit in edge})
    return core_qubits + sorted(active - set(core_qubits))
