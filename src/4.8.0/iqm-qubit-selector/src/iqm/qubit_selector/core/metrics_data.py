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

"""QPU metrics for qubit selection, and fetching them from an IQM server.

:class:`FullMetricsData` holds every calibration value the server reports. :class:`MetricsData`
holds the subset the cost model uses (CZ, single-qubit and readout fidelities), and can be built by
hand when no server is available. Clifford fidelities, ``T1`` and ``T2`` are fetched but not scored.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum, auto
import logging
import re
from typing import TypeAlias
from uuid import UUID

from iqm.iqm_client import IQMClient

from iqm.station_control.interface.models import DynamicQuantumArchitecture, ObservationBase

logger = logging.getLogger(__name__)

_MICROSECONDS_PER_SECOND = 10**6
_TWO_QUBIT_LOCUS_LENGTH = 2


class ReadoutMode(StrEnum):
    """Which readout metric, if any, a layout's cost includes.

    A :class:`~enum.StrEnum`, so ``ReadoutMode.FIDELITY == "fidelity"``.
    """

    NONE = auto()
    """Do not score readout."""

    FIDELITY = auto()
    """Score readout with the single-shot readout fidelity."""

    QNDNESS = auto()
    """Score readout with the QNDness fidelity.

    Use it when the circuit keeps using a qubit after measuring it: QNDness covers how well the
    measured state survives the measurement, which plain readout fidelity does not. It is usually
    lower than the readout fidelity.
    """


ReadoutMetrics: TypeAlias = dict[ReadoutMode, dict[str, float]]
"""Per-qubit readout metrics, keyed by the :class:`ReadoutMode` that selects them."""


def two_qubit_key(qubit_a: str, qubit_b: str) -> str:
    """Return the key of an ordered qubit pair in the two-qubit metric dicts.

    Args:
        qubit_a: First qubit, or a computational resonator on a Star QPU.
        qubit_b: Second qubit, or a computational resonator on a Star QPU.

    Returns:
        The key string, e.g. ``"['QB1', 'QB2']"``.

    """
    return str([qubit_a, qubit_b])


def two_qubit_fidelities(pairs: Mapping[tuple[str, str], float]) -> dict[str, float]:
    """Return a two-qubit fidelity dict for :class:`MetricsData`, with both orderings of each pair.

    Both orderings are stored because a circuit may apply the gate in either direction.

    Args:
        pairs: Mapping from a qubit-name pair to its two-qubit gate fidelity.

    Returns:
        Dict keyed by ``"['QBx', 'QBy']"`` strings, holding both directions of each pair.

    Example::

        >>> two_qubit_fidelities({("QB1", "QB2"): 0.995})
        {"['QB1', 'QB2']": 0.995, "['QB2', 'QB1']": 0.995}

    """
    result: dict[str, float] = {}
    for (qubit_a, qubit_b), fidelity in pairs.items():
        result[two_qubit_key(qubit_a, qubit_b)] = fidelity
        result[two_qubit_key(qubit_b, qubit_a)] = fidelity
    return result


def _resonator_loci(arch: DynamicQuantumArchitecture, gate_name: str) -> set[tuple[str, str]]:
    """Return the ``(qubit, resonator)`` loci of a gate, deduplicated across implementations."""
    gate = arch.gates.get(gate_name)
    if gate is None:
        return set()
    resonators = set(arch.computational_resonators)
    loci: set[tuple[str, str]] = set()
    for implementation in gate.implementations.values():
        for locus in implementation.loci:
            if len(locus) != _TWO_QUBIT_LOCUS_LENGTH:
                continue
            first, second = locus
            if second in resonators and first not in resonators:
                loci.add((first, second))
            elif first in resonators and second not in resonators:
                loci.add((second, first))
    return loci


def fictional_two_qubit_fidelities(
    cz_resonator: Mapping[str, float],
    move: Mapping[str, float],
    arch: DynamicQuantumArchitecture,
) -> dict[str, float]:
    """Return effective qubit-qubit CZ fidelities for a Star architecture.

    A Star QPU has no direct qubit-qubit CZ: ``CZ(q1, q2)`` runs as
    ``MOVE(q2, r) CZ(q1, r) MOVE(q2, r)`` through a shared computational resonator ``r``, so its
    fidelity is taken as ``cz(q1, r) * move(q2, r) ** 2`` (:cite:`Renger_2025`; the same model as
    ``IQMTarget._create_fictional_cz_properties``). The pairs are those
    :func:`~iqm.iqm_client.transpile.simplify_architecture` creates, so they match the collapsed
    coupling graph.

    When a pair can be routed several ways (either qubit moved, or through several resonators), the
    best fidelity is kept, since the transpiler is free to pick that route. Pairs missing a metric are
    left out, so the cost model treats them as uncalibrated rather than perfect.

    Args:
        cz_resonator: Qubit-resonator CZ fidelities, keyed as in :attr:`FullMetricsData.cz_resonator`.
        move: Single-MOVE fidelities, keyed as in :attr:`FullMetricsData.move`.
        arch: The raw Star architecture, used for its CZ and MOVE loci.

    Returns:
        Two-qubit fidelities with both orderings of each pair, for :attr:`MetricsData.two_qubit`.

    """
    best: dict[tuple[str, str], float] = {}
    for cz_qubit, cz_res in _resonator_loci(arch, "cz"):
        cz_fidelity = cz_resonator.get(two_qubit_key(cz_qubit, cz_res))
        if cz_fidelity is None:
            continue
        for move_qubit, move_res in _resonator_loci(arch, "move"):
            if move_res != cz_res or move_qubit == cz_qubit:
                continue
            move_fidelity = move.get(two_qubit_key(move_qubit, move_res))
            if move_fidelity is None:
                continue
            pair = (cz_qubit, move_qubit) if cz_qubit < move_qubit else (move_qubit, cz_qubit)
            fidelity = cz_fidelity * move_fidelity**2
            if fidelity > best.get(pair, 0.0):
                best[pair] = fidelity
    return two_qubit_fidelities(best)


@dataclass
class MetricsData:
    """QPU metrics the cost model scores layouts with.

    Single-qubit metrics are keyed by qubit name (e.g. ``"QB1"``), two-qubit fidelities by
    :func:`two_qubit_key` of an ordered pair. Both orderings of a pair must be present;
    :func:`two_qubit_fidelities` adds them.

    Without a server, build it by hand::

        metrics = MetricsData(
            two_qubit=two_qubit_fidelities({("QB1", "QB2"): 0.995, ("QB2", "QB3"): 0.990}),
            single_qubit={"QB1": 0.999, "QB2": 0.998, "QB3": 0.997},
            readout={
                ReadoutMode.FIDELITY: {"QB1": 0.98, "QB2": 0.97, "QB3": 0.96},
                ReadoutMode.QNDNESS: {"QB1": 0.95, "QB2": 0.94, "QB3": 0.93},
            },
        )

    Only the readout entry for the requested :class:`ReadoutMode` is read, so others may be left out.

    """

    two_qubit: dict[str, float] = field(default_factory=dict)
    """Two-qubit (CZ) gate fidelities, keyed by :func:`two_qubit_key`."""

    single_qubit: dict[str, float] = field(default_factory=dict)
    """Single-qubit gate fidelities, keyed by qubit name."""

    readout: ReadoutMetrics = field(default_factory=dict)
    """Readout metrics per qubit, keyed by the :class:`ReadoutMode` that selects them."""


@dataclass
class FullMetricsData:
    """Every calibration value fetched from the server; :meth:`to_metrics_data` extracts what is scored."""

    cz: dict[str, float] = field(default_factory=dict)
    """Qubit-qubit CZ fidelities, keyed by :func:`two_qubit_key`.

    Empty on a Star QPU, whose CZ gates all act on a qubit and a resonator; see
    :attr:`cz_resonator`."""

    cz_resonator: dict[str, float] = field(default_factory=dict)
    """Qubit-resonator CZ fidelities on a Star QPU, keyed by :func:`two_qubit_key`.

    :func:`fictional_two_qubit_fidelities` combines them with :attr:`move` into qubit-qubit ones."""

    move: dict[str, float] = field(default_factory=dict)
    """Fidelity of a single MOVE per qubit-resonator pair on a Star QPU, keyed by
    :func:`two_qubit_key`. A fictional CZ uses two MOVEs."""

    clifford: dict[str, float] = field(default_factory=dict)
    """Two-qubit Clifford fidelities, keyed by :func:`two_qubit_key`. Not scored."""

    single_qubit: dict[str, float] = field(default_factory=dict)
    """Single-qubit (PRX) gate fidelities, keyed by qubit name."""

    readout: dict[str, float] = field(default_factory=dict)
    """Readout fidelities, keyed by qubit name."""

    readout_qndness: dict[str, float] = field(default_factory=dict)
    """Readout QNDness fidelities, keyed by qubit name."""

    t1: dict[str, float] = field(default_factory=dict)
    """``T1`` relaxation times in microseconds, keyed by qubit name."""

    t2: dict[str, float] = field(default_factory=dict)
    """``T2`` coherence times in microseconds, keyed by qubit name."""

    def to_metrics_data(self) -> MetricsData:
        """Return the metrics the cost model scores with: CZ, single-qubit and readout fidelities.

        On a Star QPU the two-qubit fidelities are empty here; use
        :meth:`MetricsDataManager.get_metrics_data`, which fills in the fictional CZ fidelities.

        Returns:
            The scored subset of the metrics.

        """
        return MetricsData(
            two_qubit=self.cz,
            single_qubit=self.single_qubit,
            readout={
                ReadoutMode.FIDELITY: self.readout,
                ReadoutMode.QNDNESS: self.readout_qndness,
            },
        )


def _contains_all(text: str, markers: Iterable[str]) -> bool:
    """Return whether ``text`` contains every marker substring."""
    return all(marker in text for marker in markers)


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    """Return whether ``text`` contains at least one marker substring."""
    return any(marker in text for marker in markers)


def _locus_components(field: str, components: set[str]) -> list[str]:
    r"""Return the components named in an observation field, in order of appearance.

    Observation field looks like ``gates.cz.tgss.QB1__QB2.fidelity``. Whole tokens are matched against the
    architecture rather than a ``QB\d+`` pattern, so resonator loci such as ``QB1__COMPR1`` are found
    too and ``QB1`` is never mistaken for ``QB11``.
    """
    return [token for token in re.split(r"\.|__", field) if token in components]


def _store_pair(target: dict[str, float], field: str, value: float, components: set[str]) -> None:
    """Store a two-component fidelity under both loci orderings, if the field names two components."""
    matches = _locus_components(field, components)
    if len(matches) >= _TWO_QUBIT_LOCUS_LENGTH:
        target[two_qubit_key(matches[0], matches[1])] = value
        target[two_qubit_key(matches[1], matches[0])] = value


def _implementation_names(arch: DynamicQuantumArchitecture, gate_name: str) -> set[str]:
    """Return the implementation names of a gate, or an empty set if the gate is absent."""
    gate = arch.gates.get(gate_name)
    return set(gate.implementations) if gate is not None else set()


def _default_implementation(arch: DynamicQuantumArchitecture, gate_name: str) -> str | None:
    """Return the default implementation name of a gate, or ``None`` if the gate is absent."""
    gate = arch.gates.get(gate_name)
    return gate.default_implementation if gate is not None else None


def _gate_pair_keys(arch: DynamicQuantumArchitecture, gate_name: str) -> set[str]:
    """Return the valid two-component pair keys (both orderings) from a gate's loci."""
    gate = arch.gates.get(gate_name)
    if gate is None:
        return set()
    keys: set[str] = set()
    for implementation in gate.implementations.values():
        for locus in implementation.loci:
            if len(locus) == _TWO_QUBIT_LOCUS_LENGTH:
                keys.add(two_qubit_key(locus[0], locus[1]))
                keys.add(two_qubit_key(locus[1], locus[0]))
    return keys


def _qubit_name(field: str) -> str:
    """Extract a single qubit name (e.g. ``"QB3"``) from an observation field."""
    return "QB" + field.split("QB")[1].split(".", maxsplit=1)[0]


def _keep(metrics: dict[str, float], valid_keys: set[str]) -> dict[str, float]:
    """Drop metric entries whose keys are not part of the architecture."""
    return {key: value for key, value in metrics.items() if key in valid_keys}


def _sort_by_qubit(metrics: dict[str, float]) -> dict[str, float]:
    """Sort single-qubit metrics naturally by their ``QB`` index."""

    def qubit_index(item: tuple[str, float]) -> int:
        match = re.search(r"QB(\d+)", item[0])
        return int(match.group(1)) if match else 0

    return dict(sorted(metrics.items(), key=qubit_index))


class MetricsDataManager:
    """Fetch QPU metrics from an IQM server, or parse them from calibration observations.

    Takes an :class:`~iqm.iqm_client.iqm_client.IQMClient` rather than an SDK backend, so it works the
    same for Qiskit, Cirq and other front ends.
    """

    _FIDELITY_MARKER = ".fidelity"
    _READOUT_TOKENS = ("measure_fidelity", "ssro")
    """Tokens marking a readout-fidelity observation; the server reports single-shot readout fidelity
    as ``metrics.ssro.measure.<impl>.QBx.fidelity`` (the ``measure_fidelity`` quality metric)."""
    _QNDNESS_MARKERS = ("metrics.rrc.", "measure", ".qndness")
    """Substrings that must all appear in a readout-QNDness observation.

    Calibrations report QNDness from readout randomized correlations, as
    ``metrics.rrc.measure.<impl>.QBx.qndness``.
    """
    _T1_MARKERS = ("t1", "QB")
    _T2_MARKERS = ("t2", "QB")
    _SINGLE_QUBIT_GATE = "prx"

    def get_full_metrics_data(self, client: IQMClient, calibration_set_id: UUID | None = None) -> FullMetricsData:
        """Return every metric of a calibration set.

        Args:
            client: IQM client instance.
            calibration_set_id: Calibration set to fetch. If ``None``, the default one is used.

        Returns:
            All parsed calibration metrics.

        """
        arch, observations = self._fetch(client, calibration_set_id)
        return self.full_metrics_from_observations(observations, arch)

    def get_metrics_data(self, client: IQMClient, calibration_set_id: UUID | None = None) -> MetricsData:
        """Return the metrics of a calibration set that the cost model scores with.

        Pass ``arch.calibration_set_id`` to get the metrics matching an architecture you already have.

        Args:
            client: IQM client instance.
            calibration_set_id: Calibration set to fetch. If ``None``, the default one is used.

        Returns:
            The scored subset of the metrics; see :meth:`metrics_from_observations`.

        """
        arch, observations = self._fetch(client, calibration_set_id)
        return self.metrics_from_observations(observations, arch)

    def metrics_from_observations(
        self, observations: Iterable[ObservationBase], arch: DynamicQuantumArchitecture
    ) -> MetricsData:
        """Return the metrics the cost model scores with, parsed from calibration observations.

        On a Star architecture the two-qubit fidelities are the fictional qubit-qubit CZ fidelities
        from :func:`fictional_two_qubit_fidelities`, so they match the collapsed coupling graph from
        :func:`~iqm.qubit_selector.core.graph_builder.build_coupling_graph`.

        Args:
            observations: Calibration observations, e.g. ``quality_metric_set.observations``.
            arch: Architecture of the calibration the observations belong to.

        Returns:
            The scored subset of the metrics.

        """
        full = self.full_metrics_from_observations(observations, arch)
        metrics = full.to_metrics_data()
        if arch.computational_resonators:
            metrics.two_qubit = fictional_two_qubit_fidelities(full.cz_resonator, full.move, arch)
        return metrics

    def full_metrics_from_observations(
        self, observations: Iterable[ObservationBase], arch: DynamicQuantumArchitecture
    ) -> FullMetricsData:
        """Return every metric in calibration observations; non-numeric values are skipped.

        Args:
            observations: Calibration observations, e.g. ``quality_metric_set.observations``.
            arch: Architecture of the calibration the observations belong to.

        Returns:
            All parsed calibration metrics.

        """
        single_qubit_impls = _implementation_names(arch, self._SINGLE_QUBIT_GATE)
        two_qubit_impls = _implementation_names(arch, "cz")
        move_impls = _implementation_names(arch, "move")
        measure_impl = _default_implementation(arch, "measure")
        valid_two_qubit_keys = _gate_pair_keys(arch, "cz")
        valid_move_keys = _gate_pair_keys(arch, "move")
        components = set(arch.components)
        resonators = set(arch.computational_resonators)

        cz: dict[str, float] = {}
        cz_resonator: dict[str, float] = {}
        move: dict[str, float] = {}
        clifford: dict[str, float] = {}
        single_qubit: dict[str, float] = {}
        readout: dict[str, float] = {}
        readout_qndness: dict[str, float] = {}
        t1: dict[str, float] = {}
        t2: dict[str, float] = {}

        field_values = (
            (observation.dut_field, float(observation.value))
            for observation in observations
            if isinstance(observation.value, int | float)
        )
        for dut_field, value in field_values:
            if _contains_all(dut_field, self._QNDNESS_MARKERS):
                readout_qndness[_qubit_name(dut_field)] = value
            elif self._is_readout(dut_field, measure_impl):
                readout[_qubit_name(dut_field)] = value
            elif self._SINGLE_QUBIT_GATE in dut_field and _contains_any(dut_field, single_qubit_impls):
                single_qubit[_qubit_name(dut_field)] = value
            elif _contains_all(dut_field, self._T1_MARKERS):
                t1[_qubit_name(dut_field)] = value * _MICROSECONDS_PER_SECOND
            elif _contains_all(dut_field, self._T2_MARKERS):
                t2[_qubit_name(dut_field)] = value * _MICROSECONDS_PER_SECOND
            elif self._FIDELITY_MARKER in dut_field and "clifford" in dut_field:
                _store_pair(clifford, dut_field, value, components)
            elif self._FIDELITY_MARKER in dut_field and "move" in dut_field and _contains_any(dut_field, move_impls):
                _store_pair(move, dut_field, value, components)
            elif self._FIDELITY_MARKER in dut_field and "cz" in dut_field and _contains_any(dut_field, two_qubit_impls):
                # A Star QPU's CZ loci pair a qubit with a resonator; keeping them apart leaves
                # ``cz`` a qubit-qubit map on every architecture.
                pair = _locus_components(dut_field, components)
                target = cz_resonator if any(name in resonators for name in pair) else cz
                _store_pair(target, dut_field, value, components)

        return FullMetricsData(
            cz=dict(sorted(_keep(cz, valid_two_qubit_keys).items())),
            cz_resonator=dict(sorted(_keep(cz_resonator, valid_two_qubit_keys).items())),
            move=dict(sorted(_keep(move, valid_move_keys).items())),
            clifford=dict(sorted(_keep(clifford, valid_two_qubit_keys).items())),
            single_qubit=_sort_by_qubit(_keep(single_qubit, components)),
            readout=_sort_by_qubit(_keep(readout, components)),
            readout_qndness=_sort_by_qubit(_keep(readout_qndness, components)),
            t1=_sort_by_qubit(_keep(t1, components)),
            t2=_sort_by_qubit(_keep(t2, components)),
        )

    def _fetch(
        self, client: IQMClient, calibration_set_id: UUID | None
    ) -> tuple[DynamicQuantumArchitecture, Iterable[ObservationBase]]:
        """Fetch the architecture and the quality metrics of the same calibration set."""
        arch = client.get_dynamic_quantum_architecture(calibration_set_id)
        return arch, client.get_quality_metric_set(arch.calibration_set_id).observations

    def _is_readout(self, field: str, measure_impl: str | None) -> bool:
        """Check whether given observation field is a readout fidelity of the default measure gate."""
        if self._FIDELITY_MARKER not in field or not _contains_any(field, self._READOUT_TOKENS):
            return False
        return measure_impl is None or measure_impl in field
