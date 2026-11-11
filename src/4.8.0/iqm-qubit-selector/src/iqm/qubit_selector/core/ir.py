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

"""Extract interaction graphs from IQM circuit representations."""

from collections import Counter

from iqm.pulse import Circuit

# Operations that do not contribute to the interaction graph
_SKIP_OPS = frozenset({"barrier", "delay", "measure"})

_MEASURE_OPS = frozenset({"measure", "measure_fidelity"})
"""Operations that read a qubit out."""

_TWO_QUBIT_LOCUS_SIZE = 2
"""Number of qubits in the locus of a two-qubit gate."""

InteractionGraph = dict[frozenset[str], int]
"""Weighted edge dict: frozenset of two qubit names -> count of 2-qubit gates on that pair."""


def interactions_from_circuit(circuit: Circuit) -> InteractionGraph:
    """Return how many 2-qubit gates act on each qubit pair of ``circuit``.

    Barriers, delays and measurements are skipped even when they span two qubits, since they do not
    need the qubits to be coupled.

    Args:
        circuit: An IQM pulse circuit.

    Returns:
        Weighted interaction graph.

    """
    counter: Counter[frozenset[str]] = Counter()
    for op in circuit.instructions:
        if op.name in _SKIP_OPS:
            continue
        if len(op.locus) == _TWO_QUBIT_LOCUS_SIZE:
            counter[frozenset(op.locus)] += 1
    return dict(counter)


def single_qubit_gate_counts(circuit: Circuit) -> dict[str, int]:
    """Return the number of single-qubit gates on each qubit of ``circuit``.

    Args:
        circuit: An IQM pulse circuit.

    Returns:
        Mapping from qubit name to number of single-qubit gates on it.

    """
    counter: Counter[str] = Counter()
    for op in circuit.instructions:
        if op.name in _SKIP_OPS:
            continue
        if len(op.locus) == 1:
            counter[op.locus[0]] += 1
    return dict(counter)


def measured_qubits(circuit: Circuit) -> set[str]:
    """Return the qubits ``circuit`` measures.

    Args:
        circuit: An IQM pulse circuit.

    Returns:
        Qubits with a ``measure`` or ``measure_fidelity`` operation. A measurement of several qubits
        at once, as IQM Pulse and Cirq circuits use, counts for each of them.

    """
    return {qubit for op in circuit.instructions if op.name in _MEASURE_OPS for qubit in op.locus}


def active_qubits(circuit: Circuit) -> set[str]:
    """Return the qubits ``circuit`` needs a place for.

    Barriers, delays and ``measure`` are ignored, so a qubit that is only measured or delayed is
    not counted. On a Star QPU, remove the circuit's ``MOVE`` gates first
    (``transpile_remove_moves``); otherwise the computational resonator is returned as well.

    Args:
        circuit: An IQM pulse circuit.

    Returns:
        Set of qubit names.

    """
    result: set[str] = set()
    for op in circuit.instructions:
        if op.name not in _SKIP_OPS:
            result.update(op.locus)
    return result
