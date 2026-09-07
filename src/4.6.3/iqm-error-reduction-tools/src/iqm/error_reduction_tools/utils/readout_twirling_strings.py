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

"""Readout twirling string generation utilities.

This module provides helper functions to build per-qubit ``I/X`` twirling
strings used by REM workflows. The supported strategies are:

- ``HADAMARD``: Ensure balanced coverage of ``I/X`` combinations across all qubits PAIRS.
    Requires up to ``2n`` strings for ``n`` active qubits.
- ``MINIMAL``: Two alternating strings.
- ``LOCAL``: 4 strings balancing nearest-neighbor pair combinations.
- ``PROBELINES``: ``LOCAL`` strings plus probe-line specific Hadamard masks.
    The number of strings should be a multiple of 4, ideally four times the smaller
    power of 2 greater than the largest control line length.
    This ensures balanced coverage across all pairs of qubits connected by a control line.

.. note::

    All strategies ensure that, for each qubit, the number of ``I`` and ``X`` operators is equal.

"""

from .hadamard import HadamardGenerator
from ..readout_characterization.topologies import QPUTopology


def generate_rot_strings(
    qpu_topology: QPUTopology | None = None,
    active_qubits: list[str] | None = None,
    strategy: str = "LOCAL",
    max_twirling: int | None = None,
) -> list[dict[str, str]]:
    """Generate readout twirling strings for the requested strategy.

    Args:
        qpu_topology: Topology object providing qubits, neighbors, and control
            lines.  Required for the ``LOCAL`` and ``PROBELINES`` strategies.
            May be ``None`` for ``HADAMARD`` and ``MINIMAL`` when
            *active_qubits* is provided explicitly.
        active_qubits: Subset of qubits to include.  If ``None``, labels are
            derived from *qpu_topology* (which must then be provided).
        strategy: Twirling strategy name. Matching is case-insensitive and ignores
            surrounding whitespace.
        max_twirling: Maximum number of strings for strategies that can be expanded.

    Returns:
        A list of dictionaries mapping each active qubit label to ``"I"`` or ``"X"``.

    Raises:
        ValueError: If the strategy is unknown, size constraints are not
            satisfied, or required arguments are missing.

    """
    bit_to_pauli = {"0": "I", "1": "X"}
    normalized_strategy = strategy.strip().upper()

    if active_qubits is None:
        if qpu_topology is None:
            raise ValueError("Either 'active_qubits' or 'qpu_topology' must be provided.")
        active_qubits = qpu_topology.get_qubit_labels()

    if normalized_strategy in ("LOCAL", "PROBELINES") and qpu_topology is None:
        raise ValueError(
            f"The '{normalized_strategy}' strategy requires a QPU topology. "
            "Provide 'qpu_topology' or use a strategy that does not need it "
            "(e.g. 'HADAMARD' or 'MINIMAL')."
        )

    if normalized_strategy == "HADAMARD":
        hadamard_bits = HadamardGenerator(len(active_qubits))

        return [{qubit: bit_to_pauli[str(bit)] for qubit, bit in zip(active_qubits, bits)} for bits in hadamard_bits]

    if normalized_strategy == "MINIMAL":
        if max_twirling and max_twirling < 2:  # noqa: PLR2004
            raise ValueError(
                f"Minimal twirling strategy requires 2 twirling strings, but max_twirling is {max_twirling}"
            )
        return [
            {qubit: bit_to_pauli[str(j % 2)] for j, qubit in enumerate(active_qubits)},
            {qubit: bit_to_pauli[str((j + 1) % 2)] for j, qubit in enumerate(active_qubits)},
        ]

    if normalized_strategy == "LOCAL":
        if max_twirling and max_twirling < 4:  # noqa: PLR2004
            raise ValueError(f"Local twirling strategy requires 4 twirling strings, but max_twirling is {max_twirling}")
        if qpu_topology is None:
            raise ValueError("qpu_topology is required for the LOCAL twirling strategy.")
        return _generate_local_strings(qpu_topology, active_qubits)

    if normalized_strategy == "PROBELINES":
        if max_twirling and max_twirling < 4:  # noqa: PLR2004
            raise ValueError(
                f"Probelines twirling strategy requires at least 4 twirling strings, but max_twirling is {max_twirling}"
            )
        if qpu_topology is None:
            raise ValueError("qpu_topology is required for the PROBELINES twirling strategy.")
        return _generate_probeline_strings(qpu_topology, active_qubits, max_twirling or 4)

    raise ValueError(
        f"Unknown twirling strategy {strategy}. Supported strategies are "
        '"HADAMARD", "MINIMAL", "LOCAL" and "PROBELINES".'
    )


def _generate_balanced_pairing_string(existing_string: str) -> str:
    """Return a 4-char I/X string such that pairs (new[i], existing[i]) cover II, XI, IX, XX."""
    return _find_balanced_pairing_string(existing_string)


def _pairs_cover_all_cases(new_string: str, existing_string: str) -> bool:
    """Check that (new[i], existing[i]) spans II, XI, IX, XX exactly once each for i in [0..3]."""
    pairs = {(new_string[i], existing_string[i]) for i in range(4)}
    return pairs == _ALL_PAIR_CASES


_VALID_PAULI_CHARS = {"I", "X"}
_ALL_PAIR_CASES = {("I", "I"), ("X", "I"), ("I", "X"), ("X", "X")}


def _validate_ix_string(string: str) -> None:
    """Validate that a string is exactly four characters from ``{I, X}``.

    Args:
        string: Candidate string.

    Raises:
        ValueError: If ``string`` is not a valid 4-character ``I/X`` string.

    """
    if len(string) != 4 or any(char not in _VALID_PAULI_CHARS for char in string):  # noqa: PLR2004
        raise ValueError(f"Invalid existing string: {string}")


def _find_balanced_pairing_string(*existing_strings: str) -> str:
    """Find a 4-character ``I/X`` string balanced against all provided strings.

    The returned string ``new`` satisfies:

    ``{(new[i], existing[i]) for i in [0..3]} == {II, XI, IX, XX}``

    for every ``existing`` string in ``existing_strings``.

    Args:
        *existing_strings: One or more existing 4-character ``I/X`` strings.

    Returns:
        A valid 4-character ``I/X`` string balanced against all inputs.

    Raises:
        ValueError: If any input is invalid or no balanced candidate exists.

    """
    for string in existing_strings:
        _validate_ix_string(string)

    for mask in range(16):
        candidate = "".join("X" if (mask >> index) & 1 else "I" for index in range(4))
        if all(_pairs_cover_all_cases(candidate, string) for string in existing_strings):
            return candidate

    raise ValueError(
        "Cannot construct a 4-char I/X string that covers II, XI, IX, XX against all provided strings. "
        f"Got existing_strings={existing_strings}."
    )


def _generate_balanced_pairing_string_two_existing(existing_string_1: str, existing_string_2: str) -> str:
    """Return a 4-char I/X string balanced against both existing strings.

    The returned string `new` satisfies:
    - {(new[i], existing_string_1[i])} == {II, XI, IX, XX}
    - {(new[i], existing_string_2[i])} == {II, XI, IX, XX}
    """
    return _find_balanced_pairing_string(existing_string_1, existing_string_2)


def _add_mask(rot_string: dict[str, str], had_string: list[int], line: list[str]) -> dict[str, str]:
    """Apply a binary Hadamard mask to a twirling string on selected qubits.

    For each qubit in ``line`` with corresponding mask bit ``1``, the operator is
    flipped (``I <-> X``). Bits equal to ``0`` leave the value unchanged.

    Args:
        rot_string: Input twirling dictionary.
        had_string: Mask bits aligned with ``line``.
        line: Qubit labels to which the mask applies.

    Returns:
        A copied and masked twirling dictionary.

    """
    masked_rot_string = rot_string.copy()
    for qubit, had in zip(line, had_string):
        if had == 1:
            if masked_rot_string[qubit] == "I":
                masked_rot_string[qubit] = "X"
            elif masked_rot_string[qubit] == "X":
                masked_rot_string[qubit] = "I"
    return masked_rot_string


def _generate_probeline_strings(
    qpu_topology: QPUTopology,
    active_qubits: list[str],
    max_twirling: int,
) -> list[dict[str, str]]:
    """Generate PROBELINES twirling strings from LOCAL strings.

    The method first builds the 4 LOCAL strings, replicates them to
    ``max_twirling``, then applies control-line Hadamard masks blockwise on groups
    of four strings. Applying the same mask across each 4-string block preserves
    LOCAL nearest-neighbor balancing while extending balancing to control-line pairs.

    Args:
        qpu_topology: Topology with control-line definitions.
        active_qubits: Active qubits included in the twirling strings.
        max_twirling: Total number of twirling strings to return.

    Returns:
        A list of ``max_twirling`` twirling strings.

    """
    local_strings = _generate_local_strings(qpu_topology, active_qubits)
    all_rot_strings = local_strings.copy()

    active_subset_control_lines = [
        [qubit for qubit in line if qubit in active_qubits] for line in qpu_topology.control_lines
    ]

    # Find the longest control line among the active qubits to determine how many unique Hadamard masks we need
    max_line_length = max(len(line) for line in active_subset_control_lines)
    max_hadamard_masks = len(list(HadamardGenerator(max_line_length)))

    if max_twirling < 4 * max_hadamard_masks:
        # Warn the user that the number of twirling strings may not be sufficient
        # to fully balance all control line pairs
        print(
            f"Warning: The number of twirling strings ({max_twirling}) may not be sufficient "
            f"to fully balance all control line pairs. "
            f"Consider increasing max_twirling to {4 * max_hadamard_masks}."
        )
        num_twirling = max_twirling
    else:
        num_twirling = 4 * max_hadamard_masks

    for _ in range(num_twirling // 4 - 1):
        all_rot_strings.extend(local_strings)

    for line in active_subset_control_lines:
        if len(line) < 2:  # noqa: PLR2004
            continue
        had_strings = [list(bits) for bits in HadamardGenerator(len(line))]

        for rep in range(num_twirling // 4):
            had_string = had_strings[rep % len(had_strings)]
            for index in range(4):
                all_rot_strings[4 * rep + index] = _add_mask(
                    all_rot_strings[4 * rep + index],
                    had_string,
                    line,
                )

    return all_rot_strings


def _generate_local_strings(qpu_topology: QPUTopology, active_qubits: list[str]) -> list[dict[str, str]]:
    """Generate 4 LOCAL twirling strings balanced on topology neighbor pairs.

    The algorithm processes active qubits in natural qubit-number order. For each
    qubit, it ensures balanced pair coverage against one or two already-processed
    neighbors using ``_find_balanced_pairing_string``.

    Args:
        qpu_topology: Topology object providing nearest-neighbor relationships.
        active_qubits: Active qubit labels.

    Returns:
        Four twirling strings represented as dictionaries.

    Raises:
        ValueError: If a qubit has more than two already-processed neighbors.

    """
    list_of_rot_strings: list[dict[str, str]] = [{}, {}, {}, {}]

    sorted_active_qubits = QPUTopology.sort_qubit_labels(active_qubits)
    for q, qubit in enumerate(sorted_active_qubits):
        neighbors = [n for n in qpu_topology.get_neighbors(qubit) if n in sorted_active_qubits[:q]]

        if len(neighbors) == 0:
            string = "IIXX"
            for j, char in enumerate(string):
                list_of_rot_strings[j][qubit] = char

        elif len(neighbors) == 1:
            neighbor = neighbors[0]
            existing_string = "".join(list_of_rot_strings[j][neighbor] for j in range(4))
            new_string = _generate_balanced_pairing_string(existing_string)
            for j, char in enumerate(new_string):
                list_of_rot_strings[j][qubit] = char

        elif len(neighbors) == 2:  # noqa: PLR2004
            neighbor1, neighbor2 = neighbors
            existing_string1 = "".join(list_of_rot_strings[j][neighbor1] for j in range(4))
            existing_string2 = "".join(list_of_rot_strings[j][neighbor2] for j in range(4))
            new_string = _generate_balanced_pairing_string_two_existing(existing_string1, existing_string2)
            for j, char in enumerate(new_string):
                list_of_rot_strings[j][qubit] = char

        else:
            raise ValueError(
                f"Qubit {qubit} has more than 2 (already considered) neighbors "
                "in the topology, which is not supported by the LOCAL twirling strategy."
            )

    return list_of_rot_strings
