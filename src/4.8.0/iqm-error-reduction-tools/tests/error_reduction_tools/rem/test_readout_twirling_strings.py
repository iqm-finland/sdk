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

"""Integration tests for readout twirling string generation strategies."""

from collections import Counter
from itertools import combinations

from iqm.error_reduction_tools.readout_characterization.topologies import QPUTopology
from iqm.error_reduction_tools.utils.readout_twirling_strings import (
    generate_rot_strings,
)

# Self-contained test topology used as a fixture for algorithm correctness tests.
# Two 5-qubit chains (QB1-5, QB6-10) with two cross-links, and two 5-qubit control lines.
# HadamardGenerator(5) yields 8 strings so max_twirling=4*8=32 fully exercises PROBELINES.
_TEST_TOPOLOGY = QPUTopology(
    name="test",
    num_qubits=10,
    positions={},
    couplers=[
        ("QB1", "QB2"),
        ("QB2", "QB3"),
        ("QB3", "QB4"),
        ("QB4", "QB5"),
        ("QB6", "QB7"),
        ("QB7", "QB8"),
        ("QB8", "QB9"),
        ("QB9", "QB10"),
        ("QB2", "QB7"),
        ("QB4", "QB9"),  # cross-links giving 2-predecessor qubits
    ],
    control_lines=[
        ("QB1", "QB2", "QB3", "QB4", "QB5"),
        ("QB6", "QB7", "QB8", "QB9", "QB10"),
    ],
)


def qb_num(qubit: str) -> int:
    return int("".join(char for char in qubit if char.isdigit()))


def canon_pair(qubit_a: str, qubit_b: str) -> tuple[str, str]:
    return tuple(sorted((qubit_a, qubit_b), key=qb_num))


def pair_combo_counts(strings: list[dict[str, str]], pair: tuple[str, str]) -> Counter:
    qubit_1, qubit_2 = pair
    combo_count = Counter()
    for string in strings:
        combo_count[string[qubit_1] + string[qubit_2]] += 1
    return combo_count


def build_neighbor_pairs(active_qubits: list[str]) -> set[tuple[str, str]]:
    active_set = set(active_qubits)
    return {
        canon_pair(qubit_a, qubit_b)
        for qubit_a, qubit_b in _TEST_TOPOLOGY.couplers
        if qubit_a in active_set and qubit_b in active_set
    }


def build_probeline_pairs(active_qubits: list[str]) -> set[tuple[str, str]]:
    active_set = set(active_qubits)
    probeline_pairs: set[tuple[str, str]] = set()

    for line in _TEST_TOPOLOGY.control_lines:
        line_active = [qubit for qubit in line if qubit in active_set]
        for qubit_a, qubit_b in combinations(line_active, 2):
            probeline_pairs.add(canon_pair(qubit_a, qubit_b))

    return probeline_pairs


def assert_balanced_pair_set(strings: list[dict[str, str]], pair_set: set[tuple[str, str]]) -> None:
    for pair in sorted(pair_set, key=lambda item: (qb_num(item[0]), qb_num(item[1]))):
        counts = pair_combo_counts(strings, pair)
        assert counts["II"] == counts["IX"] == counts["XI"] == counts["XX"], (
            f"Unbalanced pair {pair} with counts {dict(counts)}"
        )


def test_probeline_strategy_balances_neighbor_and_control_line_pairs() -> None:
    active_qubits = [f"QB{index + 1}" for index in range(10)]
    max_twirling = 4 * 8

    rot_strings = generate_rot_strings(
        _TEST_TOPOLOGY,
        active_qubits=active_qubits,
        strategy="PROBELINES",
        max_twirling=max_twirling,
    )

    assert len(rot_strings) == max_twirling

    neighbor_pairs = build_neighbor_pairs(active_qubits)
    probeline_pairs = build_probeline_pairs(active_qubits)

    assert neighbor_pairs
    assert probeline_pairs

    assert_balanced_pair_set(rot_strings, neighbor_pairs)
    assert_balanced_pair_set(rot_strings, probeline_pairs)


def test_probeline_strategy_multiple_of_4_validation() -> None:
    active_qubits = [f"QB{index + 1}" for index in range(10)]

    assert (
        len(
            generate_rot_strings(
                _TEST_TOPOLOGY,
                active_qubits=active_qubits,
                strategy="PROBELINES",
                max_twirling=10,
            )
        )
        % 4
        == 0
    )


def test_strategy_normalization_for_generate_rot_strings() -> None:
    active_qubits = [f"QB{index + 1}" for index in range(8)]

    normalized = generate_rot_strings(
        _TEST_TOPOLOGY,
        active_qubits=active_qubits,
        strategy="PROBELINES",
        max_twirling=8,
    )
    spaced_and_lower = generate_rot_strings(
        _TEST_TOPOLOGY,
        active_qubits=active_qubits,
        strategy="  probelines  ",
        max_twirling=8,
    )

    assert normalized == spaced_and_lower
