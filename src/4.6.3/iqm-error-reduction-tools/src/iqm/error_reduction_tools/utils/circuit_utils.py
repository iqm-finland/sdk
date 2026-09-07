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

"""Utilities for working with IQM quantum circuits."""

import copy

from iqm.pulse import Circuit, CircuitOperation


class TwirledCircuit:
    """A quantum circuit wrapper that supports readout twirling and gate randomization.

    This class serves as a mutable working representation for building and manipulating
    quantum circuits. It tracks single-qubit gates, measurements, and
    readout twirling state. Use ``to_circuit(name)`` to convert to the immutable
    :class:`~iqm.pulse.Circuit`.

    .. note::

        We intentionally do **not** store a :class:`~iqm.pulse.Circuit` internally.
        ``Circuit.instructions`` is an immutable ``tuple``, which would be rebuilt on
        every ``append_operation`` call. Instead, ``operations`` is kept as a mutable
        ``list[CircuitOperation]`` throughout circuit construction, and converted to
        ``Circuit`` on demand via ``to_circuit()``.

    Attributes:
        operations: List of operations in the circuit.
        rot_dict: Dictionary for readout twirling rotations, if applied.
        sqg_counter: Counter for single-qubit gates per qubit.
        measured_qubits: List of qubits that are measured.
        active_qubits: List of all qubits involved in the circuit.

    """

    def __init__(self, operations: list[CircuitOperation]) -> None:
        self.operations = [copy.deepcopy(op) for op in operations]
        self.basic_analysis()

        # When initializing, we assume no readout twirling has been applied
        self.rot_dict: dict[str, str] | None = None

    def basic_analysis(self) -> None:
        """Perform basic analysis on quantum circuit operations.

        This method analyzes the circuit operations to extract information about
        single-qubit gates, measurements, and active qubits. It also validates
        that no mid-circuit measurements are present.

        Raises:
            ValueError: If mid-circuit measurements are detected (when any qubit is measured more than once).

        .. note::

            This method populates the following instance attributes:

                * ``sqg_counter``: Dictionary mapping qubit identifiers to the count of 'prx'
                  operations performed on each qubit.
                * ``measured_qubits``: List of qubit identifiers that have measurement operations.
                * ``active_qubits``: List of all qubit identifiers that appear in any operation.

            The method iterates through all operations in the circuit and:

                1. Counts 'prx' (parametric rotation) operations per qubit
                2. Counts measurement operations per qubit
                3. Collects all qubits involved in any operation
                4. Validates that no qubit is measured more than once

        """
        self.sqg_counter: dict[str, int] = {}
        measurement_counter: dict[str, int] = {}
        self.measurement_key_to_qubit: dict[str, str] = {}  # Maps measurement keys to qubit identifiers
        active_qubits_set = set()

        for operation in self.operations:
            locus = operation.locus

            if operation.name == "prx":
                self.sqg_counter[locus[0]] = self.sqg_counter.get(locus[0], 0) + 1
            elif operation.name == "measure":
                for loc in locus:
                    measurement_counter[loc] = measurement_counter.get(loc, 0) + 1
                if len(locus) > 1:
                    raise NotImplementedError(
                        "Measurement operations defined on more than one qubits are not supported yet."
                    )
                self.measurement_key_to_qubit[operation.args["key"]] = locus[0]
            active_qubits_set.update(set(locus))

        # Check for mid-circuit measurements
        if any(count > 1 for count in measurement_counter.values()):
            raise ValueError("Mid-circuit measurements are not supported yet.")

        self.measured_qubits = list(measurement_counter.keys())
        self.active_qubits = list(active_qubits_set)

    def append_operation(self, operation: CircuitOperation) -> None:
        """Append a deep copy of the operation to the circuit and update basic analysis.

        A deep copy is made to ensure that no external references to the operation
        are retained, preventing shared-state bugs when the same source circuit
        is randomized multiple times.
        """
        self.operations.append(copy.deepcopy(operation))

        # Update the basic analysis
        self.basic_analysis()

    def get_rot_char_per_meas_key(self) -> dict[str, str]:
        """Get rotation character per measurement key due to readout twirling.

        Returns:
            Dictionary mapping measurement keys to rotation characters ('I' or 'X').

        """
        rot_for_key_dict: dict[str, str] = {key: "I" for key in self.measurement_key_to_qubit}

        if self.rot_dict:
            rot_for_key_dict = {
                key: self.rot_dict.get(qubit_name, "I") for key, qubit_name in self.measurement_key_to_qubit.items()
            }

        return rot_for_key_dict

    def to_circuit(self, name: str) -> Circuit:
        """Convert to an immutable :class:`~iqm.pulse.Circuit`.

        Args:
            name: Name to assign to the circuit.

        Returns:
            A :class:`~iqm.pulse.Circuit` with the current operations as instructions.

        """
        return Circuit(name=name, instructions=tuple(self.operations))
