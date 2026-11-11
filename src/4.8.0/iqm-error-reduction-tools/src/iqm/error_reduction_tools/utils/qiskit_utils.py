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

"""Utility functions for converting between Qiskit and IQM representations."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey, deserialize_instructions, serialize_instructions
from iqm.qrisp_iqm import qrisp_to_iqm_converter
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import QuantumRegister
from qiskit.transpiler.layout import Layout
from qrisp import QuantumCircuit as QrispQuantumCircuit  # type: ignore[import-untyped]

from iqm.pulse import Circuit as PulseCircuit
from iqm.pulse import CircuitOperation

from .circuit_utils import TwirledCircuit


def qubit_index_to_name_map(qc: QiskitQuantumCircuit, backend: IQMBackendBase | None = None) -> dict[int, str]:
    """Build the qubit index to qubit name mapping a qiskit circuit is translated with.

    Args:
        qc: The qiskit circuit whose qubit indices are to be named.
        backend: Optional IQM Qiskit backend, whose own naming is used when given.

    Returns:
        Mapping from qiskit qubit index to IQM qubit name.

    Raises:
        ValueError: If ``backend`` is neither ``None`` nor an :class:`IQMBackendBase`.

    """
    if backend and isinstance(backend, IQMBackendBase):
        return {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}
    if not backend:
        return {i: f"QB{i + 1}" for i in range(qc.num_qubits)}
    raise ValueError("backend must be an instance of IQMBackendBase or None.")


def _names_in_operations(operations: Sequence[CircuitOperation]) -> dict[int, str]:
    """Map the indices 0, 1, ... to the sorted qubit names of a circuit that has no qiskit indices.

    The mapping is only used to convert the circuit back to qiskit. Counts are keyed by the classical bits of the
    measurement keys, not by this mapping.
    """
    names = sorted({qubit for operation in operations for qubit in operation.locus})
    return dict(enumerate(names))


def canonicalize_circuit(
    circuit: Any, *, name: str = "circuit", backend: IQMBackendBase | None = None
) -> tuple[PulseCircuit, dict[int, str]]:
    """Convert any accepted circuit representation to the canonical :class:`iqm.pulse.Circuit`.

    Every function that accepts a circuit from a caller converts it with this function.

    .. warning::

        A :class:`~iqm.error_reduction_tools.utils.circuit_utils.TwirledCircuit` is accepted,
        but its ``rot_dict`` is dropped, because an :class:`iqm.pulse.Circuit` cannot store it.
        Keep a randomized circuit as a ``TwirledCircuit``, as
        :meth:`~iqm.error_reduction_tools.twirling.twirling_api.CircuitTwirler._convert_circuits` does.

    Args:
        circuit: An :class:`iqm.pulse.Circuit`, a
            :class:`~iqm.error_reduction_tools.utils.circuit_utils.TwirledCircuit`, a qiskit
            :class:`~qiskit.circuit.QuantumCircuit`, a transpiled :class:`qrisp.QuantumCircuit`, or any object
            exposing ``to_qiskit()``. ``to_qiskit()`` is called once and must return a qiskit circuit.
        name: Name for the circuit, used only when one has to be constructed.
        backend: IQM Qiskit backend, for qubit naming. With a backend, a Qrisp circuit is converted directly and
            its qubit ``i`` becomes the ``i``-th qubit of the backend architecture. Without one, it is converted
            through ``to_qiskit()``.

    Returns:
        The canonical circuit, and its qubit index to name mapping. For input that never had
        qiskit indices the mapping is invented, as described in :func:`_names_in_operations`.

    Raises:
        TypeError: If the circuit is of an unsupported type, or if its ``to_qiskit()`` does not
            return a qiskit :class:`~qiskit.circuit.QuantumCircuit`.

    """
    if isinstance(circuit, PulseCircuit):
        return circuit, _names_in_operations(circuit.instructions)
    if isinstance(circuit, TwirledCircuit):
        return circuit.to_circuit(name), _names_in_operations(circuit.operations)
    if isinstance(circuit, QiskitQuantumCircuit):
        return _from_qiskit_quantum_circuit(circuit, name=name, backend=backend)
    if isinstance(circuit, QrispQuantumCircuit) and backend is not None:
        qubits = backend.architecture.qubits
        converted = qrisp_to_iqm_converter(circuit, backend.architecture, circuit_name=name)
        # Rename the converter's keys cb_<i> to the qiskit form that iqm-client reads results with.
        num_clbits = len(circuit.clbits)
        instructions = tuple(
            replace(
                op,
                args=op.args
                | {"key": str(MeasurementKey("c", num_clbits, 0, int(op.args["key"].removeprefix("cb_"))))},
            )
            if op.name == "measure"
            else op
            for op in converted.instructions
        )
        return PulseCircuit(name=name, instructions=instructions), dict(enumerate(qubits[: len(circuit.qubits)]))
    if hasattr(circuit, "to_qiskit"):
        # Convert once. Recursing would not terminate for a mock, whose to_qiskit() returns another mock.
        converted = circuit.to_qiskit()
        if not isinstance(converted, QiskitQuantumCircuit):
            raise TypeError(
                f"{type(circuit).__name__}.to_qiskit() returned {type(converted).__name__}, "
                "not a qiskit QuantumCircuit."
            )
        return _from_qiskit_quantum_circuit(converted, name=name, backend=backend)
    raise TypeError(
        f"Unsupported circuit type: {type(circuit).__name__}. "
        "Expected TwirledCircuit, iqm.pulse.Circuit, qiskit QuantumCircuit, "
        "or any circuit with a to_qiskit() method (e.g. Qrisp QuantumCircuit)."
    )


def _from_qiskit_quantum_circuit(
    circuit: QiskitQuantumCircuit, *, name: str, backend: IQMBackendBase | None
) -> tuple[PulseCircuit, dict[int, str]]:
    """Serialize a qiskit circuit into canonical pulse form, with its qubit naming."""
    qubit_index_to_name = qubit_index_to_name_map(circuit, backend)
    operations = serialize_instructions(circuit=circuit, qubit_index_to_name=qubit_index_to_name)
    return PulseCircuit(name=name, instructions=tuple(operations)), qubit_index_to_name


def from_qiskit_to_iqm(
    qc: QiskitQuantumCircuit, backend: IQMBackendBase | None = None
) -> tuple[TwirledCircuit, dict[int, str]]:
    """Convert a Qiskit quantum circuit to IQM format.

    Args:
        qc: Qiskit quantum circuit to convert.
        backend: Optional IQM Qiskit backend for qubit naming.

    Returns:
        Converted ``qc``, qubit index to name mapping.

    Raises:
        ValueError: If ``backend`` is not None or an IQMBackend instance

    """
    qubit_index_to_name = qubit_index_to_name_map(qc, backend)

    return (
        TwirledCircuit(serialize_instructions(circuit=qc, qubit_index_to_name=qubit_index_to_name)),
        qubit_index_to_name,
    )


def from_iqm_to_qiskit(
    circuit: TwirledCircuit,
    qubit_index_to_name: dict[int, str],
    backend: IQMBackendBase | None = None,
) -> QiskitQuantumCircuit:
    """Convert an IQM quantum circuit to a Qiskit quantum circuit.

    Args:
        circuit: IQM twirled quantum circuit.
        qubit_index_to_name: Mapping from qubit indices to names.
        backend: Quantum computer to utilise.

    Returns:
        Reconstructed Qiskit quantum circuit.

    """
    if backend and isinstance(backend, IQMBackendBase):
        num_qubits = backend.num_qubits
    else:
        num_qubits = max(qubit_index_to_name.keys()) + 1

    layout = Layout()
    layout.add_register(QuantumRegister(num_qubits, "q"))

    qiskit_circuit = deserialize_instructions(
        circuit.operations,
        qubit_name_to_index={v: k for k, v in qubit_index_to_name.items()},
        layout=layout,
    )

    # Handling readout twirling information
    if circuit.rot_dict:
        list_meas_keys = [MeasurementKey.from_string(mk) for mk in circuit.get_rot_char_per_meas_key()]

        length = list_meas_keys[0].creg_len

        for meas_key in list_meas_keys:
            if meas_key.creg_idx >= 1:
                raise (
                    NotImplementedError(
                        "Conversion of circuits with multiple classical registers is not implemented yet."
                    )
                )

            if meas_key.creg_len != length:
                raise ValueError("All measurement keys must have the same classical register length.")

        rot_qiskit_string = ["I"] * length
        for meas_key in list_meas_keys:
            index = meas_key.clbit_idx
            rot_qiskit_string[index] = circuit.get_rot_char_per_meas_key()[str(meas_key)]

        qiskit_circuit.metadata["rot_string"] = "".join(rot_qiskit_string)

    return qiskit_circuit
