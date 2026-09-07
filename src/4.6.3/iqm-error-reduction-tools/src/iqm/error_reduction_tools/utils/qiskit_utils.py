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

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey, deserialize_instructions, serialize_instructions
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import QuantumRegister
from qiskit.transpiler.layout import Layout

from .circuit_utils import TwirledCircuit


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
    qubit_index_to_name: dict[int, str] = {}
    if backend and isinstance(backend, IQMBackendBase):
        qubit_index_to_name = {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}
    elif not backend:
        qubit_index_to_name = {i: f"QB{i + 1}" for i in range(qc.num_qubits)}
    else:
        raise ValueError("backend must be an instance of IQMBackendBase or None.")

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
        list_meas_keys = [MeasurementKey.from_string(mk) for mk in circuit.get_rot_char_per_meas_key().keys()]

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
