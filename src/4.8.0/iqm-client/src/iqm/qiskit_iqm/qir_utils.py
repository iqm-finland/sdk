# Copyright 2024-2026 IQM
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
"""Utilities for working with QIR modules."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import Layout

from iqm.station_control.interface.models import QubitMapping


def generate_qiskit_qir_qubit_mapping(qiskit_circuit: QuantumCircuit, qiskit_backend: BackendV2) -> QubitMapping:
    """Generate qubit mapping for QIR.

    ``qiskit-qir`` has a bug, which causes qubit pointers to not be generated correctly
    according to the final_layout. So we replicate this logic here and generate a new mapping.
    Then we assign ``qiskit-qir`` index to the qiskit logic qubit idx.

    Args:
        qiskit_circuit: The Qiskit circuit to generate the mapping for.
        qiskit_backend: The Qiskit backend object to be used for qubit name generation.

    Returns:
        A dictionary mapping Qiskit qubit indices to QIR qubit pointers.

    """
    # Find the layout
    if qiskit_circuit.layout is None or qiskit_circuit.layout.final_layout is None:
        # generate trivial layout
        final_layout = Layout.generate_trivial_layout(*qiskit_circuit.qubits)
    else:
        # use the existing layout
        final_layout = qiskit_circuit.layout.final_layout

    # For simplicity, reverse the mapping
    layout_reverse_mapping = {bit: idx for idx, bit in final_layout.get_physical_bits().items()}
    qiskit_qir_mapping: dict[int, int] = {}

    # Replicate qiskit-qir logic for defining qubit pointer indices
    for register in qiskit_circuit.qregs:
        qiskit_qir_mapping.update(
            {layout_reverse_mapping[bit]: n + len(qiskit_qir_mapping) for n, bit in enumerate(register)}
        )

    # In the generated QIR qubit pointers will use qiskit-qir qubit labels,
    # but we already know how to map them to IQM physical qubits, through qiskit logical qubit indices.
    return {
        str(qiskit_qir_mapping[i]): str(qiskit_backend.index_to_qubit_name(i)) for i in range(qiskit_circuit.num_qubits)
    }
