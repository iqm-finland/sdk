# Copyright 2026 IQM
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
"""IQM Pulse delay operation for quantum circuits."""

from __future__ import annotations

from iqm.pulse.quantum_ops import QuantumOp

from ..pulse_operation import quantum_op_to_qrisp_func

# TODO duplicates the standard "delay" operation in iqm-pulse (with sligtly different properties), is this necessary?
delay_quantum_op = QuantumOp(
    name="delay",
    params={
        "duration": (float,),
    },
)

delay = quantum_op_to_qrisp_func(delay_quantum_op)

delay.__doc__ = """
Applies a ``delay`` operation to the specified qubits in a quantum circuit.

Args:
    qubits (QuantumVariable or Qubit): The qubits or QuantumVariables to apply the delay to.
    duration (float): The duration of the delay in seconds.
"""
