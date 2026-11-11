# Copyright 2020–2021 Cirq on IQM developers
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
"""Cirq adapter for IQM's quantum computers."""

__all__ = [
    "Adonis",
    "Aphrodite",
    "Apollo",
    "CircuitDiagramInfo",
    "CircuitDiagramInfoArgs",
    "DropRZBeforeMeasurement",
    "EigenGate",
    "IQMDevice",
    "IQMDeviceMetadata",
    "IQMMoveGate",
    "IQMResult",
    "IQMSampler",
    "MergeOneParameterGroupGates",
    "OperationNotSupportedError",
    "QasmUGate",
    "ResultMetadata",
    "circuit_from_qasm",
    "circuit_operation_to_operation",
    "deserialize_circuit",
    "operation_to_circuit_operation",
    "serialize_circuit",
    "simplify_circuit",
    "transpile_insert_moves_into_circuit",
]
from .devices import Adonis, Aphrodite, Apollo, IQMDevice, IQMDeviceMetadata
from .extended_qasm_parser import QasmUGate, circuit_from_qasm
from .iqm_gates import CircuitDiagramInfo, CircuitDiagramInfoArgs, EigenGate, IQMMoveGate
from .iqm_sampler import IQMResult, IQMSampler, ResultMetadata
from .optimizers import DropRZBeforeMeasurement, MergeOneParameterGroupGates, simplify_circuit
from .serialize import (
    OperationNotSupportedError,
    circuit_operation_to_operation,
    deserialize_circuit,
    operation_to_circuit_operation,
    serialize_circuit,
)
from .transpiler import transpile_insert_moves_into_circuit
