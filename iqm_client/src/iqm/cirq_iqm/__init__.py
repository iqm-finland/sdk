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

from .devices import *  # noqa: F403*
from .extended_qasm_parser import circuit_from_qasm
from .iqm_gates import *  # noqa: F403
from .transpiler import transpile_insert_moves_into_circuit
