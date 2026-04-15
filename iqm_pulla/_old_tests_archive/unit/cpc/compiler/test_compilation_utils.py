# Copyright 2024-2025 IQM
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
"""Testing circuit compilation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from iqm.cpc.compiler.errors import UnknownHardwareComponentError, UnknownLogicalQubitError
from iqm.cpc.compiler.standard_stages import _map_components_in_instructions, _map_old_operation_arguments
from iqm.pulse import Circuit
from iqm.pulse import CircuitOperation as I


def test_map_components_unknown_physical_qubit():
    qubit_mapping = {"Alice": "Bob"}
    circuit = Circuit(name="xxx", instructions=(I(name="measure", locus=("Alice",), args={"key": "m"}),))
    with pytest.raises(UnknownHardwareComponentError, match="logical qubit 'Alice' maps to the physical qubit 'Bob'"):
        _map_components_in_instructions(qubit_mapping, circuit.instructions, {"QB1", "QB2", "QB3"})


@pytest.mark.parametrize(
    "logical_qubit",
    [
        "Charlie",  # unknown name
        "QB1",  # physical qubit names aren't accepted as such when a mapping is used
    ],
)
def test_map_components_unknown_logical_qubit(logical_qubit):
    qubit_mapping = {"Alice": "QB1", "Bob": "QB2"}
    circuit = Circuit(name="xxx", instructions=(I(name="measure", locus=(logical_qubit,), args={"key": "m"}),))
    with pytest.raises(
        UnknownLogicalQubitError, match=f"Logical qubit '{logical_qubit}' does not map to a physical qubit"
    ):
        _map_components_in_instructions(qubit_mapping, circuit.instructions, {"QB1", "QB2", "QB3"})


def test_qubit_unknown_qubit_no_qubit_mapping():
    circuit = Circuit(name="xxx", instructions=(I(name="measure", locus=("QB0",), args={"key": "m"}),))
    with pytest.raises(UnknownHardwareComponentError, match="Unknown physical qubit 'QB0'."):
        _map_components_in_instructions(None, circuit.instructions, {"QB1", "QB2", "QB3"})


@pytest.mark.parametrize("op_name", ["prx", "cc_prx"])
@pytest.mark.parametrize(
    "args,new_args",
    [
        ({"angle": 0.122}, {"angle": 0.122}),
        ({"angle": -2.51, "phase": 1.34}, {"angle": -2.51, "phase": 1.34}),
        ({"angle_t": 0.5}, {"angle": np.pi}),
        ({"angle_t": 0.25, "phase_t": -0.5}, {"angle": np.pi / 2, "phase": -np.pi}),
        ({"angle_t": 0.25, "phase_t": -0.5, "x": "zzz"}, {"angle": np.pi / 2, "phase": -np.pi, "x": "zzz"}),
    ],
)
def test_map_old_operation_arguments_prx(op_name, args, new_args):
    old = [I(name=op_name, locus=("QB1",), args=args.copy())]  # dict is changed by map_old_operation_arguments
    new = [I(name=op_name, locus=("QB1",), args=new_args)]
    _map_old_operation_arguments(old)
    assert old == pytest.approx(new)
