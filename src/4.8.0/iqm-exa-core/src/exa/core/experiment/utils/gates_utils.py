#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Utility functions for registration of custom gates for Experiment."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy import pi

from exa.common.data import DataType, Parameter
from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.gates.delay import Delay
from iqm.pulse.locus_mappings import (
    Locus,
    StationProperties,
    common_neighbor_components,
    one_qubit_drive_readout,
    two_qubit_cqc,
)
from iqm.pulse.quantum_ops import QuantumOp
from iqm.pulse.timebox import TimeBox


class MOVE_NCZ_MOVE_Composite(CompositeGate):  # noqa: N801
    """Composite MOVE-nCZ-MOVE gate.

    Has arity 2, locus consists of the CZ qubit and the MOVE qubit. The member gates are applied
    between the corresponding qubit and their common resonator (assumes there is only one).
    """

    registered_gates = ("cz", "move")
    locus_mapping_function = two_qubit_cqc
    routing_locus_function: Callable[[Locus, StationProperties], set[Locus]] = common_neighbor_components

    parameters = {"routing_locus": Parameter("routing_locus", "Central component", data_type=DataType.STRING)}

    def _call(self, n: int) -> TimeBox | list[TimeBox]:
        neighbors = [self.builder.chip_topology.get_neighbor_locus_components([c]) for c in self.locus]
        common_neighbors = set.intersection(*neighbors)

        if not common_neighbors:
            raise ValueError(f"The locus components {self.locus} have no common neighbors.")

        central_component = self.calibration_data["routing_locus"]

        if central_component not in common_neighbors:
            raise ValueError(
                f"The chosen central component {central_component} is not a common neighbor of {self.locus}."
            )

        cz_impl = self.build("cz", (self.locus[0], central_component))
        move_impl = self.build("move", (self.locus[1], central_component))

        cz_gate: TimeBox = cz_impl()  # type: ignore [assignment]
        move: TimeBox = move_impl()  # type: ignore [assignment]
        cz_calibration = cz_impl.calibration_data
        move_calibration = move_impl.calibration_data

        cz_duration = cz_calibration.get("duration")
        detuning = move_calibration.get("detuning")
        running_phase = 2 * pi * detuning * cz_duration  # type: ignore[operator]
        cz_central_component_phase = cz_calibration["rz"][central_component]

        # TODO: this is currently faster, but the phase tracking should be done in a general compiler pass framework
        # instead of using the priority calibration
        # TODO: Alternatively, we could call apply_move_gate_phase_corrections on the MOVE_NCZ_MOVE_Composite schedule
        # here. This would allow us to remove the priority_calibration parameter of CompositeGate.build, only used here.
        move_vz = move_calibration["rz"][self.locus[1]]
        total_phase = (move_vz + (running_phase + cz_central_component_phase) * n) % (2 * pi)
        move_calib_dynamic = {"rz": {self.locus[1]: total_phase}}

        move2 = self.build("move", (self.locus[1], central_component), priority_calibration=move_calib_dynamic)()

        boxes = move + [cz_gate] * int(n) + move2  # type: ignore[operator,list-item]
        return boxes


class MOVE_CZ_MOVE_Single(MOVE_NCZ_MOVE_Composite):  # noqa: N801
    """Composite MOVE-CZ-MOVE gate, exactly one CZ."""

    def _call(self):  # noqa: ANN202
        return super()._call(n=1)


class DoubleMOVE(CompositeGate):
    """Composite MOVE-MOVE identity gate.

    Transfers a state from the target qubit to an intermediate component and back. Composite gate has arity 1,
    with the locus consisting only of the target qubit. The intermediate component can be customized using the
    ``central_component`` composite gate calibration parameter.
    """

    routing_locus_function: Callable[[Locus, StationProperties], set[Locus]] = common_neighbor_components
    locus_mapping_function = one_qubit_drive_readout

    registered_gates = ("move",)

    parameters = {"routing_locus": Parameter("routing_locus", "Central component", data_type=DataType.STRING)}

    def _call(self) -> TimeBox | list[TimeBox]:
        move_qubit = self.locus[0]
        central_component = self.calibration_data["routing_locus"]

        move = self.builder.get_implementation("move", (move_qubit, central_component))()

        box = move + move
        return box


class ConstantDelay(Delay):
    """Applies a delay on the control channels of its locus components.

    This operation applies :class:`.Wait` instructions on all the control channels of all its locus
    components. This is the same implementation as :class:`Delay`, except that `duration` is a parameter instead of an
    input argument.
    """

    parameters: dict[str, Parameter] = {"duration": Parameter("", "Gate duration", "s")}  # type: ignore[assignment]

    def _call(self):  # noqa: ANN202
        duration = self.calibration_data["duration"]
        return super()._call(duration)


cz_n_quantum_op = QuantumOp(
    "cz_n",
    2,
    {"n": (int,)},
    implementations={
        "move_cz_move": MOVE_NCZ_MOVE_Composite,
    },
    symmetric=True,
    unitary=lambda n: np.array(np.diag([1, 1, 1, (-1) ** n])),
)

id_quantum_op = QuantumOp(
    "id",
    1,
    implementations={
        "move_move": DoubleMOVE,
        "wait": ConstantDelay,
    },
    unitary=lambda: np.eye(2),
    symmetric=True,
)
