# Copyright 2025 IQM
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

"""Chip layout class."""

from __future__ import annotations

from functools import cached_property

import numpy as np

from exa.common.errors.exa_error import ExaError


class ChipLayout:
    """The chip layout contains the components and their 2D cartesian coordinates."""

    def __init__(
        self,
        qubits: dict[str, tuple[float, float]],
        couplers: dict[str, tuple[float, float]],
        computational_resonators: dict[str, tuple[float, float]],
    ) -> None:
        self._qubits = list(qubits)
        self._couplers = list(couplers)
        self._computational_resonators = list(computational_resonators)
        self._coordinates = {
            comp: (x, y) for comp, (x, y) in [*qubits.items(), *couplers.items(), *computational_resonators.items()]
        }

    @classmethod
    def from_chip_design_record(cls, record: dict) -> ChipLayout:
        """Construct the chip layout from a raw chip design record.

        Args:
            record: The chip design record as returned by station control.

        Returns:
            The corresponding chip layout.

        """
        qubits = record["content"]["components"].get("qubit", [])
        couplers = record["content"]["components"].get("tunable_coupler", [])
        comprs = record["content"]["components"].get("computational_resonator", [])
        if all("locations" in component for component in [*qubits, *couplers, *comprs]):
            return cls(
                qubits={
                    qubit["name"]: (qubit["locations"]["metro"]["x"], qubit["locations"]["metro"]["y"])
                    for qubit in qubits
                },
                couplers={
                    coupler["name"]: (coupler["locations"]["metro"]["x"], coupler["locations"]["metro"]["y"])
                    for coupler in couplers
                },
                computational_resonators={
                    compr["name"]: (compr["locations"]["metro"]["x"], compr["locations"]["metro"]["y"])
                    for compr in comprs
                },
            )
        raise ExaError("Chip design record is missing locations.")

    def normalize_coordinates(self, scale: float) -> None:
        self._coordinates = {comp: (xx * scale, yy * scale) for comp, (xx, yy) in self._coordinates.items()}

    def mirror_yaxis(self) -> None:
        self._coordinates = {comp: (xx, -yy) for comp, (xx, yy) in self._coordinates.items()}

    def rotate_layout(self) -> None:
        self._coordinates = {
            comp: ((xx + yy) / np.sqrt(2), (-xx + yy) / np.sqrt(2)) for comp, (xx, yy) in self._coordinates.items()
        }

    def move_origin(self) -> None:
        x_min, y_min = (
            min([xx for comp, (xx, yy) in self._coordinates.items()]),
            min([yy for comp, (xx, yy) in self._coordinates.items()]),
        )
        self._coordinates = {comp: (xx - x_min, yy - y_min) for comp, (xx, yy) in self._coordinates.items()}

    @property
    def qubits(self) -> list[str]:
        return self._qubits

    @property
    def couplers(self) -> list[str]:
        return self._couplers

    @property
    def computational_resonators(self) -> list[str]:
        return self._computational_resonators

    @cached_property
    def components(self) -> list[str]:
        return [*self._qubits, *self._couplers, *self._computational_resonators]

    def get_coordinates(self, component: str) -> tuple[float, float]:
        """Get the coordinates for the given component.

        Args:
            component: The name of the component.

        Returns:
            The 2D cartesian coordinates.

        """
        if component not in self.components:
            raise ValueError(f"Component {component} not in chip layout.")
        return self._coordinates[component]

    def get_all_qubit_coordinates(self) -> dict[str, tuple[float, float]]:
        """Get the coordinates for all qubits."""
        return {qubit: self._coordinates[qubit] for qubit in self._qubits}
