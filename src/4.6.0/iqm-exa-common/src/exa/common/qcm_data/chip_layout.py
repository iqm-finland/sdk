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

from collections.abc import Iterable
from functools import cached_property

import numpy as np

from exa.common.errors.iqm_error import NotFoundError, ValidationError


class ChipLayout:
    """The chip layout contains the components, their 2D cartesian coordinates, and their angles.

    The subset of qubits without any probe lines, which appears in certain constellation-type chips,
    are stored in the additional attribute qubits_without_probeline.

    The angle is assumed to lie between the positive x-axis and the orientation vector of the component
    and measured in degrees.

    Args:
        qubits: Dictionary of qubit names mapped to their 2d coordinates
        couplers: Dictionary of coupler names mapped to their 2d coordinates
        computational_resonators: Dictionary of computational resonators names mapped to their 2d coordinates
        qubits_without_probeline: Iterable of qubits without a probeline
        angles: Dictionary of component names mapped to their 2d coordinates

    """

    def __init__(
        self,
        qubits: dict[str, tuple[float, float]],
        couplers: dict[str, tuple[float, float]],
        computational_resonators: dict[str, tuple[float, float]],
        qubits_without_probeline: Iterable[str] | None = None,
        angles: dict[str, float] | None = None,
    ) -> None:
        self._qubits = list(qubits)
        self._couplers = list(couplers)
        self._computational_resonators = list(computational_resonators)
        self._coordinates = {
            comp: (x, y) for comp, (x, y) in [*qubits.items(), *couplers.items(), *computational_resonators.items()]
        }
        self._qubits_without_probeline: frozenset[str] = frozenset(qubits_without_probeline or [])
        self._angles = angles or dict.fromkeys(self._coordinates.keys(), 0)

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
        # dict of probe lines which contains their names and the components they connect to
        probe_lines = record["content"]["components"].get("probe_line", [])
        # set of probed components
        probed = {connection for pl in probe_lines for connection in pl.get("connections", [])}
        # subset of qubits not found in the set of probed components
        qubits_without_probeline = frozenset(q["name"] for q in qubits if q["name"] not in probed)

        if all("locations" in component for component in [*qubits, *couplers, *comprs]):
            return cls(
                qubits={
                    qubit["name"]: (
                        qubit["locations"]["metro"]["x"],
                        qubit["locations"]["metro"]["y"],
                    )
                    for qubit in qubits
                },
                couplers={
                    coupler["name"]: (
                        coupler["locations"]["metro"]["x"],
                        coupler["locations"]["metro"]["y"],
                    )
                    for coupler in couplers
                },
                computational_resonators={
                    compr["name"]: (
                        compr["locations"]["metro"]["x"],
                        compr["locations"]["metro"]["y"],
                    )
                    for compr in comprs
                },
                qubits_without_probeline=qubits_without_probeline,
                angles={
                    component["name"]: component["locations"]["metro"].get("angle", 0)
                    for component in [*qubits, *couplers, *comprs]
                },
            )
        raise ValidationError("Chip design record is missing locations.")

    def normalize_coordinates(self, min_spacing: float) -> None:
        """Rescale component coordinates

        Rescale all component coordinates so that the minimum distance between 2 components is min_spacing
        """
        keys = list(self._coordinates.keys())
        coords = np.array(list(self._coordinates.values()), dtype=float)

        if len(coords) < 2:
            return

        deltas = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.hypot(deltas[..., 0], deltas[..., 1])
        np.fill_diagonal(distances, np.inf)
        min_distance = float(distances.min())

        if min_distance == 0:
            raise ValueError("There are overlapping components in the chip layout.")

        scale = min_spacing / min_distance
        scaled_points = coords * scale
        self._coordinates = dict(zip(keys, map(tuple, scaled_points)))

    def mirror_yaxis(self) -> None:
        self._coordinates = {comp: (xx, -yy) for comp, (xx, yy) in self._coordinates.items()}
        self._angles = {comp: -angle for comp, angle in self._angles.items()}

    def rotate_layout(self, angle: float = 45) -> None:
        """Clockwise rotation of coordinates."""
        self._coordinates = {
            comp: (
                xx * np.cos(np.pi * angle / 180) + yy * np.sin(np.pi * angle / 180),
                -xx * np.sin(np.pi * angle / 180) + yy * np.cos(np.pi * angle / 180),
            )
            for comp, (xx, yy) in self._coordinates.items()
        }
        self._angles = {key: value - angle for key, value in self._angles.items()}

    def move_origin(self) -> None:
        """Changes the origin of the Cartesian coordinate system such that all elements move into the first quadrant.

        If there is a component with coordinates `(x_min, y_min)` it will move to `(0, 0)`.
        """
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

    @property
    def qubits_without_probeline(self) -> frozenset[str]:
        return self._qubits_without_probeline

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
            raise NotFoundError(f"Component {component} not in chip layout.")
        return self._coordinates[component]

    def get_all_qubit_coordinates(self) -> dict[str, tuple[float, float]]:
        """Get the coordinates for all qubits."""
        return {qubit: self._coordinates[qubit] for qubit in self._qubits}

    def get_angle(self, component: str) -> float:
        """Get the angle for the given component.

        Args:
            component: The name of the component.

        Returns:
            The angle of the component.

        """
        if component not in self.components:
            raise NotFoundError(f"Component {component} not in chip layout.")
        return self._angles[component]

    def get_all_qubit_angles(self) -> dict[str, float]:
        """Get the angles for all qubits."""
        return {qubit: self._angles[qubit] for qubit in self._qubits}
