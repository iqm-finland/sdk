#  *****************************************************************************
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
#  *****************************************************************************
"""Tracking plot markers."""


class PlotMarkers:
    """Track and output plot markers.

    Attributes:
        index: Index for the marker.
        markers: The Matplotlib markers.
        marker_count: The number of markers.

    """

    def __init__(self) -> None:
        """Create plot markers."""
        self.index = 0
        self.markers = ["o", "v", "P", "X", "D", "*", "s", "+", "1"]
        self.marker_count = len(self.markers)

    def get_marker(self) -> str:
        """Get the marker given the index.

        Returns:
            The type of marker in Matplotlib.

        """
        current_index = self.index
        #  the list is cyclical
        if current_index >= self.marker_count:
            current_index = 0
        current_marker = self.markers[current_index]
        self.index = current_index + 1
        return current_marker

    def reset_index(self) -> None:
        """Reset index to 0."""
        self.index = 0

    def increase_index(self, value: int = 1) -> None:
        """Increase the index by value.

        Args:
            value: Indicating how much the index should increase by.

        """
        self.index += value

    def decrease_index(self, value: int = 1) -> None:
        """Decrease the index by value.

        Args:
            value: Indicating how much the index should decrease by.

        """
        self.index -= value
