# Copyright (c) 2024-2025 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""This module contains the swap strategy from :cite:`Weidenfeller_2022`."""

from __future__ import annotations

from itertools import combinations
import math

from dimod import BinaryQuadraticModel
from iqm.qaoa.transpiler.quantum_hardware import CrystalQPUFromBackend, HardEdge, HardQubit
from iqm.qaoa.transpiler.routing import Mapping, Routing
import numpy as np


def sn_router(problem_bqm: BinaryQuadraticModel, qpu: CrystalQPUFromBackend) -> Routing:
    """The function that implements the 'swap network' swapping strategies.

    Implements approach from :cite:`Weidenfeller_2022` adapted for rectangular QPUs, not only square. If the input BQM
    is not all-to-all connected, dummy interactions (of strength 0) are added to make it all-to-all connected. Tries to
    find a sufficient rectangle in the provided Crystal QPU (square lattice topology required).

    Args:
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` description of the problem, necessary to create
            an instance of :class:`~iqm.qaoa.transpiler.routing.Routing`.
        qpu: The QPU, necessary to create an instance of :class:`~iqm.qaoa.transpiler.routing.Routing` and to get
            the topology of the qubits. The ``qpu`` also needs to contain the layout of the qubits with integer
            coordinates as an attribute.

    Returns:
        A :class:`~iqm.qaoa.transpiler.routing.Routing` object containing the routing created by using swap networks.

    Raises:
        TypeError: If the QPU's layout doesn't have integer coordinates. It is necessary to correctly identify where
            the swap and interaction gates need to be applied in the swap network approach.
        RuntimeError: If the algorithm doesn't find a sufficiently large rectangle of qubits in the QPU.

    """
    if not all(
        isinstance(coords, tuple) and all(isinstance(coord, int) for coord in coords)
        for coords in qpu.hardware_layout.values()
    ):
        raise TypeError("When using swap networks for transpilation, the QPU layout needs to use integer coordinates.")

    bqm_to_be_used = problem_bqm.copy()

    # Look for a rectangle whose "area" equals the number of problem variables.
    # Start from the rounded square root of the number of variables ...
    sqrt_n_q = int(math.sqrt(bqm_to_be_used.num_variables))
    # ... and count down.
    for i in range(sqrt_n_q, 0, -1):
        if bqm_to_be_used.num_variables % i == 0:
            # ``x`` and ``y`` could be the sides of the rectangle, because x * y = bqm_to_be_used.num_variables.
            x, y = i, bqm_to_be_used.num_variables // i
            # Find a rectangle with the sides ``x`` and ``y`` in the QPU graph.
            # ``rectangle_info[0]`` is the top-left corner of the rectangle.
            # ``rectangle_info[1]`` and ``rectangle_info[2]`` are the dimensions of the rectangle (it may be sideways).
            rectangle_info = _find_rectangular_subgraph(qpu, x, y)
            if rectangle_info is not None:
                corner_coords = qpu.hardware_layout[rectangle_info[0]]
                # Part of the QPU layout which the transpilation takes place on.
                relevant_layout = {
                    node: qpu.hardware_layout[node]
                    for node, value in qpu.hardware_layout.items()
                    if (corner_coords[0] <= value[0] < corner_coords[0] + rectangle_info[1])
                    and (corner_coords[1] <= value[1] < corner_coords[1] + rectangle_info[2])
                }
                break  # Break if we found the rectangle in the QPU, otherwise continue the for loop.
    else:  # If the for loop counted all the way down.
        raise RuntimeError("No rectangle of sufficient size can be embedded in the QPU.")

    # If an interaction is missing, we add an interaction of strength 0.
    # That way the problem formally has interactions between all pairs of variables.
    for v1, v2 in combinations(bqm_to_be_used.variables, 2):
        if (v1, v2) not in bqm_to_be_used.quadratic and (v2, v1) not in bqm_to_be_used.quadratic:
            bqm_to_be_used.add_quadratic(v1, v2, 0)

    # ``s`` refers to the four swap layers described in fig. 3 in the paper :cite:`Weidenfeller_2022`.
    s = _get_s(relevant_layout)  # type: ignore[arg-type]

    # The keys of ``relevant_layout`` correspond to the hardware qubit we use.
    # As initial mapping, we assign all the logical qubits to those.
    mapping = Mapping(qpu, bqm_to_be_used, {hq: i for i, hq in enumerate(relevant_layout.keys())})
    route = Routing(bqm_to_be_used, qpu, mapping)

    # Doing the routing (given ``s`` and ``rectangle_info``) is a bit long, so it's separatred in a helper function.
    _do_routing(route, s, rectangle_info[1], rectangle_info[2])

    return route


def _get_s(layout: dict[HardQubit, tuple[int, int]]) -> list[set[HardEdge]]:
    """Calculates the variable ``s``, the list of the four sets of (swap) gates to do in the routing.

    The variable ``s`` is defined in Figure 3 in :cite:`Weidenfeller_2022`. In the paper, ``s`` is described as
    containing swap layers, but here we interpret it simply as a set of edges along which gates can be executed
    in parallel.

    Args:
        layout: The layout of the qubits of the QPU. Their coordinates are needed to find neighbors in the four
            cardinal directions. It needs to describe a rectangular arrangement of qubits.

    Returns:
        The variable ``s`` as a list of four sets, each representing one swap layer.

    Raises:
        ValueError: If the provided layout doesn't describe a rectangular arrangement of qubits.

    """
    corner = (min(coords[0] for coords in layout.values()), min(coords[1] for coords in layout.values()))
    n_rows = max(coords[0] for coords in layout.values()) - corner[0] + 1
    n_columns = max(coords[1] for coords in layout.values()) - corner[1] + 1
    if n_rows * n_columns != len(layout):
        raise ValueError("The provided layout is not a rectangular arrangement of qubits.")

    # In this function we address qubits by their coordinates, so we need the layout dictionary inverted.
    layout_inverted = {coords: q for q, coords in layout.items()}

    # Swap layer S_0
    horizontal_odd: set[HardEdge] = set()
    # Swap layer S_1
    horizontal_even: set[HardEdge] = set()
    # Swap layer S_2
    vertical_even: set[HardEdge] = set()
    # Swap layer S_3
    vertical_odd: set[HardEdge] = set()
    for qubit, coords in layout.items():
        rel_coords = (coords[0] - corner[0], coords[1] - corner[1])  # Relative coordinates with respect to the corner.
        if rel_coords[0] % 2 == 0:
            if rel_coords[0] != n_rows - 1:
                # Connect ``qubit`` to the qubit under it, swap layer S_2
                vertical_even.add(HardEdge((qubit, layout_inverted[(coords[0] + 1, coords[1])])))
            if rel_coords[0] != 0:
                # Connect ``qubit`` to the qubit above it, swap layer S_3
                vertical_odd.add(HardEdge((qubit, layout_inverted[(coords[0] - 1, coords[1])])))
        if rel_coords[1] % 2 == 0:
            if rel_coords[1] + (-1) ** (rel_coords[0] % 2) not in {-1, n_columns}:
                # Connect ``qubit`` in the ``horizontal_even`` set of edges, swap layer S_1
                horizontal_even.add(
                    HardEdge((qubit, layout_inverted[(coords[0], coords[1] + (-1) ** (rel_coords[0] % 2))]))
                )
            if rel_coords[1] - (-1) ** (rel_coords[0] % 2) not in {-1, n_columns}:
                # Connect ``qubit`` in the ``horizontal_odd`` set of edges, swap layer S_0
                horizontal_odd.add(
                    HardEdge((qubit, layout_inverted[(coords[0], coords[1] - (-1) ** (rel_coords[0] % 2))]))
                )

    s = [horizontal_odd, horizontal_even, vertical_even, vertical_odd]
    return s


# pylint: disable=too-many-branches
def _do_routing(route: Routing, s: list[set[HardEdge]], h: int, w: int) -> None:
    """Applies the swap and interaction gates to do the routing.

    Works for any rectangle dimensions. Follows steps 1 and 2 in subsection 2.2.2 of :cite:`Weidenfeller_2022`.
    The reference :cite:`Weidenfeller_2022` doesn't mention executing interaction gates in between the swap layers, but
    here we have to explicitly include them, which is why in practice it's a bit more complex than just
    the aforementioned steps 1 and 2. The function modifies ``route`` in-place.

    Args:
        route: The :class:`~iqm.qaoa.transpiler.routing.Routing` to be modified in-place by adding all the routing
            layers.
        s: A list of sets of gates to be executed in parallel (either swaps or interactions or both).
        h: The height of the rectangle on which the routing is done.
        w: The width of the rectangle on which the routing is done.

    """
    # Start with applying the vertical interactions (before any horizontal swapping happens).
    for gate in s[2]:
        route.apply_int(gate)
    for gate in s[3]:
        route.apply_int(gate)
    # Also apply the horizontal interactions between qubits that will soon be separated (by the swaps in ``s[1]``).
    for gate in s[0]:
        route.apply_int(gate)

    for j in range(math.ceil(h / 2)):
        # Apply the horizontal swapping strategy, alternating ``s[0]`` and ``s[1]``.
        # If it's the first time doing it, we also apply interactions.
        for i in range(w - 1):
            if i % 2 == 0:
                for gate in s[1]:
                    route.apply_swap(gate, attempt_int=not j)
            else:
                for gate in s[0]:
                    route.apply_swap(gate, attempt_int=not j)
            # After each layer of horizontal swaps, we apply all vertical interactions.
            for gate in s[3]:
                route.attempt_apply_int(gate)
            # Skip this at the end, because these interactions can be bundled with the upcoming swaps.
            # Unless we're at the very end of the algorithm, in which case there will be no more swaps.
            if i != w - 2 or j == math.ceil(h / 2) - 1:
                for gate in s[2]:
                    route.attempt_apply_int(gate)
        # After the horizontal swapping, apply ``s[2]`` and ``s[3]`` once, with interactions (vertical swap).
        # Skip this at the end of the algorithm.
        if j != math.ceil(h / 2) - 1:
            for gate in s[2]:
                route.apply_swap(gate, attempt_int=True)
            for gate in s[3]:
                route.apply_swap(gate, attempt_int=True)
            # After applying the vertical swaps, we try to apply one more vertical interaction.
            for gate in s[2]:
                route.attempt_apply_int(gate)


# pylint: disable=too-many-locals
def _find_rectangular_subgraph(qpu: CrystalQPUFromBackend, h: int, w: int) -> tuple[HardQubit, int, int] | None:
    """Finds a rectangular subgraph of a given height and width in the QPU.

    Looks at the :attr:`~iqm.qaoa.transpile.quantum_hardware.CrystalQPUFromBackend.hardware_layout` of the QPU. Goes
    over the coordinates from the layout, constructs a boolean :class:`~np.ndarray` and then applies a rectangular
    sliding window to find the rectangle in there. Also considers the 90° rotation of the rectangle.

    Args:
        qpu: The :class:`~iqm.qaoa.transpiler.quantum_hardware.CrystalQPUFromBackend` in which the function is looking
            for the rectangle.
        h: The height of the rectangle which we the function looks for.
        w: The width of the rectangle which we the function looks for.

    Returns:
        A tuple containing the top-left qubit of the placed rectangle, its width and height. Or ``None``, if no such
            rectangle exists.

    """
    # Get min/max coordinates to create a bounded grid.
    x_vals, y_vals = zip(*qpu.hardware_layout.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Create a boolean grid (filled with ``False`` at first).
    grid_width, grid_height = x_max - x_min + 1, y_max - y_min + 1
    boolean_grid = np.zeros((grid_width, grid_height), dtype=bool)

    # Mark occupied nodes in the grid.
    for x, y in qpu.hardware_layout.values():
        boolean_grid[x - x_min, y - y_min] = True

    def _fits_at(x: int, y: int, w: int, h: int) -> np.bool_ | bool:
        """Check if a rectangle of size (``w``, ``h``) fits with its top-left corner at position (``x``, ``y``)."""
        if x + w > grid_width or y + h > grid_height:
            return False  # Out of bounds.
        return np.all(boolean_grid[x : x + w, y : y + h])  # Check if all nodes exist.

    # Iterate over possible top-left corners, return the first one found (if any).
    for hw_qubit in qpu.hardware_graph.nodes():
        if _fits_at(qpu.hardware_layout[hw_qubit][0], qpu.hardware_layout[hw_qubit][1], w, h):  # type: ignore[arg-type]
            return (hw_qubit, w, h)  # Return position and size.
        if _fits_at(
            qpu.hardware_layout[hw_qubit][0],  # type: ignore[arg-type]
            qpu.hardware_layout[hw_qubit][1],  # type: ignore[arg-type]
            h,
            w,
        ):  # Check rotated case
            return (hw_qubit, h, w)

    return None  # No valid rectangle found.
