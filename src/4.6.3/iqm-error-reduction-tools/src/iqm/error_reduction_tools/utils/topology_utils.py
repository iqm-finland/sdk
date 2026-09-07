# Copyright 2022-2026 IQM
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

"""Utility functions for resolving QPU topology from remote quantum computer.

Provides helpers that convert chip topology objects into the
:class:`~iqm.error_reduction_tools.readout_characterization.topologies.QPUTopology` used by
readout-twirling string generation, and detection of IQM Star QPUs
where circuit twirling is not supported because those QPUs
use MOVE gates and computational resonators rather than direct qubit-qubit CZ gates, which
are not handled by the PMPT implementation.
"""

from __future__ import annotations

from collections import deque
import itertools
import logging
from typing import TYPE_CHECKING

from iqm.error_reduction_tools.readout_characterization.topologies import QPUTopology

if TYPE_CHECKING:
    from iqm.pulla.pulla import Pulla

logger = logging.getLogger(__name__)


def compute_crystal_layout(
    couplers: list[tuple[str, str]],
    qubits: list[str],
) -> dict[str, tuple[int, int]]:
    """Compute a square-grid layout for IQM Crystal architecture QPUs.

    Uses the coupler connectivity graph to recursively identify 4-qubit
    plaquettes and assign (x, y) grid coordinates.  This algorithm works
    for any IQM Crystal QPU (Garnet, Emerald, etc.) but will not work with
    the IQM Star topology.

    Args:
        couplers: Qubit pairs representing the connectivity.
        qubits: Qubit names on the QPU.

    Returns:
        Mapping from qubit names to (x, y) integer grid positions.
        Qubits that could not be placed (e.g. in a Star topology) are omitted.

    """
    # Build neighbor graph
    neighbors: dict[str, set[str]] = {}
    for q1, q2 in couplers:
        neighbors.setdefault(q1, set()).add(q2)
        neighbors.setdefault(q2, set()).add(q1)

    # Ensure all qubits appear in the neighbor dict (isolated qubits get empty sets)
    for q in qubits:
        neighbors.setdefault(q, set())

    layout: dict[str, tuple[int, int]] = {}

    def _assign_coords(plaquette: list[str] | deque[str], x: int, y: int) -> None:
        """Assign coordinates for a 4-qubit plaquette.

        plaquette[0] is the lower-left corner, continuing counter-clockwise.
        """

        def set_coords(q: str, coords: tuple[int, int]) -> None:
            if (old := layout.get(q)) is not None:
                if coords != old:
                    raise RuntimeError(f"Ambiguous coords for qubit {q}")
                return
            layout[q] = coords

        set_coords(plaquette[0], (x, y))
        set_coords(plaquette[1], (x + 1, y))
        set_coords(plaquette[2], (x + 1, y + 1))
        set_coords(plaquette[3], (x, y + 1))

    def _create_plaquette(q1: str, q2: str, taken: set[str]) -> list[str] | None:
        """Try to construct a new 4-qubit plaquette from edge (q1, q2)."""
        n1 = neighbors[q1] - taken
        n2 = neighbors[q2] - taken
        for a, b in itertools.product(n1, n2):
            if b in neighbors[a]:
                break
        else:
            return None
        plaq = [q1, q2, b, a]
        if all(q in layout for q in plaq):
            return None
        return plaq

    def _handle_plaquette(plaquette: list[str] | deque[str], x: int, y: int) -> None:
        """Recursively assign coordinates for a plaquette and its neighbors."""
        _assign_coords(plaquette, x, y)
        p: deque[str] = deque(plaquette)
        taken = set(plaquette)
        for k, (dx, dy) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
            new = _create_plaquette(p[3], p[2], taken)
            if new is not None:
                new_d: deque[str] = deque(new)
                new_d.rotate(-k)
                _handle_plaquette(new_d, x + dx, y + dy)
            p.rotate(1)

    # Find a corner qubit (degree 2) to start the layout
    corners = [q for q, n in neighbors.items() if len(n) == 2]  # noqa: PLR2004
    if not corners:
        # Not a crystal lattice (e.g. star topology) — return empty
        return {}

    q1 = corners[0]
    n = list(neighbors[q1])
    nn = neighbors[n[0]] & neighbors[n[1]]
    nn.discard(q1)
    if not nn:
        return {}
    fourth = next(iter(nn))
    first_plaquette = [q1, n[0], fourth, n[1]]

    _handle_plaquette(first_plaquette, 0, 0)
    return layout


def topology_from_qc(client: Pulla) -> QPUTopology | None:
    """QPU topology representing a remote quantum computer.

    Retrieves the QPU topology information from the remote quantum computer and converts it into a
    :class:`QPUTopology` suitable for twirling-string generation.

    Args:
        client: Client instance for connecting to the quantum computer.

    Returns:
        A :class:`QPUTopology`, or ``None`` if the topology cannot be
        retrieved from the server.

    """
    try:
        chip_topo = client.get_chip_topology()
    except Exception:
        logger.debug("Could not retrieve chip topology from the server.", exc_info=True)
        return None

    qubits_sorted = list(chip_topo.qubits_sorted)
    num_qubits = len(qubits_sorted)

    # Extract CZ couplers via coupler_to_components: each value is a sorted tuple of
    # the two qubit names the coupler connects.
    couplers: list[tuple[str, str]] = []
    dropped = 0
    for name, comps in chip_topo.coupler_to_components.items():
        qubit_comps = [c for c in comps if c in chip_topo.qubits]
        if len(qubit_comps) == 2:  # noqa: PLR2004
            couplers.append((qubit_comps[0], qubit_comps[1]))
        elif len(qubit_comps) > 2:  # noqa: PLR2004
            dropped += 1
            logger.debug(
                "Coupler %s connects %d qubits (%s); skipped (only 2-qubit couplers supported).",
                name,
                len(qubit_comps),
                qubit_comps,
            )
    if dropped:
        logger.warning(
            "Skipped %d multi-qubit coupler(s) when building QPUTopology. "
            "Twirling strategies that depend on connectivity (LOCAL) may be inaccurate.",
            dropped,
        )

    # Extract control lines (probe-line → qubit groupings) from the chip topology.
    # Each probe line groups the qubits that share the same readout line.
    control_lines: list[tuple[str, ...]] = [
        tuple(c for c in comps if c in chip_topo.qubits) for comps in chip_topo.probe_line_to_components.values()
    ]

    # Compute qubit grid positions from connectivity (works for Crystal architectures).
    # Returns qubit_name -> (x, y); convert to qubit_index -> (x, y) for QPUTopology.
    name_positions = compute_crystal_layout(couplers, qubits_sorted)
    positions: dict[int, tuple[int, int]] = {}
    for qname, coords in name_positions.items():
        positions[QPUTopology.parse_qubit_index(qname)] = coords

    return QPUTopology(
        name="backend",
        num_qubits=num_qubits,
        positions=positions,
        couplers=couplers,
        control_lines=control_lines,
    )


def operational_qubits_from_qc(client: Pulla) -> list[str] | None:
    """Operational qubit names on the remote quantum computer.

    A qubit is considered operational if the current default calibration set
    contains a calibrated ``measure`` gate for it, i.e. it can be read out. This is the
    relevant notion of "operational" for readout error characterization and mitigation:
    qubits without readout calibration cannot be meaningfully characterized.

    The information is taken from the dynamic quantum architecture (DQA), which describes
    exactly the operations for which calibration data exists.

    Args:
        client: Client instance for connecting to the quantum computer.

    Returns:
        Operational qubit names, sorted as reported by the DQA, or ``None`` if the
        information cannot be retrieved from the server.

    """
    try:
        dqa = client._iqm_server_client.get_dynamic_quantum_architecture("default")
    except Exception:
        logger.debug("Could not retrieve dynamic quantum architecture from the server.", exc_info=True)
        return None

    measure_gate = dqa.gates.get("measure")
    if measure_gate is None:
        # No calibrated measure gate reported by the DQA. Fall back to the qubit list the
        # DQA itself advertises so that callers still get a sensible default instead of None.
        dqa_qubits = getattr(dqa, "qubits", None)
        return list(dqa_qubits) if dqa_qubits else None

    # ``measure_gate.loci`` is sorted by the DQA; preserve that order while collecting
    # the single-qubit loci of the measure gate.
    qubits: list[str] = []
    seen: set[str] = set()
    for locus in measure_gate.loci:
        for qubit in locus:
            if qubit not in seen:
                seen.add(qubit)
                qubits.append(qubit)

    return qubits


def uses_move_gates(client: Pulla) -> bool:
    """Detect whether a QPU uses MOVE gates, which makes circuit twirling unsupported.

    IQM QPUs that include computational resonators use MOVE gates
    rather than direct CZ gates between qubits.  The PMPT implementation only handles
    ``prx`` and ``cz`` operations and will raise a ``ValueError`` if it
    encounters a ``move`` instruction, so twirling cannot be applied to circuits
    compiled for these QPUs.

    This function detects such QPUs directly by checking whether the chip
    topology reports any computational resonators, which is the authoritative
    indicator that MOVE gates are present.

    Args:
        client: Client instance for connecting to the quantum computer.

    Returns:
        ``True`` if the QPU has computational resonators (and therefore uses
        MOVE gates), ``False`` otherwise or if the topology cannot be retrieved.

    """
    try:
        chip_topo = client.get_chip_topology()
    except Exception:
        logger.debug("Could not retrieve chip topology to check for MOVE gates.", exc_info=True)
        return False

    return bool(chip_topo.computational_resonators)
