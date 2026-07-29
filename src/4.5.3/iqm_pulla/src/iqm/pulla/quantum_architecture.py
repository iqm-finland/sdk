# Copyright 2025-2026 IQM
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
"""Creating static and dynamic quantum architectures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from exa.common.qcm_data.chip_topology import ChipTopology, sort_components
from iqm.pulla.interface import CalibrationSetValues
from iqm.pulse.builder import build_quantum_ops
from iqm.station_control.client.qon import (
    QON,
    QONGateDefinition,
    QONGateParam,
    UnknownObservationError,
    locus_str_to_locus,
)
from iqm.station_control.interface.models import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    StaticQuantumArchitecture,
)

if TYPE_CHECKING:
    from iqm.pulse import Locus
    from iqm.station_control.interface.models.observation import Value

logger = logging.getLogger(__name__)


def create_static_quantum_architecture(dut_label: str, chip_topology: ChipTopology) -> StaticQuantumArchitecture:
    """Create a static quantum architecture (SQA) for the given chip topology.

    Args:
        dut_label: Identifies the QPU that ``chip_topology`` describes.
        chip_topology: The chip topology.

    Returns:
         Static quantum architecture containing information about qubits, computational resonators, and connectivity.

    """
    # Components within each connection are sorted by coupler_to_components
    unsorted_connections = list(chip_topology.coupler_to_components.values())
    # Sort connections first based on the first component, then the second component etc. of each connection
    sorted_components = sort_components({component for connection in unsorted_connections for component in connection})
    component_to_index = {component: index for index, component in enumerate(sorted_components)}

    def sort_key(connection: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(component_to_index[component] for component in connection)

    # The components in each connection are already sorted, now we sort the connections
    connectivity = sorted(unsorted_connections, key=sort_key)
    return StaticQuantumArchitecture(
        dut_label=dut_label,
        qubits=list(chip_topology.qubits_sorted),
        computational_resonators=list(chip_topology.computational_resonators_sorted),
        connectivity=connectivity,
    )


def create_dynamic_quantum_architecture(  # noqa: PLR0915
    calibration_set_id: UUID,
    observations: CalibrationSetValues,
    chip_topology: ChipTopology,
) -> DynamicQuantumArchitecture:
    """Create a dynamic quantum architecture (DQA) for the given calibration set.

    Has deprecated support for old-style calibration sets without default implementation observations.

    Args:
        calibration_set_id: ID of the calibration set used to create the DQA.
        observations: Contents of the calibration set used to create the DQA.
        chip_topology: Topology of the QPU the calibration set is for.

    Returns:
         Dynamic quantum architecture containing information about calibrated operations.

    """
    # TODO cache the DQA based on calibration_set_id

    qubits: set[str] = set()
    computational_resonators: set[str] = set()
    gates: dict[str, GateInfo] = {}

    # known gates and implementations
    quantum_op_table = build_quantum_ops({})

    def log(level: int, message: str) -> None:
        """Logging errors."""
        base = f"Calibration set {calibration_set_id}: "
        logger.log(level, base + message)

    # error reporting
    unknown_ops: set[str] = set()
    unknown_implementations: set[str] = set()

    def analyze_observation(obs_name: str, obs_value: Value) -> None:
        """Deduce ops/implementations/loci and used QPU components from the gate cal data.

        The assumption is that calibration sets are complete: If there is even one observation for a
        (gate, implementation, locus) triple, the entire triple is assumed to be calibrated and is added to the DQA.
        """
        try:
            qon = QON.from_str(obs_name)
        except (UnknownObservationError, ValueError):
            # unparseable observation
            log(logging.WARNING, f"Unparseable observation {obs_name}")
            return
        if not isinstance(qon, (QONGateParam, QONGateDefinition)):
            # log(logging.WARNING, f"Unexpected observation type {obs_name}")
            return

        # ignore unknown gates and implementations (for backwards and forwards compatibility)
        if (quantum_op := quantum_op_table.get(qon.gate)) is None:
            unknown_ops.add(qon.gate)
            return
        gate_info = gates.setdefault(
            qon.gate,
            GateInfo(
                implementations={},
                default_implementation="",
                override_default_implementation={},
            ),
        )
        if (
            isinstance(qon, QONGateDefinition)
            and qon.implementation is None
            and qon.quantity == "default_implementation"
        ):
            # default implementation is the only gate property we care about
            impl = str(obs_value)
            if impl not in quantum_op.implementations:
                log(logging.ERROR, f"default implementation '{impl}' for the gate {qon.gate} is unknown")
            gate_info.default_implementation = impl
            return

        if qon.implementation not in quantum_op.implementations:
            unknown_implementations.add(f"{qon.gate}.{qon.implementation}")
            return
        if isinstance(qon, QONGateDefinition):
            if qon.quantity != "override_default_for_loci":
                log(logging.WARNING, f"Unexpected observation {obs_name}")
                return
            # default implementation override is the only implementation property we care about
            if not isinstance(obs_value, Iterable):
                raise ValueError(f"Observation {qon} has the invalid value {obs_value}")
            for locus_str in obs_value:
                gate_info.override_default_implementation[locus_str_to_locus(locus_str)] = qon.implementation
            return

        # the observation must be a QONGateParam
        impl_info = gate_info.implementations.setdefault(qon.implementation, GateImplementationInfo(loci=tuple()))
        locus = qon.locus
        if locus not in impl_info.loci:
            impl_info.loci += (locus,)
        for locus_component in locus:
            if chip_topology.is_qubit(locus_component):
                qubits.add(locus_component)
            if chip_topology.is_computational_resonator(locus_component):
                computational_resonators.add(locus_component)

    for obs_name, obs_value in observations.items():
        analyze_observation(obs_name, obs_value)

    for name in unknown_ops:
        log(logging.INFO, f"Unknown operation '{name}'")
    for name in unknown_implementations:
        log(logging.INFO, f"Unknown implementation '{name}'")

    # remove gates with no known implementations
    gates = {gate_name: gate_info for gate_name, gate_info in gates.items() if gate_info.implementations}

    # check that there are no discrepancies
    for gate_name, gate_info in gates.items():
        def_impl = gate_info.default_implementation
        if def_impl:
            if def_impl not in gate_info.implementations:
                log(
                    logging.ERROR,
                    f"Default implementation '{def_impl}' for the gate {gate_name} has no calibrated loci",
                )

            for locus, def_impl in gate_info.override_default_implementation.items():
                impl_info = gate_info.implementations.get(def_impl)
                if impl_info is None or locus not in impl_info.loci:
                    log(
                        logging.ERROR,
                        f"Locus {locus} default implementation '{def_impl}' for the gate {gate_name} is not calibrated",
                    )

    def pick_default_impl(gate_name: str, gate_info: GateInfo) -> None:
        """Old method for picking default implementations for gates."""
        locus_default_implementations: dict[Locus, str] = {}
        # pick the default implementation for each locus using the hardcoded priority order
        for implementation in quantum_op_table[gate_name].implementations:
            if implementation in gate_info.implementations:
                for locus in gate_info.implementations[implementation].loci:
                    locus_default_implementations.setdefault(locus, implementation)

        # choose default implementation to be the most common locus-specific default,
        # and add the other locus-specific defaults to override_default_implementation
        most_common = Counter(locus_default_implementations.values()).most_common(1)
        gate_info.default_implementation = most_common[0][0]
        gate_info.override_default_implementation = {
            locus: implementation
            for locus, implementation in locus_default_implementations.items()
            if implementation != gate_info.default_implementation
        }

    # Deprecated backwards compatibility for old calsets without default implementation observables:
    # ``gates`` only contains known gates and implementations, and each implementation has
    # at least one locus. Pick a default implementations for each available (gate, locus).
    for gate_name, gate_info in gates.items():
        if not gate_info.default_implementation:
            # already defined, skip the old method
            log(logging.WARNING, f"Gate {gate_name} has no default_implementation observation")
            pick_default_impl(gate_name, gate_info)

    return DynamicQuantumArchitecture(
        calibration_set_id=calibration_set_id,
        qubits=sort_components(qubits),
        computational_resonators=sort_components(computational_resonators),
        gates=gates,
    )
