"""Methods for creating static and dynamic quantum architectures."""

from collections import Counter
import logging
from uuid import UUID

from exa.common.qcm_data.chip_topology import ChipTopology, sort_components
from iqm.pulse.builder import build_quantum_ops
from iqm.station_control.interface.models import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    Locus,
    ObservationLite,
    StaticQuantumArchitecture,
)

logger = logging.getLogger(__name__)


def create_static_quantum_architecture(chip_topology: ChipTopology) -> StaticQuantumArchitecture:
    """Creates a static quantum architecture (SQA) for the given chip topology.

    Args:
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
        qubits=chip_topology.qubits_sorted,  # type: ignore[arg-type]
        computational_resonators=chip_topology.computational_resonators_sorted,  # type: ignore[arg-type]
        connectivity=connectivity,
    )


def create_dynamic_quantum_architecture(
    calibration_set_id: UUID,
    observations: list[ObservationLite],
    chip_topology: ChipTopology,
) -> DynamicQuantumArchitecture:
    """Creates a dynamic quantum architecture (DQA) for the given calibration set.

    Args:
        calibration_set_id: ID of the calibration set used to create the DQA.
        observations: Calibration set observations used to create the DQA.
        chip_topology: The chip topology.

    Returns:
         Dynamic quantum architecture containing information about calibrated gates/operations.

    """
    qubits: set[str] = set()
    computational_resonators: set[str] = set()
    gates: dict[str, GateInfo] = {}

    # known gates and implementations
    quantum_op_table = build_quantum_ops({})

    # error reporting
    unknown_ops: set[str] = set()
    unknown_implementations: set[str] = set()

    def analyze_observation(obs_name: str) -> None:
        """Deduce ops/implementations/loci and used QPU components from the gate cal data."""
        parts = obs_name.split(".")
        if parts[0] == "gates":
            gate_name, gate_implementation, gate_locus = parts[1:4]
            # ignore unknown gates and implementations (for backwards and forwards compatibility)
            if (quantum_op := quantum_op_table.get(gate_name)) is None:
                unknown_ops.add(gate_name)
                return
            if gate_implementation not in quantum_op.implementations:
                unknown_implementations.add(f"{gate_name}.{gate_implementation}")
                return

            gate_info = gates.setdefault(
                gate_name,
                GateInfo(
                    implementations={},
                    default_implementation="",
                    override_default_implementation={},
                ),
            )
            gate_implementation_info = gate_info.implementations.setdefault(
                gate_implementation, GateImplementationInfo(loci=tuple())
            )
            gate_locus_components = tuple(gate_locus.split("__"))
            if gate_locus_components not in gate_implementation_info.loci:
                gate_implementation_info.loci += (gate_locus_components,)
            for locus_component in gate_locus_components:
                if chip_topology.is_qubit(locus_component):
                    qubits.add(locus_component)
                if chip_topology.is_computational_resonator(locus_component):
                    computational_resonators.add(locus_component)

    for observation in observations:
        analyze_observation(observation.dut_field)

    for name in unknown_ops:
        logger.info("Unknown operation '%s' found in calibration set %s", name, str(calibration_set_id))
    for name in unknown_implementations:
        logger.info("Unknown implementation '%s' found in calibration set %s", name, str(calibration_set_id))

    # Now ``gates`` only contains known gates and implementations, and each implementation has
    # at least one locus. Pick a default implementations for each available (gate, locus).
    for gate_name, gate_info in gates.items():
        locus_default_implementations: dict[Locus, str] = {}

        # pick the default implementation for each locus using the hardcoded priority order
        for implementation in quantum_op_table[gate_name].implementations:
            if implementation in gate_info.implementations:
                for locus in gate_info.implementations[implementation].loci:
                    locus_default_implementations.setdefault(locus, implementation)

        # choose default implementation to be the most common locus-specific default,
        # and add the other locus-specific defaults to override_default_implementation
        gate_info.default_implementation = Counter(locus_default_implementations.values()).most_common(1)[0][0]
        gate_info.override_default_implementation = {
            locus: implementation
            for locus, implementation in locus_default_implementations.items()
            if implementation != gate_info.default_implementation
        }

    return DynamicQuantumArchitecture(
        calibration_set_id=calibration_set_id,
        qubits=sort_components(qubits),
        computational_resonators=sort_components(computational_resonators),
        gates=gates,
    )
