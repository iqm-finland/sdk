# Copyright 2022-2025 Qiskit on IQM developers
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
"""Transpilation target for IQM quantum computers."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import TypeAlias

from iqm.iqm_client import ObservationFinder
from iqm.qiskit_iqm.move_gate import MoveGate
from qiskit.circuit import Delay, Gate, IfElseOp, Parameter, Reset
from qiskit.circuit.library import CZGate, IGate, Measure, RGate
from qiskit.providers import QubitProperties
from qiskit.transpiler import InstructionProperties, Target

from iqm.station_control.interface.models import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo

Locus: TypeAlias = tuple[str, ...]
"""Sequence of QPU component names on which a gate acts."""
LocusIdx: TypeAlias = tuple[int, ...]
"""Sequence of qubit indices on which a gate acts."""


_QISKIT_IQM_GATE_MAP: dict[str, Gate] = {
    "prx": RGate(Parameter("theta"), Parameter("phi")),
    "measure": Measure(),
    "reset": Reset(),
    "cz": CZGate(),
    "move": MoveGate(),
    "id": IGate(),
    "if_else": IfElseOp,
}
"""Maps IQM native operation names to corresponding Qiskit gate objects."""

logger = logging.getLogger(__name__)


class IQMTarget(Target):
    """Transpilation target for an IQM architecture.

    Contains the mapping of physical qubit name on the device to qubit index in the Target.

    Args:
        architecture: Quantum architecture that defines the target.
        component_to_idx: Mapping from QPU component names to integer indices used by Qiskit to refer to them.
        include_resonators: Whether to include computational resonators (and MOVE gates) in the target,
            if present in ``architecture``.
        include_fictional_czs: Whether to include "fictional" CZs that are not natively supported,
            but could be routed using MOVE.
        metrics: Optional calibration data and related quality metrics to improve the transpilation.

    """

    def __init__(
        self,
        *,
        architecture: DynamicQuantumArchitecture,
        component_to_idx: dict[str, int],
        include_resonators: bool,
        include_fictional_czs: bool,
        metrics: ObservationFinder | None = None,
    ):
        super().__init__()
        # HACK: As of Qiskit 1.3, Target.__init__ does nothing with its args, instead Target.__new__
        # handles them.  IQMTarget.__init__ has different args than Target.__init__, hence we must either
        # (1) define IQMTarget.__new__ with matching args that calls Target.__new__, or
        # (2) make all the IQMTarget.__init__ args keyword-only with non-colliding names, and init
        # the necessary superclass attributes ourselves.
        # (1) seems to break pickling during concurrent transpilation using Qiskit's ``transpile``, so we use (2).
        # (2) is broken from Qiskit 2.2 onwards because Target.__init__ no longer has **kwargs nor *args.
        self.qubit_properties = self._create_qubit_properties(architecture.qubits, metrics)

        # Using iqm_ as a prefix to avoid name clashes with the base class.
        self.iqm_dqa = architecture
        self.iqm_component_to_idx = component_to_idx
        self.iqm_idx_to_component = {v: k for k, v in component_to_idx.items()}
        self.iqm_include_resonators = include_resonators
        self.iqm_include_fictional_czs = include_fictional_czs
        self.iqm_metrics = metrics
        self._add_instructions_from_DQA()

    def _add_instructions_from_DQA(self):  # noqa: ANN202
        """Initializes the Target with instructions and properties that represent the
        dynamic quantum architecture :attr:`iqm_dqa`.

        """
        dqa = self.iqm_dqa
        # mapping from op name to all its allowed loci
        op_loci = {gate_name: gate_info.loci for gate_name, gate_info in dqa.gates.items()}

        def add_gate(gate: str) -> None:
            """Adds ``gate`` instructions to the Target."""
            props = {self.locus_to_idx(locus): self._create_gate_properties(gate, locus) for locus in op_loci[gate]}
            self.add_instruction(_QISKIT_IQM_GATE_MAP[gate], props)

        # identity gate does nothing and is removed in serialization, so we may as well allow it everywhere
        # Except if it is defined for the resonator, the graph is disconnected and the transpiler will fail.
        self.add_instruction(
            IGate(),
            {
                self.locus_to_idx((component,)): None
                for component in (dqa.components if self.iqm_include_resonators else dqa.qubits)
            },
        )

        # like barrier, delay is always available for all single-qubit loci
        self.add_instruction(Delay(0), {self.locus_to_idx((q,)): None for q in dqa.qubits})

        if "measure" in op_loci:
            add_gate("measure")

        if "prx" in op_loci:
            add_gate("prx")

        if "cc_prx" in op_loci:
            # IfElseOp is a global 'gate' so it's slightly different from the others.
            self.add_instruction(
                instruction=_QISKIT_IQM_GATE_MAP["if_else"],
                name="if_else",
            )
            # HACK reset gate shares cc_prx loci for now, until reset is also in the DQA/metrics
            self.add_instruction(
                _QISKIT_IQM_GATE_MAP["reset"],
                {self.locus_to_idx(locus): None for locus in op_loci["cc_prx"]},
            )

        if self.iqm_include_resonators and "move" in op_loci:
            add_gate("move")

        if "cz" in op_loci:
            self._add_instructions_cz_gates(op_loci)

    def _add_instructions_cz_gates(self, op_loci: dict[str, Iterable[Locus]]) -> None:
        """Adds CZ gate instructions to the Qiskit Target.

        This method handles both "real" and "fictional" CZ gates based on the
        provided operation loci.

        1.  **Real CZ Gates**:
            A CZ gate for a locus is considered "real" if the locus
            exists in ``dynamic_quantum_architecture.gates['cz']``.
            A "real" CZ instruction is added if either the target is configured to
            include resonators (:attr:`iqm_include_resonators` is True) or if all
            the locus components are qubits.

        2.  **Fictional CZ Gates**: For STAR QPU architectures, a "fictional" CZ between
            two qubits that are not physically connected is defined
            iff there is a CZ gate from one qubit to a resonator, and a MOVE gate
            from the other qubit to the resonator.
            See :mod:`iqm.iqm_client.transpile`.

        Args:
            op_loci: Mapping of operation names (e.g., "cz", "move") to their available loci.

        """
        cz_props: dict[LocusIdx, InstructionProperties] = {}
        cz_loci = op_loci["cz"]

        # 1. Add real CZs
        for locus in cz_loci:
            if self.iqm_include_resonators or all(component in self.iqm_dqa.qubits for component in locus):
                locus_idx = self.locus_to_idx(locus)  # convert locus from IQM to Qiskit format
                cz_props[locus_idx] = self._create_gate_properties("cz", locus)

        # 2. Add fictional CZs if applicable
        if self.iqm_include_fictional_czs and "move" in op_loci:
            fictional_cz_loci = self._determine_fictional_cz_loci(op_loci["move"], cz_loci)
            for q1, q2, res in fictional_cz_loci:
                locus_idx = self.locus_to_idx((q1, q2))
                props = self._create_fictional_cz_properties(q1, q2, res)
                cz_props[locus_idx] = props
                cz_props[locus_idx[::-1]] = props  # fictional CZs are symmetric
                # TODO: if the fictional CZ can be implemented using both CZ(a,r) and MOVE(b,r), and
                # CZ(b,r) and MOVE(a,r), props will be only included for one implementation at random.
                # This is not necessarily the implementation the transpiler chooses!

        if cz_props:
            self.add_instruction(_QISKIT_IQM_GATE_MAP["cz"], cz_props)

    def _create_fictional_cz_properties(self, q1: str, q2: str, res: str) -> InstructionProperties:
        """Create ``InstructionProperties`` for a fictional CZ gate.

        A fictional CZ gate is comprised of the sequence MOVE(q2, res), CZ(q1, res), MOVE(q2, res).

        The duration and error for the fictional CZ gate are calculated as follows:

        - Duration: 2 * duration_move(q2, res) + duration_cz(q1, res)
        - Error: 1 - (1 - error_move(q2, res)))^2 * (1 - error_cz(q1, res))

        See https://arxiv.org/pdf/2503.10903 for more details.

        Args:
            q1: CZ qubit.
            q2: MOVE qubit.
            res: Shared computational resonator through which the fictional CZ gate is routed.

        Returns:
            Instruction properties for the fictional CZ gate.

        """
        if self.iqm_metrics is None:
            return None

        props_cz = self._create_gate_properties("cz", (q1, res))
        props_move = self._create_gate_properties("move", (q2, res))

        if props_cz is None or props_move is None:
            return None

        duration = None
        if props_move.duration is not None and props_cz.duration is not None:
            duration = 2 * props_move.duration + props_cz.duration

        error: float | None = None
        if props_move.error is not None and props_cz.error is not None:
            error = 1 - (1 - props_move.error) ** 2 * (1 - props_cz.error)

        return InstructionProperties(duration=duration, error=error)

    @staticmethod
    def _determine_fictional_cz_loci(move_loci: Iterable[Locus], cz_loci: Iterable[Locus]) -> list[Locus]:
        """Determine fictional CZ loci, i.e. pairs of qubits between which we can implement a CZ using MOVE gates and
        an intermediate computational resonator.

        Only applicable for the Star architecture.

        Args:
            move_loci: The loci for which a MOVE operation is defined.
            cz_loci: The loci for which a CZ operation is defined.

        Returns:
            Loci for fictional CZ gates, *including the resonator*. Each locus is given as ``(q1, q2, res)``.

        """
        fictional_cz_loci: list[Locus] = []
        for q1, res1 in cz_loci:
            for q2, res2 in move_loci:
                if res1 == res2 and q1 != q2:
                    # shared resonator, different qubits
                    fictional_cz_loci.append((q1, q2, res1))

        return fictional_cz_loci

    def locus_to_idx(self, locus: Locus) -> LocusIdx:
        """Map the given locus to use component indices instead of component names."""
        return tuple(self.iqm_component_to_idx[component] for component in locus)

    @staticmethod
    def _create_qubit_properties(
        qubits: Iterable[str], metrics: ObservationFinder | None
    ) -> list[QubitProperties] | None:
        """Creates qubit properties from the quality metrics."""
        if metrics is None:
            return None

        qubit_props = []
        t1_times, t2_times = metrics.get_coherence_times(qubits)
        for q in qubits:
            frequency = metrics.get_qubit_frequency(q)
            qubit_props.append(
                QubitProperties(
                    t1=t1_times.get(q),
                    t2=t2_times.get(q),
                    frequency=frequency,
                )
            )
        return qubit_props

    def _create_gate_properties(self, gate_name: str, locus: Locus) -> InstructionProperties | None:
        """Creates InstructionProperties for a single gate on a specific locus.

        Args:
            gate_name: Name of the IQM native operation to look up properties for (e.g., 'cz' or 'move').
            locus: Locus of the operation.

        Returns:
            Properties for the (default implementation of the) given gate at the given locus, or None if not available.

        """
        if self.iqm_metrics is None:
            return None

        # the properties are for the default implementation (other implementations may be available also)
        impl_name = self.iqm_dqa.gates[gate_name].get_default_implementation(locus)
        duration = self.iqm_metrics.get_gate_duration(gate_name, impl_name, locus)
        fidelity = self.iqm_metrics.get_gate_fidelity(gate_name, impl_name, locus)
        return InstructionProperties(duration=duration, error=None if fidelity is None else 1 - fidelity)

    @property
    def physical_qubits(self) -> list[str]:
        """Return the ordered list of physical qubits in the target."""
        # Overrides the property from the superclass to contain the correct information.
        return [self.iqm_idx_to_component[i] for i in range(self.num_qubits)]

    def restrict_to_qubits(self, qubits: list[int] | list[str]) -> IQMTarget:
        """Generated a restricted transpilation target from this Target that only contains the given qubits.

        Args:
            qubits: Qubits to restrict the target to. Can be either a list of qubit indices or qubit names.

        Returns:
            restricted target

        """
        qubits_str = [self.iqm_idx_to_component[q] if isinstance(q, int) else str(q) for q in qubits]
        return _restrict_dqa_to_qubits(
            self.iqm_dqa,
            qubits_str,
            self.iqm_include_resonators,
            self.iqm_include_fictional_czs,
            metrics=self.iqm_metrics,
        )


def _restrict_dqa_to_qubits(
    architecture: DynamicQuantumArchitecture,
    components: list[str],
    include_resonators: bool,
    include_fictional_czs: bool,
    metrics: ObservationFinder | None = None,
) -> IQMTarget:
    """Generated a restricted transpilation target that only contains the given QPU components.

    Args:
        architecture: The dynamic quantum architecture to restrict.
        components: QPU components to restrict the target to. Qubits first, then resonators.
        include_resonators: Whether to include computational resonators (and MOVE gates) in the target.
        include_fictional_czs: Whether to include fictional CZs that are not natively supported,
            but could be routed via MOVE.
        metrics: Optional calibration data and related quality metrics to improve the transpilation.

    Returns:
        restricted target

    """
    # include gate loci that only involve the given components
    c_set = set(components)
    new_gates = {}
    for gate_name, gate_info in architecture.gates.items():
        new_implementations = {}
        for implementation_name, implementation_info in gate_info.implementations.items():
            new_loci = tuple(locus for locus in implementation_info.loci if set(locus) <= c_set)
            if new_loci:
                new_implementations[implementation_name] = GateImplementationInfo(loci=new_loci)
        if new_implementations:
            new_gates[gate_name] = GateInfo(
                implementations=new_implementations,
                default_implementation=gate_info.default_implementation,
                override_default_implementation=gate_info.override_default_implementation,
            )
    new_arch = DynamicQuantumArchitecture(
        calibration_set_id=architecture.calibration_set_id,
        qubits=[q for q in architecture.qubits if q in c_set],
        computational_resonators=[r for r in architecture.computational_resonators if r in c_set],
        gates=new_gates,
    )
    return IQMTarget(
        architecture=new_arch,
        component_to_idx={name: idx for idx, name in enumerate(components)},
        include_resonators=include_resonators,
        include_fictional_czs=include_fictional_czs,
        metrics=metrics,
    )
