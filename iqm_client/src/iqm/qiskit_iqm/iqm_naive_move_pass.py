# Copyright 2024 Qiskit on IQM developers
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
"""Naive transpilation for the IQM Star architecture."""

from typing import Any
import warnings

from iqm.iqm_client.transpile import ExistingMoveHandlingOptions, transpile_insert_moves
import numpy as np
from pydantic_core import ValidationError
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

from iqm.pulse import Circuit

from .iqm_backend import IQMBackendBase, IQMTarget
from .iqm_move_layout import generate_initial_layout
from .qiskit_to_iqm import deserialize_instructions, serialize_instructions


class IQMNaiveResonatorMoving(TransformationPass):
    """Naive transpilation pass for resonator moving.

    The logic of this pass is deferred to :func:`~iqm.iqm_client.transpile.transpile_insert_moves`.
    This pass is a wrapper that converts the circuit into the IQM circuit format,
    runs the ``transpile_insert_moves`` function, and then converts the result back to a Qiskit circuit.

    Args:
        target: Transpilation target.
        existing_moves_handling: How to handle existing MOVE gates in the circuit.

    """

    def __init__(
        self,
        target: IQMTarget,
        existing_moves_handling: ExistingMoveHandlingOptions = ExistingMoveHandlingOptions.KEEP,
    ):
        super().__init__()
        self.target = target
        self.architecture = target.iqm_dqa
        self.idx_to_component = target.iqm_idx_to_component
        self.component_to_idx = target.iqm_component_to_idx
        self.existing_moves_handling = existing_moves_handling

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on a circuit.

        Args:
            dag: DAG to insert MOVE gates into.

        Returns:
            The new ``dag`` now including MOVE gates as needed.

        Raises:
            TranspilerError: The layout is not compatible with the DAG, or if something goes wrong during transpilation.

        """
        if len(dag.op_nodes()) == 0:
            return dag  # Empty circuit, no need to transpile.
        symbolic_gates = self._remove_parameterized_gates(dag)
        layout = self._calculate_initial_layout(dag)
        new_dag = self._insert_move_gate(dag, layout)
        self._insert_parameterized_gates(new_dag, symbolic_gates)
        self.property_set["final_layout"] = self._calculate_final_layout(dag, layout)
        return new_dag

    def _calculate_initial_layout(self, dag: DAGCircuit) -> Layout:
        """Calculate the initial physical to logical qubit layout before the circuit.

        Args:
            dag: The transpiled DAGCircuit.

        Returns:
            The initial layout of which logical qubits are in which physical qubits at the start of the circuit.

        """
        # For some reason, the dag does not always contain the layout, so we need to do a bunch of fixing.
        if self.property_set.get("layout"):
            return self.property_set["layout"]
        # Reconstruct the layout from the dag.
        layout = Layout()
        for qreg in dag.qregs:
            layout.add_register(qreg)
        for i, qubit in enumerate(dag.qubits):
            layout.add(qubit, i)
        return layout

    def _remove_parameterized_gates(self, dag: DAGCircuit) -> dict[float, tuple[Any, ...]]:
        """Remove parameterized gates from the DAGCircuit.

        Args:
            dag: The input DAGCircuit, which is changed in place.

        Returns:
            A mapping from symbolic index to the original gate parameters.

        """
        # TODO: Temporary hack to get the symbolic parameters to work: replace symbols with (inf, idx).
        # Replace symbolic parameters with indices and store the index to symbol mapping.
        symbolic_gates = {}
        symbolic_index = 0.0
        for node in dag.topological_op_nodes():
            # This only works for prx gates because that has two parameters
            # We use one to mark that it is a symbolic gate (np.inf) and the other to store the index.
            if node.name == "r" and not all(isinstance(param, (float, int)) for param in node.op.params):
                symbolic_gates[symbolic_index] = node.op.params
                dag.substitute_node(node, RGate(np.inf, float(symbolic_index)))
                symbolic_index += 1
        return symbolic_gates

    def _insert_parameterized_gates(self, dag: DAGCircuit, symbolic_gates: dict[float, tuple[Any, ...]]) -> None:
        """Reinsert parameterized gates into the DAGCircuit.

        Args:
            dag: The input DAGCircuit, which is changed in place.
            symbolic_gates: A mapping from symbolic index to the original gate parameters.

        """
        for node in dag.topological_op_nodes():
            # This only works for prx gates because that has two parameters
            # We use one to mark that it is a symbolic gate (np.inf) and the other to store the index.
            if node.name == "r" and not np.isfinite(node.op.params[0]):
                dag.substitute_node(node, RGate(*symbolic_gates[int(np.round(node.op.params[1]))]))

    def _calculate_final_layout(self, dag: DAGCircuit, layout: Layout) -> Layout:
        """Calculate the final physical to logical qubit layout after the circuit.
        The original final layout from the previous passes uses the old qubit registers that are no longer valid.

        Args:
            dag: The original DAGCircuit without the inserted MOVE gates..
            layout: The initial physical to logical qubit layout at the start of the circuit.

        Returns:
            The updated layout of which logical qubits are in which physical qubits at the end of the circuit.

        """
        if "final_layout" not in self.property_set or self.property_set["final_layout"] is None:
            # If the final layout is not set, return the initial layout.
            return layout
        # This final layout only has a single QuantumRegister where the qubits might be swapped.
        inv_layout = layout.get_physical_bits()
        old_final_layout = self.property_set["final_layout"]
        # Swap the physical bits to the new layout
        to_swap = {
            physical: dag.find_bit(virtual).index for physical, virtual in old_final_layout.get_physical_bits().items()
        }
        # Add identity mappings for qubits that are not in the old final layout.
        to_swap.update({p: p for p in inv_layout if p not in to_swap})
        # Build the new final layout.
        return Layout({physical: inv_layout[to_swap[physical]] for physical in layout.get_physical_bits()})

    def _insert_move_gate(self, dag: DAGCircuit, layout: Layout) -> DAGCircuit:
        """Insert MOVE gates into the circuit using the IQM transpiler.

        Args:
            dag: The DAGCircuit to insert MOVE gates into; is not modified in place.
            layout: The physical to logical qubit layout at the start of the circuit.

        Returns:
            The updated DAGCircuit with MOVE gates inserted.

        """
        # Convert the DAG to a QuantumCircuit
        circuit = dag_to_circuit(dag)
        # Convert the circuit to the Circuit format
        iqm_circuit = Circuit(
            name="Transpiling Circuit",
            instructions=tuple(serialize_instructions(circuit, self.idx_to_component, overwrite_layout=layout)),
            metadata=None,
        )
        try:
            # Use the iqm-client transpiler to insert MOVE gates
            routed_iqm_circuit = transpile_insert_moves(
                iqm_circuit,
                self.architecture,
                existing_moves=self.existing_moves_handling,
            )
            # Turn the routed Circuit back into a Qiskit QuantumCircuit
            routed_circuit = deserialize_instructions(
                list(routed_iqm_circuit.instructions), self.component_to_idx, layout
            )
        except ValidationError as e:
            errors = e.errors()
            if (
                len(errors) == 1
                and errors[0]["msg"] == "Value error, Each circuit should have at least one instruction."
            ):  # Error because the Circuit without move gates is empty.
                routed_circuit = QuantumCircuit(
                    *layout.get_registers(), *(arg for arg in [circuit.num_ancillas, circuit.num_clbits] if arg > 0)
                )
            else:
                raise e

        # Create the new DAG and make sure that the qubits are properly ordered.
        return circuit_to_dag(
            routed_circuit,
            qubit_order=[layout.get_physical_bits()[i] for i in range(len(layout.get_physical_bits()))],
            clbit_order=routed_circuit.clbits,
        )


def _get_scheduling_method(
    perform_move_routing: bool,
    optimize_single_qubits: bool,
    remove_final_rzs: bool,
    ignore_barriers: bool,
    existing_moves_handling: ExistingMoveHandlingOptions | None,
) -> str:
    """Determine scheduling based on flags."""
    if perform_move_routing:
        if optimize_single_qubits:
            if not remove_final_rzs and ignore_barriers and existing_moves_handling is None:
                raise ValueError(
                    f"Move gate routing not compatible with {optimize_single_qubits=}, "
                    f"{remove_final_rzs=}, and {ignore_barriers=}."
                )
            if not remove_final_rzs:
                scheduling_method = "move_routing_exact_global_phase"
            elif ignore_barriers:
                scheduling_method = "move_routing_rz_optimization_ignores_barriers"
            else:
                scheduling_method = "move_routing"
        else:
            scheduling_method = "only_move_routing"
        if existing_moves_handling is not None:
            if not scheduling_method.endswith("routing"):
                raise ValueError(
                    "Existing Move handling options are not compatible with `remove_final_rzs` and \
                    `ignore_barriers` options."
                )  # No technical reason for this, just hard to maintain all combinations.
            scheduling_method += "_" + existing_moves_handling.value
    elif optimize_single_qubits:
        scheduling_method = "only_rz_optimization"
        if not remove_final_rzs:
            scheduling_method += "_exact_global_phase"
        if ignore_barriers:
            scheduling_method += "_ignore_barriers"
    else:
        scheduling_method = "default"
    return scheduling_method


def transpile_to_IQM(  # noqa: PLR0913
    circuit: QuantumCircuit,
    backend: IQMBackendBase,
    *,
    initial_layout: Layout | dict | list | None = None,
    perform_move_routing: bool = True,
    optimize_single_qubits: bool = True,
    ignore_barriers: bool = False,
    remove_final_rzs: bool = True,
    existing_moves_handling: ExistingMoveHandlingOptions | None = None,
    restrict_to_qubits: list[int] | list[str] | None = None,
    **qiskit_transpiler_kwargs,
) -> QuantumCircuit:
    """Customized transpilation to IQM backends.

    Works with both the Crystal and Star architectures.

    Note: When transpiling a circuit with MOVE gates, you might need to set ``optimization_level`` lower.
    If ``optimization_level`` is set too high, the transpiler might add single qubit gates onto the resonator,
    which is not supported by the IQM Star architectures. If this in undesired, it is best to have the transpiler
    add the MOVE gates automatically, rather than manually adding them to the circuit.

    Args:
        circuit: The circuit to transpile.
        backend: The backend to transpile to.
        initial_layout: The initial layout to use for the transpilation, same as :func:`~qiskit.compiler.transpile`.
        perform_move_routing: Whether to perform MOVE gate routing.
        optimize_single_qubits: Whether to optimize single qubit gates away.
        ignore_barriers: Whether to ignore barriers when optimizing single qubit gates away.
        remove_final_rzs: Whether to remove the final z rotations. It is recommended always to set this to true as
            the final RZ gates do no change the measurement outcomes of the circuit.
        existing_moves_handling: How to handle existing MOVE gates in the circuit, required if the circuit contains
            MOVE gates.
        restrict_to_qubits: Restrict the transpilation to only use these specific physical qubits. Note that you will
            also have to pass this information to :meth:`.IQMBackend.run` using the ``qubit_mapping`` parameter.
        qiskit_transpiler_kwargs: Arguments to be passed to the Qiskit transpiler.

    Returns:
        Transpiled circuit ready for running on the backend.

    """
    circuit_has_moves = circuit.count_ops().get("move", 0) > 0
    # get the target from the backend
    if circuit_has_moves:
        target = backend.target_with_resonators
        if perform_move_routing and existing_moves_handling is None:
            raise ValueError("The circuit contains MOVE gates but existing_moves_handling is not set.")
    else:
        target = backend.target

    if restrict_to_qubits is not None:
        # qubit names to qiskit indices
        restrict_to_qubits = [backend.qubit_name_to_index(q) if isinstance(q, str) else q for q in restrict_to_qubits]
        target = target.restrict_to_qubits(restrict_to_qubits)

    if circuit_has_moves and initial_layout is None:
        # Create a sensible initial layout if none is provided, since
        # the standard Qiskit transpile function does not do this well with resonators.
        real_target = backend.get_real_target()
        if restrict_to_qubits is not None:
            real_target = real_target.restrict_to_qubits(restrict_to_qubits)
        initial_layout = generate_initial_layout(real_target, circuit)

    # Determine which scheduling method to use
    scheduling_method = qiskit_transpiler_kwargs.pop("scheduling_method", None)
    if scheduling_method is None:
        scheduling_method = _get_scheduling_method(
            perform_move_routing=perform_move_routing,
            optimize_single_qubits=optimize_single_qubits,
            remove_final_rzs=remove_final_rzs,
            ignore_barriers=ignore_barriers,
            existing_moves_handling=existing_moves_handling,
        )
    else:
        warnings.warn(
            f"Scheduling method is set to {scheduling_method}, but it is normally used to pass other transpiler "
            + "options, ignoring the `perform_move_routing`, `optimize_single_qubits`, `remove_final_rzs`, "
            + "`ignore_barriers`, and `existing_moves_handling` arguments."
        )
    qiskit_transpiler_kwargs["scheduling_method"] = scheduling_method
    new_circuit = transpile(
        circuit,
        target=target,
        initial_layout=initial_layout,
        **qiskit_transpiler_kwargs,
    )
    return new_circuit
