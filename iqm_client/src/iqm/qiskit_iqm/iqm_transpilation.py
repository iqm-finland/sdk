# Copyright 2023 Qiskit on IQM developers
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
"""Transpilation tool to optimize the decomposition of single-qubit gates tailored to IQM hardware."""

import math
import warnings

import numpy as np
from packaging.version import Version
from qiskit import QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import RGate, RZGate, UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import (
    BasisTranslator,
    Optimize1qGatesDecomposition,
    RemoveBarriers,
)
from qiskit.transpiler.passmanager import PassManager

TOLERANCE = 1e-10  # The tolerance for equivalence checking against zero.


class IQMOptimizeSingleQubitGates(TransformationPass):
    r"""Optimize the decomposition of single-qubit gates for the IQM gate set.

    This optimization pass expects the circuit to be correctly layouted and translated to the IQM architecture
    and raises an error otherwise.
    The optimization logic follows the steps:

    1. Convert single-qubit gates to :math:`U` gates and combine all neighboring :math:`U` gates.
    2. Convert :math:`U` gates according to
       :math:`U(\theta , \phi , \lambda) = ~ RZ(\phi + \lambda) R(\theta, \pi / 2  - \lambda)`.
    3. Commute `RZ` gates to the end of the circuit using the fact that `RZ` and `CZ` gates commute, and
       :math:`R(\theta , \phi) RZ(\lambda) = RZ(\lambda) R(\theta, \phi - \lambda)`.
    4. Drop `RZ` gates immediately before measurements, and otherwise replace them according to
       :math:`RZ(\lambda) = R(\pi, \lambda / 2) R(- \pi, 0)`.

    Args:
        drop_final_rz: Drop terminal RZ gates even if there are no measurements following them (since they do not affect
            the measurement results). Note that this will change the unitary propagator of the circuit.
            It is recommended always to set this to true as the final RZ gates do no change the measurement outcomes of
            the circuit.
        ignore_barriers (bool): Removes the barriers from the circuit before optimization (default = False).

    """

    def __init__(self, drop_final_rz: bool = True, ignore_barriers: bool = False):
        super().__init__()
        self._basis = ["r", "cz", "move", "if_else"]
        self._intermediate_basis = ["u", "cz", "move", "if_else"]
        self._drop_final_rz = drop_final_rz
        self._ignore_barriers = ignore_barriers

    def run(self, dag: DAGCircuit, decompose_rz_to_r: bool = True) -> DAGCircuit:
        """Runs the single-qubit gate optimization pass.

        Args:
            dag: The input DAG circuit to optimize.
            decompose_rz_to_r: Whether to decompose RZ gates into R gates, or add the to the DAG as
                RZ gates. This is used in recursive calls to communicate the accumulated RZ angles in ``rz_angles``.

        Returns:
            The optimized DAG circuit.

        """
        if decompose_rz_to_r:
            self._validate_ops(dag)
        # accumulated RZ angles for each qubit, from the beginning of the circuit to the current gate
        rz_angles: list[float] = [0] * dag.num_qubits()

        # Handle old conditional gates
        if Version(qiskit_version) < Version("2.0"):
            # This needs to be done before the BasisTranslation as that pass does not retain the condition.
            dag = self._handle_c_if_blocks(dag)

        if self._ignore_barriers:
            dag = RemoveBarriers().run(dag)
        # convert all gates in the circuit to U and CZ gates
        dag = BasisTranslator(SessionEquivalenceLibrary, self._intermediate_basis).run(dag)
        # combine all sequential U gates into one
        dag = Optimize1qGatesDecomposition(self._intermediate_basis).run(dag)
        for node in dag.topological_op_nodes():
            if node.name == "u":
                dag, rz_angles = self._handle_u_gates(dag, node, rz_angles)
            elif node.name in {"measure", "reset"}:
                # measure and reset destroy phase information. The local phases before and after such
                # an operation are in principle independent, and the local computational frame phases
                # are arbitrary so we could set rz_angles to any values here, but zeroing the
                # angles results in fewest changes to the circuit.
                for qubit in node.qargs:
                    rz_angles[dag.find_bit(qubit).index] = 0
            elif node.name == "barrier":
                # TODO barriers are meant to restrict circuit optimization, so strictly speaking
                # we should output any accumulated ``rz_angles`` here as explicit z rotations (like
                # the final rz:s). However, ``rz_angles`` simply represents a choice of phases for the
                # local computational frames for the rest of the circuit (the initial part has already
                # been transformed). This choice of local phases is in principle arbitrary, so maybe it
                # makes no sense to convert it into active z rotations if we hit a barrier?
                pass
            elif node.name == "move":
                # acts like iSWAP with RZ, moving it to the other component
                qb, res = (
                    dag.find_bit(node.qargs[0]).index,
                    dag.find_bit(node.qargs[1]).index,
                )
                rz_angles[res], rz_angles[qb] = rz_angles[qb], rz_angles[res]
            elif node.name in {"cz", "delay"}:
                pass  # commutes with RZ gates
            elif node.name == "if_else":
                dag, rz_angles = self._handle_if_else_block(dag, node, rz_angles)
            else:
                raise ValueError(
                    f"Unexpected operation '{node.name}' in circuit given to IQMOptimizeSingleQubitGates pass"
                )

        if not decompose_rz_to_r:
            for qubit_index, rz_angle in enumerate(rz_angles):
                dag.apply_operation_back(RZGate(rz_angle), qargs=(dag.qubits[qubit_index],))
        elif not self._drop_final_rz:
            dag, rz_angles = self._apply_final_r_gates(dag, rz_angles)

        return dag

    def _apply_final_r_gates(self, dag: DAGCircuit, rz_angles: list[float]) -> tuple[DAGCircuit, list[float]]:
        """Helper function that adds the final PRX/R gates to the circuit according to the accumulated angles.

        Returns the updated dag and a list of zero angles since the final RZ rotations are already applied.

        Args:
            dag: The input DAG circuit we are optimizing.
            rz_angles: The accumulated RZ angles for each qubit.

        Returns:
            The updated DAG circuit and a list of zero angles.

        """
        for qubit_index, rz_angle in enumerate(rz_angles):
            if not math.isclose(rz_angle, 0, abs_tol=TOLERANCE):
                qubit = dag.qubits[qubit_index]
                dag.apply_operation_back(RGate(-np.pi, 0), qargs=(qubit,))
                dag.apply_operation_back(RGate(np.pi, rz_angle / 2), qargs=(qubit,))
        # Return resetted angles
        return dag, [0.0] * dag.num_qubits()

    def _handle_u_gates(
        self, dag: DAGCircuit, node: DAGOpNode, rz_angles: list[float]
    ) -> tuple[DAGCircuit, list[float]]:
        """Helper function that converts U gates to PRXs and RZ gates,
        so that the RZ gates can be commuted to the end of the circuit.

        Args:
            dag: The input DAG circuit we are optimizing.
            node: The DAG node containing the U gate to convert.
            rz_angles: The accumulated RZ angles for each qubit.

        Returns:
            The updated DAG circuit and the updated list of accumulated RZ angles.

        """
        qubit_index = dag.find_bit(node.qargs[0]).index
        if isinstance(node.op.params[0], float) and math.isclose(node.op.params[0], 0, abs_tol=TOLERANCE):
            dag.remove_op_node(node)
        else:
            dag.substitute_node(
                node,
                RGate(
                    node.op.params[0],
                    np.pi / 2 - node.op.params[2] - rz_angles[qubit_index],
                ),
            )
        phase = node.op.params[1] + node.op.params[2]
        dag.global_phase += phase / 2
        rz_angles[qubit_index] += phase
        return dag, rz_angles

    def _handle_if_else_block(
        self, dag: DAGCircuit, node: DAGOpNode, rz_angles: list[float]
    ) -> tuple[DAGCircuit, list[float]]:
        """Call the optimization recursively on both branches of the if_else node.

        The accumulated RZ angles are added to both branches before optimizing them.
        The accumulated RZ angles after the optimization are taken from the else branch
        and the adjoint is applied to the if branch to correct for the overrotation.

        Args:
            dag: The input DAG circuit we are optimizing.
            node: The DAG node containing the if_else block to optimize.
            rz_angles: The accumulated RZ angles for each qubit.

        Returns:
            The updated DAG circuit and the updated list of accumulated RZ angles.

        """
        # Add the Rz angles to each circuit block of the if_else node
        # and run this pass recursively
        sub_dags = []
        for circuit_block in node.op.params:
            new_circuit = QuantumCircuit(list(node.qargs + node.cargs))
            # Prepend Rz angle to circuit block
            for qubit in node.qargs:
                new_circuit.append(RGate(-np.pi, 0), [qubit])
                new_circuit.append(RGate(np.pi, rz_angles[dag.find_bit(qubit).index] / 2), [qubit])
            if circuit_block is not None:
                new_circuit.compose(circuit_block, node.qargs, node.cargs, inplace=True)
            # Run optimization pass on the block
            block_dag = circuit_to_dag(new_circuit)
            block_dag = self.run(block_dag, decompose_rz_to_r=False)
            sub_dags.append(block_dag)
        # Pick up the final rotation
        for qubit in node.qargs:
            # Find the last node on the qubit
            final_rzs = [list(block_dag.nodes_on_wire(qubit, only_ops=True))[-1] for block_dag in sub_dags]
            # Assertions because this cannot go wrong by user error
            assert len(final_rzs) == 2, "IfElseOp should have exactly two circuit blocks"
            assert final_rzs[0].name == "rz" and final_rzs[1].name == "rz", (
                "The last operation on each qubit in an IfElseOp should be an RZ gate, "
                + f"found {final_rzs[0].name} and {final_rzs[1].name} instead"
            )
            # Extract the angles
            rz1, rz2 = final_rzs[0].op.params[0], final_rzs[1].op.params[0]
            # We take the else_block rotation as the one to continue pushing through the circuit
            # because we don't support else_blocks in the circuit at the moment.
            # Update the rz_angle on this qubit with the one found
            rz_angles[dag.find_bit(qubit).index] = rz2
            # Remove the final rz from the dag in both circuit blocks
            for block_dag, final_node in zip(sub_dags, final_rzs):
                block_dag.remove_op_node(final_node)
            # Fix the overrotation of the if_block when the final Rz does not match
            if not math.isclose(rz1, rz2):
                rz_angle = rz1 - rz2
                sub_dags[0].apply_operation_back(RGate(-np.pi, 0), qargs=(qubit,))
                sub_dags[0].apply_operation_back(RGate(np.pi, rz_angle / 2), qargs=(qubit,))
        # Replace the params in the if_else node with the optimized circuits
        new_params = []
        for idx, sub_dag in enumerate(sub_dags):
            # Optimize the PRXs on the block_dag, but now keep the final Rzs
            block_dag = IQMOptimizeSingleQubitGates(drop_final_rz=False, ignore_barriers=self._ignore_barriers).run(
                sub_dag
            )
            # Ensure the qubits act on the same qubits as before
            if node.op.params[idx] is not None and block_dag.qubits != node.op.params[idx].qubits:
                # Sometimes the circuit_block.qubits != node.qargs,
                # so we need to make sure that they act on the same qubits as before
                new_circuit = QuantumCircuit(list(node.op.params[idx].qubits + node.op.params[idx].clbits))
                new_circuit.compose(
                    dag_to_circuit(block_dag),
                    node.op.params[idx].qubits,
                    node.op.params[idx].clbits,
                    inplace=True,
                )
            else:
                new_circuit = dag_to_circuit(block_dag)
            new_params.append(new_circuit)
        dag.substitute_node(
            node,
            IfElseOp(
                node.op.condition,
                new_params[0],
                false_body=new_params[1] if new_params[1].size() > 0 else None,
                label=node.op.label,
            ),
        )
        return dag, rz_angles

    def _handle_c_if_blocks(self, dag: DAGCircuit) -> DAGCircuit:
        """Helper function that replaces all classically controlled RGates with an if_else operator.

        This is needed because the BasisTranslator pass does not retain the condition on the nodes.
        This is only needed for Qiskit versions < 2.0.0.

        Args:
            dag: The input DAG circuit we are optimizing.

        Returns:
            The updated DAG circuit with if_else blocks instead of R gates with a condition.

        """
        for node in dag.topological_op_nodes():
            if hasattr(node, "condition") and node.condition and node.name != "if_else":
                # Manually parse the node to a circuit because helper functions don't exist
                # NOTE if_block needs to have the same size as node or else it cannot be replaced later.
                if_block = QuantumCircuit(list(node.qargs))
                # NOTE Need to reconstruct the node.op manually because rust panics when using node.op directly
                if node.op.name != "r":
                    raise ValueError(
                        f"Unexpected operation '{node.name}' in circuit given to IQMOptimizeSingleQubitGates pass"
                    )
                if_block.append(RGate(node.op.params[0], node.op.params[1], label=node.op.label), node.qargs)
                new_op = IfElseOp(
                    node.condition,
                    if_block,
                )
                dag.substitute_node(
                    node,
                    new_op,
                )
        return dag

    def _validate_ops(self, dag: DAGCircuit):  # noqa: ANN202
        """Helper function that validates that the operations in the circuit are compatible
        with the IQMOptimizeSingleQubitGates pass.

        Args:
            dag: The input DAG circuit to validate before optimization.

        Raises:
            ValueError: If an invalid operation is found in the circuit.

        """
        valid_ops = self._basis + ["measure", "reset", "delay", "barrier"]
        for node in dag.op_nodes():
            if node.name not in valid_ops:
                raise ValueError(
                    f"Invalid operation '{node.name}' found in IQMOptimize1QbDecomposition pass, "
                    + f"expected operations {valid_ops}"
                )


def optimize_single_qubit_gates(
    circuit: QuantumCircuit, drop_final_rz: bool = True, ignore_barriers: bool = False
) -> QuantumCircuit:
    """Optimize number of single-qubit gates in a transpiled circuit exploiting the IQM specific gate set.

    Args:
        circuit: quantum circuit to optimize
        drop_final_rz: Drop terminal RZ gates even if there are no measurements following them (since they do not affect
            the measurement results). Note that this will change the unitary propagator of the circuit.
            It is recommended always to set this to true as the final RZ gates do no change the measurement outcomes of
            the circuit.
        ignore_barriers (bool): Removes barriers from the circuit if they exist (default = False) before optimization.

    Returns:
        optimized circuit

    """
    warnings.warn(
        DeprecationWarning(
            "This function is deprecated and will be removed in a later version of `iqm.qiskit_iqm`. "
            + "Single qubit gate optimization is now automatically applied when running `qiskit.transpile()` on any "
            + "IQM device. If you want to have more fine grained control over the optimization, please use the "
            + "`iqm.qiskit_iqm.transpile_to_IQM` function."
        )
    )
    # Code not updated to use transpile_to_IQM due to circular imports
    new_circuit = PassManager(IQMOptimizeSingleQubitGates(drop_final_rz, ignore_barriers)).run(circuit)
    new_circuit._layout = circuit.layout
    return new_circuit


class IQMReplaceGateWithUnitaryPass(TransformationPass):
    """Transpiler pass that replaces all gates with given name in a circuit with a UnitaryGate.

    Args:
        gate: The name of the gate to replace.
        unitary: The unitary matrix to replace the gate with.

    """

    def __init__(self, gate: str, unitary: list[list[float]]):
        super().__init__()
        self.gate = gate
        self.unitary = unitary

    def run(self, dag):  # noqa: ANN001, ANN201
        for node in dag.op_nodes():
            if node.name == self.gate:
                dag.substitute_node(node, UnitaryGate(self.unitary))
        return dag
