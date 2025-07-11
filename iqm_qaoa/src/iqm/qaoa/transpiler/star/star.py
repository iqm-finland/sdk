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
"""The module for the router for the star QPU."""

import copy as cp

from dimod import BinaryQuadraticModel, to_networkx_graph
from dimod.typing import Variable
from iqm.qaoa.transpiler.quantum_hardware import HardQubit, StarQPU
from iqm.qaoa.transpiler.routing import Mapping, Routing
from iqm.qiskit_iqm.move_gate import MoveGate
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit


class RoutingStar(Routing):
    """This class represents a routing of a QAOA phase separator on the star topology.

    The main difference from the parent class :class:`~iqm.qaoa.transpiler.routing.Routing` is that :class:`RoutingStar`
    doesn't use :class:`~iqm.qaoa.transpiler.routing.Layer` for its layers. The layers in :class:`RoutingStar` are much
    simpler (just one gate in each layer). Here a layer is just a tuple containing a string describing the gate and an
    integer describing the :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` involved.

    Args:
        problem_bqm: The optimization problem represented as :class:`~dimod.BinaryQuadraticModel`.
        qpu: The QPU representing the hardware qubit topology.
        initial_mapping: The starting mapping of the logical-to-hardware qubits.

    """

    def __init__(self, problem_bqm: BinaryQuadraticModel, qpu: StarQPU, initial_mapping: Mapping | None = None) -> None:
        super().__init__(problem_bqm, qpu, initial_mapping)
        # For star QPU, each layer contains only one operation, so no need to use a sophisticated ``Layer`` class
        self.layers: list[tuple[str, HardQubit]] = []  # type: ignore[assignment]

    def apply_move_in(self, qubit: HardQubit) -> None:
        """Apply move gate (to move a qubit into the resonator).

        Args:
            qubit: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` whose logical qubit is moved to
                the central resonator.

        Raises:
            ValueError: If the central resonator is already occupied by a logical qubit (so a different logical qubit
                can't be moved there).
            ValueError: If the target qubit doesn't contain a logical qubit.

        """
        if 0 in self.mapping.hard2log:
            raise ValueError("The central resonator is occupied, another qubit can't be moved in.")
        if qubit not in self.mapping.hard2log:
            raise ValueError("The target qubit is not assigned a logical qubit.")

        self.layers.append(("move_in", qubit))
        self.mapping.move_hard(source_qubit=qubit, target_qubit=0)

    def apply_move_out(self, qubit: HardQubit) -> None:
        """Apply move gate (to move a qubit out of the resonator).

        Args:
            qubit: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` where the logical qubit from
                the resonator is moved to.

        Raises:
            ValueError: If the central resonator is empty, so nothing can't be moved out of it.
            ValueError: If the target qubit is already occupied by a different logical qubit.

        """
        if 0 not in self.mapping.hard2log:
            raise ValueError("The central resonator is empty, a qubit can't be moved out of it.")
        if qubit in self.mapping.hard2log:
            raise ValueError("The target qubit is already occupied by a logical qubit.")

        self.layers.append(("move_out", qubit))
        self.mapping.move_hard(source_qubit=0, target_qubit=qubit)

    def apply_directed_int(self, target: HardQubit) -> None:
        """Apply interaction between the resonator and an outer qubit.

        The resonator doesn't support single-qubit gates, so the interaction has to be decomposed in such a way that
        the single-qubit rotation is applied to the outer ``target`` qubit.

        Args:
            target: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` that interacts with the central qubit.

        Raises:
            ValueError: If the resonator is empty (so no interaction can be done).
            ValueError: If the target qubit doesn't correspond to any logical qubit.

        """
        if 0 not in self.mapping.hard2log:
            raise ValueError("The central resonator is empty, interaction can't be applied.")
        if target not in self.mapping.hard2log:
            raise ValueError("The target qubit is 'empty', interaction can't be applied.")

        self.layers.append(("int", target))
        log_qb0, log_qb1 = self.mapping.hard2log[0], self.mapping.hard2log[target]
        self.remaining_interactions.remove_edge(log_qb0, log_qb1)

    def count_move_gates(self) -> int:
        """Counts the number of move gates in the routing.

        Counts both 'move_in' and 'move_out' gates in the routing. In theory, each 'move_in' gate should be followed by
        a corresponding 'move_out' gate, but this counts the gates separately because there might be as of yet
        unforseen situations in which this won't be true anymore.

        Returns:
            The number of 'move_in' and 'move_out' gates in the entire routing.

        """
        layers: list[tuple[str, HardQubit]] = self.layers
        number_of_moves_in_layers = 0
        for gate in layers:
            # Pylint is being dumb here, ``gate`` is definitely a tuple and therefore subscriptable.
            if gate[0] in {"move_in", "move_out"}:  # pylint: disable=unsubscriptable-object
                number_of_moves_in_layers += 1

        return number_of_moves_in_layers

    # The following function builds the circuit of the QAOA. It also doesn't make much sense to split it up into smaller
    # functions, so that's why the pylint warning is disabled.
    # pylint: disable=too-many-locals
    def build_qiskit(self, betas: list[float], gammas: list[float]) -> QuantumCircuit:
        """Build the entire QAOA circuit in :mod:`qiskit`.

        The :class:`~iqm.qaoa.transpiler.star.star.RoutingStar` object contains all the information needed to create
        the phase separator part of the QAOA circuit. This method builds the rest of the circuit from it, i.e.:

        1. It initializes the qubits in the :math:`| + >` state by applying the Hadamard gate to all of them.
        2. It goes through the routing layers and applies the corresponding interactions (or move gates).
        3. It applies local fields.
        4. It applies the driver.
        5. It repeats steps 2-4 until it uses up all ``betas`` and ``gammas``.
        6. It applies the measurements and a barrier before them.

        Args:
            betas: The QAOA parameters to be used in the driver (*RX* gate).
            gammas: The QAOA parameters to be used in the phase separator (*RZ* and *RZZ* gates).

        Returns:
            A complete QAOA :class:`~qiskit.circuit.QuantumCircuit`.

        """
        if len(betas) != len(gammas):
            raise ValueError("The lengths of ``gammas`` and ``betas`` need to be the same!")

        layers = cp.deepcopy(self.layers)
        # The mapping to be used throughout the circuit construction. It begins identical to `self.initial_mapping`.
        mapping = cp.deepcopy(self.initial_mapping)
        qb_register = sorted(self.qpu.qubits)

        qiskit_circ = QuantumCircuit(len(qb_register), len(mapping.hard2log))

        # Prepare uniform superposition.
        qiskit_circ.h(mapping.hard2log.keys())

        for gamma, beta in zip(gammas, betas):  # Each pair of ``gamma, beta`` corresponds to one QAOA layer.
            # Apply phase separator.
            for op_type, qubit in layers:
                if op_type == "int":
                    log_qb0 = mapping.hard2log[0]
                    log_qb1 = mapping.hard2log[qubit]
                    weight = self.problem.get_quadratic(log_qb0, log_qb1)
                    if weight != 0:
                        # Decompose the RZZ gate into RZ sandwitched between two CNOTs
                        qiskit_circ.cx(qb_register.index(0), qb_register.index(qubit))
                        qiskit_circ.rz(2 * gamma * weight, qb_register.index(qubit))
                        qiskit_circ.cx(qb_register.index(0), qb_register.index(qubit))

                # Don't confuse IQM's MoveGate with qiskit's Move. MoveGate always takes qubit and resonator as input,
                # regardless of whether it's move in or move out.
                if op_type == "move_in":
                    qiskit_circ.append(MoveGate(), [qubit, 0])
                    mapping.move_hard(source_qubit=qubit, target_qubit=0)
                if op_type == "move_out":
                    qiskit_circ.append(MoveGate(), [qubit, 0])
                    mapping.move_hard(source_qubit=0, target_qubit=qubit)

            for hard_qb in mapping.hard2log.keys():
                log_qb = mapping.hard2log[hard_qb]
                local_field = self.problem.get_linear(log_qb)
                qiskit_circ.rz(2 * gamma * local_field, qb_register.index(hard_qb))

            # The list of layers is reversed to go through it in the opposite order in the next QAOA layer.
            layers.reverse()
            # On top of reversing the order of layers, we need to replace 'move_in' with 'move_out' and vice versa.
            layers = [
                (op_type.replace("_out", "_in") if op_type.endswith("_out") else op_type.replace("_in", "_out"), qubit)
                for op_type, qubit in layers
            ]

            # Apply driver.
            qiskit_circ.rx(2 * beta, mapping.hard2log.keys())

        qiskit_circ.barrier()
        qiskit_circ.measure(mapping.hard2log.keys(), range(len(mapping.hard2log)))

        return qiskit_circ

    # The draw method is very similar to the ``draw`` method of ``Routing``, so pylint raises a 'duplicate-code' error.
    # pylint: disable=duplicate-code
    def draw(self) -> None:
        """Plot all layers of the routing in batches of 9.

        This creates a series of plots that are shown on the screen. Each plot contains 9 layers arranged in a 3x3 grid.
        Each layer is drawn using a custom procedure. It has the shape of the QPU topology with edges colored based on
        what is happening on them in the given layer.

        - Teal highlight if a 'move_in' gate is applied.
        - Plum highlight if a 'move_out' gate is applied.
        - Green highlight if an 'int' gate is applied.
        - No highlight (black) if nothing is happening along the edge.

        Raises:
            ValueError: If the first element of the layers is neither of 'int', 'move_in' or 'move_out'.

        """
        layer_count = len(self.layers)

        gate_to_color = {
            "int": "g",
            "move_in": "teal",
            "move_out": "plum",
        }

        if layer_count > 1:
            layer_batches = [self.layers[x : x + 9] for x in range(0, len(self.layers), 9)]
            # Throughout the plotting we keep track of the mapping.
            # It is used to label the :class:`HardQubit`\s with the corresponding :class:`LogQubit` label.
            mapping = cp.deepcopy(self.initial_mapping)
            layer_index = 0
            for layers in layer_batches:
                _, axs = plt.subplots(3, 3)
                for layer in layers:
                    row = (layer_index % 9) // 3
                    column = (layer_index % 9) % 3

                    self.qpu.draw(
                        gate_lists={gate_to_color[layer[0]]: [(0, layer[1])]},  # type: ignore[list-item]
                        ax=axs[row, column],
                        mapping=mapping,
                        show=False,
                    )

                    axs[row, column].set_axis_off()
                    axs[row, column].autoscale_view()
                    axs[row, column].set_title(f"Layer {layer_index}")

                    if layer[0] == "move_in":
                        mapping.move_hard(layer[1], 0)
                    elif layer[0] == "move_out":
                        mapping.move_hard(0, layer[1])

                    layer_index += 1
                plt.show()
        else:
            self.qpu.draw(
                gate_lists={gate_to_color[self.layers[0][0]]: [(0, self.layers[0][1])]} if self.layers else None,  # type: ignore[list-item]
                ax=None,
                mapping=self.initial_mapping,
                show=True,
            )


def star_router(problem_bqm: BinaryQuadraticModel, qpu: StarQPU) -> RoutingStar:
    """The function that implements the optimal swapping strategy on the star QPU.

    Looks for the minimum vertex cover set on the problem graph and then swaps the nodes belonging to this cover into
    the center of the star (adding the interaction gates in between).

    Args:
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` description of the problem, necessary to create
            an instance of :class:`~iqm.qaoa.transpiler.star.star.RoutingStar`.
        qpu: The QPU, necessary to create an instance of :class:`~iqm.qaoa.transpiler.star.star.RoutingStar`.

    Returns:
        A :class:`~iqm.qaoa.transpiler.star.star.RoutingStar` object containing the optimal star swapping strategy.

    """
    bqm = problem_bqm.copy()

    # We need to create a partial initial mapping that leaves the central resonator unassigned.
    available_hard_qubits = qpu.qubits - {0}  # The central resonator isn't available as a qubit for initial mapping.
    partial_initial_mapping = dict(zip(available_hard_qubits, bqm.variables))
    initial_mapping = Mapping(qpu, bqm, partial_initial_mapping)
    route = RoutingStar(bqm, qpu, initial_mapping)

    problem_graph = to_networkx_graph(bqm)  # The nodes of the graph have type ``Variable``
    independent_sets = nx.find_cliques(nx.complement(problem_graph))
    max_independent_set: list[Variable] = max(independent_sets, key=len, default=[])
    min_vertex_cover = set(problem_graph.nodes()) - set(max_independent_set)

    for node in min_vertex_cover:
        hard_qb = route.mapping.log2hard[node]
        route.apply_move_in(hard_qb)

        # Create list before iterating to avoid "RuntimeError: dictionary changed size during iteration"
        for neigh_var in list(route.remaining_interactions.neighbors(node)):
            # The interaction is applied between the resonator and the neighboring variables.
            route.apply_directed_int(route.mapping.log2hard[neigh_var])

        route.apply_move_out(hard_qb)

    return route
