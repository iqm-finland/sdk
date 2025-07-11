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
"""This module contains the object classes :class:`~iqm.qaoa.transpiler.routing.Mapping`,
:class:`~iqm.qaoa.transpiler.routing.Layer` and :class:`~iqm.qaoa.transpiler.routing.Routing` to be used throughout any
transpilation algorithm.
"""

from __future__ import annotations

import copy as cp
import warnings

from dimod import BINARY, SPIN, BinaryQuadraticModel, to_networkx_graph
from iqm.qaoa.transpiler.quantum_hardware import QPU, HardEdge, HardQubit, LogEdge, LogQubit
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit


class Mapping:
    """This class is responsible for a mapping between logical and hardware qubits.

    It maintains two dictionaries: :attr:`log2hard` and :attr:`hard2log` which are mappings between logical
    and hardware qubits. They are automatically kept in sync. The names for the hardware and logical qubits are
    extracted from ``qpu`` and ``problem_bqm`` at initialization.

    Args:
        qpu: a :class:`~iqm.qaoa.transpiler.quantum_hardware.QPU` object describing the topology of the QPU, used to
            get hardware qubits.
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` of the problem we're trying to solve, used to get
            logical qubits.
        partial_initial_mapping: An optional dictionary that contains a partial mapping to use as a starting point.
            The keys should be :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` and the values
            :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`.

    Raises:
        ValueError: If ``partial_initial_mapping`` is provided, but it's not bijective.

    """

    def __init__(
        self,
        qpu: QPU,
        problem_bqm: BinaryQuadraticModel,
        partial_initial_mapping: dict[HardQubit, LogQubit] | None = None,
    ) -> None:
        if problem_bqm.vartype is BINARY:
            problem_bqm.change_vartype(SPIN)

        # Take the variables from ``problem_bqm`` and ``qpu``.
        self.log_qbs = set(problem_bqm.variables)
        self.hard_qbs = qpu.qubits

        if len(self.hard_qbs) != len(self.log_qbs):
            warnings.warn("The QPU has more qubits than the problem has variables. Some QPU qubits will not be used.")

        # If no partial initial mapping is provided, just map the qubits to each other arbitrarily.
        if partial_initial_mapping is None:
            self._hard2log = dict(zip(self.hard_qbs, self.log_qbs))

        # If a partial inital mapping is provided, use it.
        else:
            if len(set(partial_initial_mapping.values())) != len(partial_initial_mapping.values()):
                raise ValueError("The initial mapping between hardware and logical qubits is not bijective!")
            if len(partial_initial_mapping) < len(self.log_qbs):
                remaining_hard_qbs = self.hard_qbs - set(partial_initial_mapping.keys())
                remaining_log_qbs = self.log_qbs - set(partial_initial_mapping.values())
                initial_mapping = partial_initial_mapping
                # The qubits not covered by the partial inital mapping get mapped arbitrarily.
                for hard_qb, log_qb in zip(remaining_hard_qbs, remaining_log_qbs):
                    initial_mapping[hard_qb] = log_qb  # type: ignore[assignment]
            else:
                initial_mapping = partial_initial_mapping

            self._hard2log = initial_mapping  # type: ignore[assignment]

    @property
    def hard2log(self) -> dict[HardQubit, LogQubit]:
        """The dictionary containing the mapping from hardware qubits to logical qubits."""
        return self._hard2log  # type: ignore[return-value]

    @property
    def log2hard(self) -> dict[LogQubit, HardQubit]:
        """The dictionary :attr:`log2hard` is calculated lazily from :attr:`hard2log`."""
        return {log_qb: hard_qb for hard_qb, log_qb in self._hard2log.items()}  # type: ignore[misc]

    def swap_log(self, gate: LogEdge) -> None:
        """Swap association between a pair of logical qubits.

        Updates the dictionary :attr:`hard2log` (:attr:`log2hard` gets updated automatically).

        Args:
            gate: The pair of logical qubits to swap.

        """
        qb0, qb1 = gate
        hard_qb0 = self.log2hard[qb0]
        hard_qb1 = self.log2hard[qb1]
        self._hard2log[hard_qb0], self._hard2log[hard_qb1] = qb1, qb0

    def swap_hard(self, gate: HardEdge) -> None:
        """Swap association between a pair of hardware qubits.

        Updates the dictionary :attr:`hard2log` (:attr:`log2hard` gets updated automatically).

        Args:
            gate: The pair of hardware qubits to swap.

        """
        qb0, qb1 = gate
        self._hard2log[qb0], self._hard2log[qb1] = self._hard2log[qb1], self._hard2log[qb0]

    def move_hard(self, source_qubit: HardQubit, target_qubit: HardQubit) -> None:
        """Moves a logical qubit from a one hardware qubit to a different hardware qubit on the QPU which is not part of
        the mapping.

        Updates the dictionary :attr:`hard2log` (:attr:`log2hard` gets updated automatically). The dictionary is
        changed as follows:

        * If the dictionary :attr:`hard2log` has a key ``source_qubit`` (but not ``target_qubit``), this method removes
          the key ``source_qubit``, creates a new key ``target_qubit`` and gives it the value formerly associated to
          ``source_qubit``

        * The dictionary :attr:`log2hard` is modified correspondingly. The value ``source_qubit`` is changed to
          ``target_qubit``.

        Args:
            source_qubit: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` whose
                :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit` is being moved.
            target_qubit: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` where
                the :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit` is being moved.

        Raises:
            ValueError: If the ``target_qubit`` is already assigned to a different logical qubit.

        """
        if target_qubit in self._hard2log:
            raise ValueError(
                f"The target qubit {target_qubit} is already occupied by a logical qubit "
                f"{self._hard2log[target_qubit]}."
            )
        corresponding_log_qb = self._hard2log[source_qubit]

        # Modify ``self._hard2log``
        self._hard2log[target_qubit] = corresponding_log_qb
        del self._hard2log[source_qubit]

    def update(self, layer: "Layer") -> None:
        """Convenience function that updates the mapping based on the swap gates found in
        a :class:`~iqm.qaoa.transpiler.routing.Layer` object.

        Iterates over the gates in a :class:`~iqm.qaoa.transpiler.routing.Layer` object and swaps the hardware qubits
        corresponding to swap gates.

        Args:
            layer: The layer whose swap gates are used.

        """
        for hard_qb0, hard_qb1 in layer.gates.edges():
            if layer.gates[hard_qb0][hard_qb1]["swap"]:
                self.swap_hard(frozenset((hard_qb0, hard_qb1)))


class Layer:
    """A class describing one layer of the QAOA phase separator, consisting of swap and interaction gates.

    The class knows about the QPU topology (from ``qpu``) and uses it to decide which gates are applicable.
    A :class:`Layer` object contains an internal copy of the QPU graph
    :attr:`iqm.qaoa.transpiler.quantum_hardware.QPU.hardware_graph`, but with edges labelled based on whether
    an interaction or a swap occurs along that edge in this layer. Similarly, the nodes are labelled based on whether
    they're "occupied" in the present layer.

    Args:
        qpu: A ``QPU`` object, containing the underlying QPU topology.
        int_gates: A set of :class:`~iqm.qaoa.transpiler.quantum_hardware.HardEdge` interaction gates to be implemented
            in the layer. Further interaction gates may be added to the layer by using :meth:`apply_int_gate`.
        swap_gates: A set of :class:`~iqm.qaoa.transpiler.quantum_hardware.HardEdge` swap gates to be implemented in
            the layer. Further swap gates may be added to the layer by using :meth:`apply_swap_gate`.

    """

    def __init__(
        self, qpu: QPU, int_gates: set[HardEdge] | None = None, swap_gates: set[HardEdge] | None = None
    ) -> None:
        int_gates = int_gates or set()  # If ``int_gates`` is not given, it is instantiatied as an empty set
        swap_gates = swap_gates or set()  # If ``swap_gates`` is not given, it is instantiatied as an empty set
        self.qpu = qpu
        self.gates = nx.Graph()  # type: ignore[var-annotated]
        for hard_qb0, hard_qb1 in self.qpu.hardware_graph.edges():
            self.gates.add_edge(hard_qb0, hard_qb1, swap=False, int=False)
            self.gates.nodes[hard_qb0]["blocked"] = False
            self.gates.nodes[hard_qb1]["blocked"] = False

        for gate in int_gates:
            self.apply_int_gate(gate)

        for gate in swap_gates:
            self.apply_swap_gate(gate)

    def _qbs_not_involved_in_other_gate(self, gate: HardEdge) -> bool:
        """Are the two qubits involved in the proposed gate not already involved in other gates?"""
        hard_qb0, hard_qb1 = gate

        return not (self.gates.nodes[hard_qb0]["blocked"] or self.gates.nodes[hard_qb1]["blocked"])

    def int_gate_applicable(self, gate: HardEdge) -> bool:
        """Can the proposed interaction gate be executed within the given layer?

        Goes through a few checks:

        - If the required connection doesn't exist in the QPU, return ``False``.
        - If there is already a gate applied between these qubits, and it's the swap gate, return ``True`` since
          the interaction gate can be combined with it. If it's not a simple swap gate, return ``False``.
        - Otherwise, check if either of the qubits is involved in other gates and return the outcome of that.

        Args:
            gate: The pair of qubits for which we're checking the applicability of an interaction gate.

        """
        if not self.qpu.has_edge(gate):
            return False

        hard_qb0, hard_qb1 = gate
        # If there is already an interaction gate, we can't apply another one.
        if self.gates[hard_qb0][hard_qb1]["int"]:
            return False
        # If there is only a swap gate, we can apply an interaction gate over it.
        if self.gates[hard_qb0][hard_qb1]["swap"]:
            return True
        return self._qbs_not_involved_in_other_gate(gate)

    def apply_int_gate(self, gate: HardEdge) -> None:
        """Apply an interaction gate if it is applicable within the given layer.

        Args:
            gate: The pair of qubits between which we apply the interaction gate.

        Raises:
            ValueError: If for whatever reason the interaction gate cannot be applied in this layer.

        """
        if self.int_gate_applicable(gate):
            hard_qb0, hard_qb1 = gate
            self.gates[hard_qb0][hard_qb1]["int"] = True
            self.gates.nodes[hard_qb0]["blocked"] = True
            self.gates.nodes[hard_qb1]["blocked"] = True
        else:
            raise ValueError(f"Interaction gate {gate} cannot be applied in layer")

    def swap_gate_applicable(self, gate: HardEdge) -> bool:
        """Can the proposed swap gate be executed within the given layer?

        Goes through a few checks:

        - If the required connection doesn't exist in the QPU, return ``False``.
        - If there is already a swap gate between these qubits, return ``True`` (since the new swap gate can cancel
          it).
        - If there is already an interaction gate between these qubits, return ``True`` (since the swap gate can
          combine with it).
        - Otherwise, check if either of the qubits is involved in other gates.

        Args:
            gate: The pair of qubits for which we're checking the applicability of a swap gate.

        """
        if not self.qpu.has_edge(gate):
            return False

        hard_qb0, hard_qb1 = gate
        # If there is either a swap or an interaction gate, we can apply a swap (potentially undoing the previous swap).
        if self.gates[hard_qb0][hard_qb1]["swap"] or self.gates[hard_qb0][hard_qb1]["int"]:
            return True
        return self._qbs_not_involved_in_other_gate(gate)

    def apply_swap_gate(self, gate: HardEdge) -> None:
        """Apply swap gate if it is applicable within the given layer.

        Args:
            gate: The pair of qubits between which we apply the swap gate.

        Raises:
            ValueError: If for whatever reason the swap gate cannot be applied in this layer.

        """
        if self.swap_gate_applicable(gate):
            hard_qb0, hard_qb1 = gate

            # Change the "swap" status (add a swap if there is no swap and remove a swap if it is already there)
            self.gates[hard_qb0][hard_qb1]["swap"] = not self.gates[hard_qb0][hard_qb1]["swap"]

            # If there IS NOT an interaction gate on these qubits, we change their "blocked" status.
            # If there IS an interaction gate on these qubits, they should remain "blocked".
            if not self.gates[hard_qb0][hard_qb1]["int"]:
                self.gates.nodes[hard_qb0]["blocked"] = not self.gates.nodes[hard_qb0]["blocked"]
                self.gates.nodes[hard_qb1]["blocked"] = not self.gates.nodes[hard_qb1]["blocked"]
        else:
            raise ValueError(f"Swap gate {gate} cannot be applied in layer")

    def draw(self, mapping: Mapping | None = None, ax: Axes | None = None, show: bool = True) -> None:
        """Plot a sketch of the QPU, coloring the physical couplers based on the gate applied.

        - Yellow highlight if a combination of swap and int is applied.
        - Blue highlight if a swap gate is applied.
        - Green highlight if an interaction gate is applied.
        - No highlight (black) if nothing is happening along the edge.

        The labels for the hardware qubits in the plot are the names of the associated logical qubits.

        Args:
            mapping: The :class:`~iqm.qaoa.transpiler.routing.Mapping` object used.
            ax: :class:`matplotlib.axes.Axes` object to specify where to draw the picture.
            show: Boolean to specift if the plot is to be shown (or e.g., processed somehow).

        """
        gate_lists: dict[str, list[tuple[HardQubit, HardQubit]]] = {"y": [], "b": [], "g": []}
        for hard_qb0, hard_qb1 in self.gates.edges():
            swap_b, int_b = self.gates[hard_qb0][hard_qb1]["swap"], self.gates[hard_qb0][hard_qb1]["int"]
            if swap_b and int_b:
                gate_lists["y"].append((hard_qb0, hard_qb1))
            elif swap_b:
                gate_lists["b"].append((hard_qb0, hard_qb1))
            elif int_b:
                gate_lists["g"].append((hard_qb0, hard_qb1))

        if mapping is None:
            self.qpu.draw(gate_lists=gate_lists, ax=ax, show=show)  # type: ignore[arg-type]
        else:
            self.qpu.draw(gate_lists=gate_lists, ax=ax, mapping=mapping, show=show)  # type: ignore[arg-type]


class Routing:
    """This class represents a routing of a QAOA phase separator.

    A :class:`~iqm.qaoa.transpiler.routing.Routing` object is intended to be directly used by a router during routing
    (any router). To that end it maintains a list of :class:`~iqm.qaoa.transpiler.routing.Layer` objects,
    a :class:`~networkx.Graph` with the interactions not implemented yet and
    a :class:`~iqm.qaoa.transpiler.routing.Mapping` object that represents the current status of mapping between
    hardware and logical qubits.

    A router interacts with a :class:`~iqm.qaoa.transpiler.routing.Routing` object by using the methods
    :meth:`apply_swap` and :meth:`apply_int`. Optionally also :meth:`attempt_apply_int`. If the problem BQM contains
    interactions of strength 0 (e.g., because of padding), those won't be added into the list of layers. When the method
    :meth:`apply_int` is called on those interactions, it is skipped.

    Args:
        problem_bqm: The optimization problem represented as :class:`~dimod.BinaryQuadraticModel`.
        qpu: The QPU representing the hardware qubit topology.
        initial_mapping: The starting mapping of the logical-to-hardware qubits.

    """

    def __init__(self, problem_bqm: BinaryQuadraticModel, qpu: QPU, initial_mapping: Mapping | None = None) -> None:
        self.problem = problem_bqm
        # The variable :meth:`remaining_interactions` keeps track of all interactions remaining to be executed.
        # So at the beginning it's equal to all of the iteractions in ``problem_bqm``.
        self.remaining_interactions = to_networkx_graph(problem_bqm)

        self.qpu = qpu
        if initial_mapping is None:
            self.initial_mapping = Mapping(self.qpu, self.problem)
        else:
            self.initial_mapping = initial_mapping

        self.mapping = cp.deepcopy(self.initial_mapping)
        self.layers = [Layer(self.qpu)]

    @property
    def active_subgraph(self) -> nx.Graph:
        """The topology of the QPU that is being used in the routing."""
        return nx.subgraph(self.qpu.hardware_graph, self.mapping.hard2log.keys())

    # pylint: disable=anomalous-backslash-in-string
    def apply_swap(self, gate: HardEdge, attempt_int: bool = False) -> None:
        r"""Apply swap gate at the earliest possible :class:`~iqm.qaoa.transpiler.routing.Layer`, add a new layer if
        needed.

        Goes through the existing :class:`~iqm.qaoa.transpiler.routing.Layer`\s from the end and tries to apply a swap
        gate between the qubits defined in ``gate`` at the earliest possible
        :class:`~iqm.qaoa.transpiler.routing.Layer`. That means, as early as possible without crossing any other swap or
        interaction acting on the same :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit`\s.

        Args:
            gate: An edge between two :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit`\s where the swap should
                be applied.
            attempt_int: Boolean saying whether an interaction gate should be combined with the swap.

        Raises:
            ValueError: If there is no edge connecting the two hardware qubits in ``gate`` on the hardware graph.

        """
        if not self.qpu.has_edge(gate):
            raise ValueError(f"SWAP gate on hardware qubits {gate} not supported on hardware graph.")

        def _internal_apply_swap(layer_index: int) -> None:
            """An internal function that applies the swap gate in the :class:`~iqm.qaoa.transpiler.routing.Layer`
            defined by the index ``layer_index``.
            """
            # Apply the swap gate in the correct :class:`~iqm.qaoa.transpiler.routing.Layer`.
            self.layers[layer_index].apply_swap_gate(gate)
            # Update the :class:`~iqm.qaoa.transpiler.routing.Mapping`.
            self.mapping.swap_hard(gate)
            if attempt_int:
                hard_qb0, hard_qb1 = gate
                log_qb0, log_qb1 = self.mapping.hard2log[hard_qb0], self.mapping.hard2log[hard_qb1]
                if self.remaining_interactions.has_edge(log_qb0, log_qb1):
                    self.apply_int(gate)

        if not self.layers[-1].swap_gate_applicable(gate):
            self.layers.append(Layer(self.qpu))
            _internal_apply_swap(-1)
        elif len(self.layers) == 1:
            _internal_apply_swap(-1)
        else:
            for layer_index in range(len(self.layers) - 1, 0, -1):
                if not self.layers[layer_index - 1].swap_gate_applicable(gate):
                    _internal_apply_swap(layer_index)
                    break
                if layer_index == 1:
                    _internal_apply_swap(0)

    # pylint: disable=anomalous-backslash-in-string
    def apply_int(self, gate: HardEdge) -> None:
        r"""Apply interaction gate at the earliest possible :class:`~iqm.qaoa.transpiler.routing.Layer`, add a new layer
        if necessary.

        Goes through the existing :class:`~iqm.qaoa.transpiler.routing.Layer`\s from the end and tries to apply
        an interaction gate between the qubits defined in ``gate`` at the earliest possible
        :class:`~iqm.qaoa.transpiler.routing.Layer`. That means, as early as possible without crossing any other swap or
        interaction acting on the same :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit`\s. If an interaction has
        strength 0, it isn't added!

        Args:
            gate: An edge between two :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit`\s where the interaction
                should be applied.

        Raises:
            ValueError: If there is no edge connecting the two hardware qubits in ``gate`` on the hardware graph.
            ValueError: If there is no interaction to be applied between the two corresponding logical qubits.

        """
        hard_qb0, hard_qb1 = gate

        if not self.qpu.has_edge(gate):
            raise ValueError(f"Interaction gate on hardware qubits {gate} not supported on hardware graph.")
        log_qb0, log_qb1 = self.mapping.hard2log[hard_qb0], self.mapping.hard2log[hard_qb1]

        if self.problem.get_quadratic(log_qb0, log_qb1) == 0:
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)
            return  # If the interaction strength is 0, don't add any interaction to the routing.

        if not self.remaining_interactions.has_edge(log_qb0, log_qb1):
            raise ValueError(
                f"interaction gate between hardware qubits {hard_qb0} and {hard_qb1}, i.e., "
                f"logical qubits {log_qb0} and {log_qb1} does not process any remaining interaction"
            )

        # If it's not possible to apply in the latest layer, add a new layer.
        if not self.layers[-1].int_gate_applicable(gate):
            self.layers.append(Layer(self.qpu))
            self.layers[-1].apply_int_gate(gate)
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)
        # If there is only a single layer, apply the interaction there.
        elif len(self.layers) == 1:
            self.layers[-1].apply_int_gate(gate)
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)
        else:
            for layer_index in range(len(self.layers) - 1, 0, -1):
                # If the interaction isn't applicable in (layer_index - 1)th layer, apply it in the next layer.
                if not self.layers[layer_index - 1].int_gate_applicable(gate):
                    self.layers[layer_index].apply_int_gate(gate)
                    self.remaining_interactions.remove_edge(log_qb0, log_qb1)
                    return
            self.layers[0].apply_int_gate(gate)
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)

    def attempt_apply_int(self, gate: HardEdge) -> None:
        r"""This is a softer version of :meth:`apply_int`.

        It first checks if there is an interaction to be done and doesn't do anything if there isn't, as opposed to
        raising an error. This method is made for cases when it's not clear whether an interaction has been applied
        between two logical qubits already.

        Args:
            gate: An edge between two :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit`\s where the interaction
                should be applied.

        """
        hard_qb0, hard_qb1 = gate

        log_qb0, log_qb1 = self.mapping.hard2log[hard_qb0], self.mapping.hard2log[hard_qb1]
        if self.remaining_interactions.has_edge(log_qb0, log_qb1):
            self.apply_int(gate)

    def count_swap_gates(self) -> int:
        r"""Counts the number of swap gates in all :class:`~iqm.qaoa.transpiler.routing.Layer`\s so far."""
        layers = self.layers
        number_of_swaps_in_layers = 0
        for layer in layers:
            for i in layer.gates.edges(data=True):
                if i[2]["swap"]:
                    number_of_swaps_in_layers += 1

        return number_of_swaps_in_layers

    # The following function builds the circuit of the QAOA. I tried hard to reduce the number of local
    # variables, but didn't manage to get under 19. It also doesn't make much sense to split it up into smaller
    # functions, so that's why the pylint warning is disabled.
    # pylint: disable=too-many-locals
    def build_qiskit(self, betas: list[float], gammas: list[float]) -> QuantumCircuit:
        r"""Build the QAOA circuit from the :class:`~iqm.qaoa.transpiler.routing.Routing` (``self``) in :mod:`qiskit`.

        The :class:`~iqm.qaoa.transpiler.routing.Routing` (``self``) contains all the information needed to create
        the phase separator part of the QAOA circuit. This method builds the rest of the circuit from it, i.e.:

        1. It initializes the qubits in the :math:`| + >` state by applying the Hadamard gate to all of them.
        2. It applies the interactions by going through the :class:`~iqm.qaoa.transpiler.routing.Layer`\s of
           the :class:`~iqm.qaoa.transpiler.routing.Routing`.
        3. It applies local fields.
        4. It applies the driver.
        5. It repeats steps 2-4 until it uses up all ``betas`` and ``gammas``.
        6. It applies the measurements and barrier before them.

        Args:
            betas: The QAOA parameters to be used in the driver (*RX* gate).
            gammas: The QAOA parameters to be used in the phase separator (*RZ* and *RZZ* gates).

        Returns:
            A complete QAOA :class:`~qiskit.circuit.QuantumCircuit`.

        """
        if len(betas) != len(gammas):
            raise ValueError("The lengths of ``gammas`` and ``betas`` need to be the same!")

        layers = cp.deepcopy(self.layers)
        mapping = cp.deepcopy(self.initial_mapping)
        qb_register = sorted(self.mapping.hard2log.keys())

        qiskit_circ = QuantumCircuit(len(qb_register), len(qb_register))

        # Prepare uniform superposition.
        qiskit_circ.h(range(len(qb_register)))

        for gamma, beta in zip(gammas, betas):
            # Apply phase separator.
            for layer in layers:
                for i in layer.gates.edges(data=True):
                    if i[2]["int"]:
                        log_qb0 = mapping.hard2log[i[0]]
                        log_qb1 = mapping.hard2log[i[1]]
                        weight = self.problem.get_quadratic(log_qb0, log_qb1)
                        if weight != 0:
                            qiskit_circ.rzz(2 * gamma * weight, qb_register.index(i[0]), qb_register.index(i[1]))

                for i in layer.gates.edges(data=True):
                    if i[2]["swap"]:
                        qiskit_circ.swap(qb_register.index(i[0]), qb_register.index(i[1]))

                mapping.update(layer)

            for hard_qb in mapping.hard2log.keys():
                log_qb = mapping.hard2log[hard_qb]
                local_field = self.problem.get_linear(log_qb)
                qiskit_circ.rz(2 * gamma * local_field, qb_register.index(hard_qb))

            layers.reverse()

            # Apply driver.
            qiskit_circ.rx(2 * beta, range(len(qb_register)))

        classical_bits = [mapping.hard2log[hard_qb] for hard_qb in qb_register]

        qiskit_circ.barrier()
        qiskit_circ.measure(np.arange(len(qb_register)), classical_bits)

        return qiskit_circ

    def draw(self) -> None:
        r"""Plot all :class:`~iqm.qaoa.transpiler.routing.Layer`\s of the routing in batches of 9.

        This creates a series of plots that are shown on the screen. Each plot contains 9
        :class:`~iqm.qaoa.transpiler.routing.Layer`\s arranged in a 3x3 grid. Each
        :class:`~iqm.qaoa.transpiler.routing.Layer` is drawn using :meth:`~iqm.qaoa.transpiler.routing.Layer.draw`.
        Therefore, it has the shape of the QPU topology with edges colored based on what is happening on them in
        the given :class:`~iqm.qaoa.transpiler.routing.Layer`.

        - Yellow highlight if a combination of swap and int is applied.
        - Blue highlight if a swap gate is applied.
        - Green highlight if an interaction gate is applied.
        - No highlight (black) if nothing is happening along the edge.
        """
        layer_count = len(self.layers)
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
                    layer.draw(mapping=mapping, ax=axs[row, column], show=False)
                    axs[row, column].set_axis_off()
                    axs[row, column].autoscale_view()
                    axs[row, column].set_title(f"Layer {layer_index}")
                    mapping.update(layer)
                    layer_index += 1
                plt.show()
        else:
            self.layers[0].draw(mapping=self.initial_mapping)
