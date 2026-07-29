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
"""Module containing the main object classes used throughout any routing algorithm.

Specififally, there is :class:`~iqm.qaoa.transpiler.routing.Mapping`, which keeps track of the mapping between
the logical and the hardware qubits. There is :class:`~iqm.qaoa.transpiler.routing.Routing` used to construct and save
the routing of the phase separator of the QAOA. The routing is saved as a list of layers,
:class:`~iqm.qaoa.transpiler.routing.Layer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy as cp
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any
import warnings

from dimod import BinaryQuadraticModel, to_networkx_graph
from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.vartypes import BINARY
from iqm.qaoa.transpiler.quantum_hardware import QPU, HardEdge, HardQubit, LogEdge, LogQubit
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import IGate


class BaseMapping:
    """The base class for various mappings used in routing of QAOA circuits on the QPU."""

    def __init__(self, qpu: QPU) -> None:
        self.hard_qbs = qpu.qubits
        self.qpu = qpu


class Mapping(BaseMapping):
    """Mapping between logical and hardware qubits.

    It maintains two dictionaries: :attr:`log2hard` and :attr:`hard2log` which are mappings between logical
    and hardware qubits. They are always kept in sync. The names for the hardware and logical qubits are extracted from
    ``qpu`` and ``problem_bqm`` at initialization.

    Args:
        qpu: a :class:`~iqm.qaoa.transpiler.quantum_hardware.QPU` object describing the topology of the QPU, used to
            get hardware qubits.
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` of the problem we're trying to solve, used to get
            logical qubits.
        partial_initial_mapping: An optional dictionary that contains a partial mapping to use as a starting point.
            The keys should be :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` and the values
            :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`.

    Raises:
        ValueError: If there are more logical qubits than hardware qubits.
        ValueError: If ``partial_initial_mapping`` is provided, but it's not bijective.
        ValueError: If ``partial_initial_mapping`` is provided, but it contains qubits not existing on the QPU.

    """

    def __init__(
        self,
        qpu: QPU,
        variables: Iterable[LogQubit],
        partial_initial_mapping: dict[HardQubit, LogQubit] | None = None,
    ) -> None:
        super().__init__(qpu)
        self.log_qbs = set(variables)

        # This should probably never happen, but it never hurts to add an extra check.
        if len(self.hard_qbs) < len(self.log_qbs):
            raise ValueError("There is fewer hardware qubits than logical qubits, mapping is impossible!")

        # If no partial initial mapping is provided, just map the qubits to each other arbitrarily.
        log_qbs_padded = list(self.log_qbs) + [None] * (len(self.hard_qbs) - len(self.log_qbs))
        if partial_initial_mapping is None:
            self._hard2log: dict[HardQubit, LogQubit | None] = dict(zip(self.hard_qbs, log_qbs_padded, strict=True))

        # If a partial inital mapping is provided, use it.
        else:
            if not (set(partial_initial_mapping.keys()) <= self.hard_qbs):
                raise ValueError(
                    f"The initial mapping contains qubits which don't exist on the QPU. "
                    f"Qubits on the QPU: {self.hard_qbs}. Qubits in the initial mapping: "
                    f"{partial_initial_mapping.keys()}."
                )
            if len(set(partial_initial_mapping.values())) != len(partial_initial_mapping.values()):
                raise ValueError("The initial mapping between hardware and logical qubits is not bijective!")
            if len(partial_initial_mapping) < len(self.log_qbs):
                remaining_hard_qbs = self.hard_qbs - set(partial_initial_mapping.keys())
                remaining_log_qbs = self.log_qbs - set(partial_initial_mapping.values())
                initial_mapping: dict = partial_initial_mapping
                # The qubits not covered by the partial inital mapping get mapped arbitrarily.
                for hard_qb, log_qb in zip_longest(remaining_hard_qbs, remaining_log_qbs, fillvalue=None):
                    initial_mapping[hard_qb] = log_qb
            else:
                initial_mapping = {hw_qb: partial_initial_mapping.get(hw_qb) for hw_qb in self.hard_qbs}

            self._hard2log = initial_mapping

        self._log2hard = {log_qb: hard_qb for hard_qb, log_qb in self._hard2log.items() if log_qb is not None}

    @property
    def hard2log(self) -> dict[HardQubit, LogQubit | None]:
        """The dictionary containing the mapping from hardware qubits to logical qubits."""
        return self._hard2log

    @property
    def log2hard(self) -> dict[LogQubit, HardQubit]:
        """The dictionary containing the mapping from logical qubits to hardware qubits."""
        return self._log2hard

    def swap_log(self, gate: LogEdge) -> None:
        """Swap association between a pair of logical qubits.

        Updates the dictionaries :attr:`hard2log` and :attr:`log2hard`.

        Args:
            gate: The pair of logical qubits to swap.

        """
        qb0, qb1 = gate
        hard_qb0 = self.log2hard[qb0]
        hard_qb1 = self.log2hard[qb1]
        self._hard2log[hard_qb0], self._hard2log[hard_qb1] = qb1, qb0
        self._log2hard[qb0], self._log2hard[qb1] = self._log2hard[qb1], self._log2hard[qb0]

    def swap_hard(self, gate: HardEdge) -> None:
        """Swap association between a pair of hardware qubits.

        Updates the dictionaries :attr:`hard2log` and :attr:`log2hard`.

        Args:
            gate: The pair of hardware qubits to swap.

        """
        qb0, qb1 = gate
        log_qb0 = self.hard2log[qb0]
        log_qb1 = self.hard2log[qb1]
        self._hard2log[qb0], self._hard2log[qb1] = self._hard2log[qb1], self._hard2log[qb0]
        # swap log→hard, but only if logical qubits are assigned (not ``None``).
        if log_qb0 is not None:
            self._log2hard[log_qb0] = qb1
        if log_qb1 is not None:
            self._log2hard[log_qb1] = qb0

    def move_hard(self, source_qubit: HardQubit, target_qubit: HardQubit) -> None:
        """Move a logical qubit from a one hardware qubit to a an unassigned hardware qubit on the QPU.

        The target ``target_qubit`` must be mapped to ``None`` in :attr:`hard2log` and (correspondingly) it must not
        appear among the values of :attr:`log2hard`. The mapping dictionaries :attr:`hard2log` and :attr:`log2hard` are
        changed as follows:

        * The value assigned to the key ``source_qubit`` in :attr:`hard2log` changes to ``None``. The value assigned to
          ``target_qubit`` changes to the previous value of ``source_qubit``.

        * The dictionary :attr:`log2hard` is modified correspondingly. The value ``source_qubit`` is changed to
          ``target_qubit``.

        Args:
            source_qubit: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` whose
                :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit` is being moved.
            target_qubit: The :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit` where
                the :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit` is being moved.

        Raises:
            ValueError: If the ``target_qubit`` is already assigned to a different logical qubit.
            ValueError: If the ``source_qubit`` is not assigned to any logical qubit.

        """
        if self._hard2log[target_qubit] is not None:
            raise ValueError(
                f"The target qubit {target_qubit} is already occupied by a logical qubit "
                f"{self._hard2log[target_qubit]}."
            )
        corresponding_log_qb = self._hard2log[source_qubit]
        if corresponding_log_qb is None:
            raise ValueError(f"The source qubit {source_qubit} is not assigned to any logical qubit.")

        # Modify ``self._hard2log``
        self._hard2log[target_qubit] = corresponding_log_qb
        self._hard2log[source_qubit] = None
        self._log2hard[corresponding_log_qb] = target_qubit

    def update(self, layer: Layer) -> None:
        """Update the mapping based on the swap gates found in a :class:`~iqm.qaoa.transpiler.routing.Layer` object.

        A convenience function that iterates over the gates in a :class:`~iqm.qaoa.transpiler.routing.Layer` object and
        swaps the hardware qubits corresponding to swap gates.

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
        self.gates: nx.Graph = nx.Graph()
        for hard_qb0, hard_qb1 in self.qpu.hardware_graph.edges():
            self.gates.add_edge(hard_qb0, hard_qb1, swap=False, int=False)
            self.gates.nodes[hard_qb0]["blocked"] = False
            self.gates.nodes[hard_qb1]["blocked"] = False

        for gate in int_gates:
            self.apply_int_gate(gate)

        for gate in swap_gates:
            self.apply_swap_gate(gate)

    def _qbs_not_involved_in_other_gate(self, gate: HardEdge) -> bool:
        """True iff the two qubits involved in the proposed gate are not already involved in other gates."""
        hard_qb0, hard_qb1 = gate

        return not (self.gates.nodes[hard_qb0]["blocked"] or self.gates.nodes[hard_qb1]["blocked"])

    def int_gate_applicable(self, gate: HardEdge) -> bool:
        """True iff the proposed interaction gate can be executed within the given layer.

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
        """True iff the proposed SWAP gate can be executed within the given layer.

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
        gate_lists: dict[str, list[tuple[HardQubit, ...]]] = {"y": [], "b": [], "g": []}
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


class BaseRouting(ABC):
    """The abstract base class for various routing sub-classes.

    Generally speaking, routing refers to the process of 'moving' the problem qubits around the QPU to allow the
    execution of the interactions between the problem qubits. The standard way to do this is with *swap* gates, but
    there are special variations e.g., if using *move* gates on a computational resonator.
    """

    def __init__(self, problem_bqm: BinaryQuadraticModel, qpu: QPU) -> None:
        self.problem = problem_bqm
        # The variable :meth:`remaining_interactions` keeps track of all interactions remaining to be executed.
        # So at the beginning it's equal to all of the iteractions in ``problem_bqm``.
        self.remaining_interactions = to_networkx_graph(problem_bqm)
        self.qpu = qpu

    @property
    @abstractmethod
    def layers(self) -> list[Any]:
        """The list of layers of the routing object.

        Different subclasses of :class:`BaseRouting` represent the individual layers differently, which is why the type
        of the layers here is the generic :class:`~typing.Any`.
        """
        raise NotImplementedError

    @abstractmethod
    def build_qiskit(self, betas: list[float], gammas: list[float], measurement: bool = True) -> QuantumCircuit:
        """The method to construct a qiskit circuit out of the routing, given a list of ``gammas`` and ``betas`` angles.

        Args:
            betas: The QAOA parameters (angles) to be used in the mixer.
            gammas: The QAOA parameters (angles) to be used in the phase separator.
            measurement: Should the quantum circuit end with a measurement of all qubits?

        Returns:
            A qiskit circuit implementing the QAOA.

        """
        raise NotImplementedError


class Routing(BaseRouting):
    """The 'standard' routing of a QAOA phase separator.

    A :class:`~iqm.qaoa.transpiler.routing.Routing` object is intended to be directly used by a router during routing.
    This class is meant to be used for the relatively 'standard' routers which work by use *swap* gates and interaction
    gates to move the problem qubits around the QPU. To that end it maintains a list of
    :class:`~iqm.qaoa.transpiler.routing.Layer` objects, a :class:`~networkx.Graph` with the interactions not
    implemented yet and a :class:`~iqm.qaoa.transpiler.routing.Mapping` object that represents the current status of
    mapping between hardware and logical qubits.

    A router interacts with a :class:`~iqm.qaoa.transpiler.routing.Routing` object by using the methods
    :meth:`apply_swap` and :meth:`apply_int`. Optionally also :meth:`attempt_apply_int`. If the problem BQM contains
    interactions of strength 0 (e.g., because of padding), those won't be added into the list of layers. When the method
    :meth:`apply_int` is called on those interactions, it is skipped.

    Args:
        problem_bqm: The optimization problem represented as :class:`~dimod.BinaryQuadraticModel`.
        qpu: The QPU representing the hardware qubit topology.
        initial_mapping: The starting mapping of the logical-to-hardware qubits.

    Generally speaking, routing refers to the process of 'moving' the problem qubits around the QPU to allow the
    execution of the interactions between the problem qubits. The standard way to do this is with *swap* gates, but
    there are special variations e.g., if using *move* gates on a computational resonator.

    """

    def __init__(self, problem_bqm: BinaryQuadraticModel, qpu: QPU, initial_mapping: Mapping | None = None) -> None:
        super().__init__(problem_bqm=problem_bqm, qpu=qpu)
        if initial_mapping is None:
            self.initial_mapping = Mapping(self.qpu, self.problem.variables)
        else:
            self.initial_mapping = initial_mapping

        self.mapping = cp.deepcopy(self.initial_mapping)

        self._layers: list[Layer] = [Layer(self.qpu)]

    @property
    def layers(self) -> list[Layer]:
        """The list of layers of the routing object."""
        return self._layers

    def apply_swap(self, gate: HardEdge, attempt_int: bool = False) -> None:
        r"""Apply swap gate at the earliest possible layer, add a new layer if needed.

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
            """Applies the swap gate in the layer defined by the index ``layer_index``."""
            # Apply the swap gate in the correct :class:`~iqm.qaoa.transpiler.routing.Layer`.
            self._layers[layer_index].apply_swap_gate(gate)
            # Update the :class:`~iqm.qaoa.transpiler.routing.Mapping`.
            self.mapping.swap_hard(gate)
            if attempt_int:
                hard_qb0, hard_qb1 = gate
                log_qb0, log_qb1 = self.mapping.hard2log[hard_qb0], self.mapping.hard2log[hard_qb1]
                if self.remaining_interactions.has_edge(log_qb0, log_qb1):
                    self.apply_int(gate)

        if not self._layers[-1].swap_gate_applicable(gate):
            self._layers.append(Layer(self.qpu))
            _internal_apply_swap(-1)
        elif len(self._layers) == 1:
            _internal_apply_swap(-1)
        else:
            for layer_index in range(len(self._layers) - 1, 0, -1):
                if not self._layers[layer_index - 1].swap_gate_applicable(gate):
                    _internal_apply_swap(layer_index)
                    break
                if layer_index == 1:
                    _internal_apply_swap(0)

    def apply_int(self, gate: HardEdge) -> None:
        r"""Apply interaction gate at the earliest possible layer, add a new layer if necessary.

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
        if not self._layers[-1].int_gate_applicable(gate):
            self._layers.append(Layer(self.qpu))
            self._layers[-1].apply_int_gate(gate)
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)
        # If there is only a single layer, apply the interaction there.
        elif len(self._layers) == 1:
            self._layers[-1].apply_int_gate(gate)
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)
        else:
            for layer_index in range(len(self._layers) - 1, 0, -1):
                # If the interaction isn't applicable in (layer_index - 1)th layer, apply it in the next layer.
                if not self._layers[layer_index - 1].int_gate_applicable(gate):
                    self._layers[layer_index].apply_int_gate(gate)
                    self.remaining_interactions.remove_edge(log_qb0, log_qb1)
                    return
            self._layers[0].apply_int_gate(gate)
            self.remaining_interactions.remove_edge(log_qb0, log_qb1)

    def attempt_apply_int(self, gate: HardEdge) -> None:
        r"""Softer version of :meth:`apply_int`.

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
        layers = self._layers
        number_of_swaps_in_layers = 0
        for layer in layers:
            for i in layer.gates.edges(data=True):
                if i[2]["swap"]:
                    number_of_swaps_in_layers += 1

        return number_of_swaps_in_layers

    def build_qiskit(
        self, betas: list[float], gammas: list[float], measurement: bool = True, cancel_cnots: bool = True
    ) -> QuantumCircuit:
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
            measurement: Should the circuit contain a layer of measurements or not?
            cancel_cnots: The routing is likely to contain a *SWAP* gate followed by an *RZZ* gate (or vice versa). When
                decomposed into a particular basis gate set, these contain a pair of *CNOT* gates, which can be
                cancelled. Iff ``cancel_cnots`` is ``True``, those will be cancelled already in :meth:`build_qiskit`.

        Returns:
            A complete QAOA :class:`~qiskit.circuit.QuantumCircuit`.

        Raises:
            ValueError: If the lengths of the provided ``betas`` and ``gammas`` aren't the same.

        """
        if len(betas) != len(gammas):
            raise ValueError("The lengths of ``gammas`` and ``betas`` need to be the same!")

        layers = cp.deepcopy(self._layers)
        mapping = cp.deepcopy(self.initial_mapping)

        qiskit_circ = QuantumCircuit(len(self.mapping.hard2log), len(self.mapping.log2hard))

        # Prepare uniform superposition.
        qiskit_circ.h(mapping.log2hard.values())  # Only act on the HW qubits which carry a logical qubit.

        for gamma, beta in zip(gammas, betas, strict=True):
            # Apply phase separator.
            for layer in layers:
                # Iterate through the layers of the routing.
                for i in layer.gates.edges(data=True):
                    if i[2]["int"]:
                        # If there is an interaction, we need to figure out its strength.
                        log_qb0 = mapping.hard2log[i[0]]
                        log_qb1 = mapping.hard2log[i[1]]
                        weight = self.problem.get_quadratic(log_qb0, log_qb1)

                        if (
                            weight != 0 and i[2]["swap"] and cancel_cnots
                        ):  # The only situation in which we cancel CNOTs.
                            qiskit_circ.cx(i[0], i[1])
                            qiskit_circ.rz(2 * gamma * weight, i[1])
                            qiskit_circ.cx(i[1], i[0])
                            qiskit_circ.cx(i[0], i[1])

                        elif weight != 0:
                            qiskit_circ.rzz(2 * gamma * weight, i[0], i[1])

                    # Avoid the case when we already did the cancellation (or there is no interaction)
                    if i[2]["swap"] and (not i[2]["int"] or not cancel_cnots):
                        qiskit_circ.swap(i[0], i[1])

                mapping.update(layer)

            for log_qb, hard_qb in mapping.log2hard.items():
                local_field = self.problem.get_linear(log_qb)
                qiskit_circ.rz(2 * gamma * local_field, hard_qb)

            layers.reverse()

            # Apply driver, only acting on the HW qubits which correspond to a logical qubit.
            qiskit_circ.rx(2 * beta, mapping.log2hard.values())

        if measurement:
            qiskit_circ.barrier()
            for log_qb, hard_qb in mapping.log2hard.items():
                qiskit_circ.measure(hard_qb, self.problem.variables.index(log_qb))

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
        layer_count = len(self._layers)
        if layer_count > 1:
            layer_batches = [self._layers[x : x + 9] for x in range(0, len(self._layers), 9)]
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
            self._layers[0].draw(mapping=self.initial_mapping)


class ParityMapping(BaseMapping):
    """Maps hardware qubits to sets of logical qubits representing parities.

    This class maintains a dynamic mapping between physical (hardware) qubits and logical qubit sets that encode
    parity relationships used in quantum circuit synthesis of phase polynomials.

    Args:
        qpu: The QPU providing the hardware graph.
        initial_mapping: Initial mapping from hardware qubits to sets of logical qubits representing parities.

    Raises:
        ValueError: If the initial mapping contains qubits not present in the QPU.
        ValueError: If the initial mapping maps any hardware qubit to a parity of multiple qubits. The initial mapping
            may only map hardware qubits to either the empty set ``set()`` or to sets containing a single logical qubit,
            such as e.g., ``{log_qb}``.

    """

    def __init__(self, qpu: QPU, initial_mapping: dict[HardQubit, set[LogQubit]]) -> None:
        super().__init__(qpu)

        if not (set(initial_mapping.keys()) <= self.hard_qbs):
            raise ValueError(
                f"The initial mapping contains qubits which don't exist on the QPU. "
                f"Qubits on the QPU: {self.hard_qbs}. Qubits in the initial mapping: "
                f"{initial_mapping.keys()}."
            )
        maximum_parity_size = max(len(parity) for parity in initial_mapping.values())
        if maximum_parity_size > 1:
            raise ValueError(
                f"Invalid initial_mapping: each hardware qubit must map to either an empty set or a set containing "
                f"exactly one logical qubit; found maximum parity size {maximum_parity_size}."
            )

        remaining_hard_qbs = self.hard_qbs - set(initial_mapping.keys())
        for hard_qb in remaining_hard_qbs:
            initial_mapping[hard_qb] = set()

        self.parity_mapping = initial_mapping
        # The matrix that remembers the transformation with respect to the initial mapping.
        self.parity_transform_matrix = np.eye(len(qpu.qubits), dtype=bool)

    def cnot(self, control: HardQubit, target: HardQubit) -> None:
        """Simulates the effect of a CNOT gate on the parity mapping.

        The CNOT updates the parity of the target qubit by XOR-ing it with the control qubit's parity set.

        Args:
            control: The control hardware qubit.
            target: The target hardware qubit.

        Raises:
            ValueError: If the control and target qubits are not connected in the QPU hardware graph.

        """
        if not self.qpu.hardware_graph.has_edge(control, target):
            raise ValueError(
                f"The two qubits {control} and {target} are not connected on the QPU graph. "
                "CNOT gate can not be applied."
            )
        self.parity_mapping[target] = self.parity_mapping[target] ^ self.parity_mapping[control]
        self.parity_transform_matrix[:, target] ^= self.parity_transform_matrix[:, control]

    def swap_hard(self, gate: HardEdge) -> None:
        """Swaps the parity mapping between two hardware qubits.

        This corresponds to exchanging the logical parity assignments of two connected hardware qubits.

        Args:
            gate: A pair of hardware qubits (qb0, qb1) to swap.

        """
        qb0, qb1 = gate
        self.parity_mapping[qb0], self.parity_mapping[qb1] = self.parity_mapping[qb1], self.parity_mapping[qb0]
        self.parity_transform_matrix[:, [qb0, qb1]] = self.parity_transform_matrix[:, [qb1, qb0]]


class CircuitSynthesis:
    """Synthesize a quantum circuit using parity-based interaction mapping.

    The hardware-to-parity mapping is saved in an internal :class:`ParityMapping`.

    There are two main regimes to use the class:

    1. If a diagonal Hamiltonian is known beforehand, only the method :meth:`cnot` is used for constructing the circuit.
       When a Qiskit :class:`~qiskit.circuit.QuantumCircuit` is created with :meth:`build_qiskit`, RZ gates are included
       automatically based on the provided diagonal Hamiltonian (as a
       :class:`~dimod.higherorder.polynomial.BinaryPolynomial` object).
    2. Alternatively, one may manually insert CNOT and RZ gates. The attribute ``possible_ints`` contains a set of all
       possible interactions that could be implemented given the CNOT gates (so it can be used to construct a
       Hamiltonian). The attribute ``constructed_hamiltonian_bp`` contains the Hamiltonian that has been implicitly
       constructed by manually applying the RZ gates.

    An external algorithm might be needed to determine the optimal placement of the CNOT gates.

    Args:
        mapping: Initial parity mapping used to construct the circuit.

    """

    @dataclass(frozen=True)
    class CNOTStep:
        """A custom dataclass representing one CNOT step in the circuit synthesis.

        Attributes:
            control: The control qubit of the CNOT gate.
            target: The target qubit of the CNOT gate.
            allow_interactions: True iff we permit the circuit synthesis algorithm to add RZ gates directly after this
                CNOT gate. This is useful for better control over the structure of the quantum circuit.

        """

        control: HardQubit
        target: HardQubit
        allow_interactions: bool = True

    def __init__(self, mapping: ParityMapping) -> None:
        self.initial_mapping = mapping  # Save for later use.
        self.mapping = cp.deepcopy(self.initial_mapping)
        self.pre_uncomputation_mapping: ParityMapping | None = None

        self.possible_ints: set[frozenset[LogQubit]] = {frozenset(qbts) for qbts in mapping.parity_mapping.values()}
        self.possible_ints.discard(frozenset())  # Remove the trivial 0-order term.
        self.constructed_hamiltonian_bp = BinaryPolynomial({}, vartype="SPIN")

        self.compute_cnots: list[CircuitSynthesis.CNOTStep] = []
        self.uncompute_cnots: list[CircuitSynthesis.CNOTStep] = []

    @property
    def computing(self) -> bool:
        """A flag signalling that the circuit synthesis is currently in the 'computing' mode."""
        return self.pre_uncomputation_mapping is None

    @property
    def cnot_list(self) -> list[CircuitSynthesis.CNOTStep]:
        """Returns the list of all CNOT gates applied throughout the circuit synthesis process."""
        return self.compute_cnots + self.uncompute_cnots

    def begin_uncompute(self) -> None:
        """Calling this ends the computing phase of the circuit synthesis and begins the uncomputing phase.

        Raises:
            RuntimeError: If uncomputation has already begun previously.

        """
        if not self.computing:
            raise RuntimeError("Uncomputation has already begun.")
        self.pre_uncomputation_mapping = cp.deepcopy(self.mapping)

    def cnot(self, control: HardQubit, target: HardQubit, allow_int_after: bool = True) -> None:
        """Adds a CNOT gate to the synthesized circuit.

        Updates the internal parity mapping and records the interaction.

        Args:
            control: Control hardware qubit.
            target: Target hardware qubit.
            allow_int_after: True iff interactions are allowed to take place immediately after the CNOT gate.

        """
        self.mapping.cnot(control, target)
        if self.computing:
            self.compute_cnots.append(self.CNOTStep(control, target, allow_int_after))
            self.possible_ints.add(frozenset(self.mapping.parity_mapping[target]))
        else:
            # allow_interactions forced False: the uncompute segment is physically incapable of an RZ.
            self.uncompute_cnots.append(self.CNOTStep(control, target, allow_interactions=False))

    def rz(self, interaction: float, qubit: HardQubit, allow_overwrite: bool = False) -> None:
        """Adds an RZ rotation gate associated with a parity interaction.

        Also, adds an interaction term to the internal :class:`~dimod.higherorder.polynomial.BinaryPolynomial` implied
        by applying the RZ gate. This internal :class:`~dimod.higherorder.polynomial.BinaryPolynomial` may be accessed
        by the attribute ``self.constructed_hamiltonian_bp``.

        Args:
            interaction: The interaction strength corresponding to the RZ gate.

                .. warning::

                    The input is the **interaction strength** corresponding to the RZ gate, i.e., the term that will be
                    added to the Hamiltonian. The rotation angle of the RZ gate is this value times the γ QAOA angle.

            qubit: Hardware qubit on which the RZ gate is applied.
            allow_overwrite: If ``True``, allows overwriting an existing interaction term in the internal
                :class:`~dimod.higherorder.polynomial.BinaryPolynomial`.

        Raises:
            ValueError: If the interaction term already exists in the internal
                :class:`~dimod.higherorder.polynomial.BinaryPolynomial` and overwrite is disabled
            ValueError: If the qubit on which the RZ rotation acts carries no parity information.
            RuntimeError: If attempted to place an RZ gate during uncomputation.

        """
        if not self.computing:
            raise RuntimeError("RZ gates cannot be placed during uncomputation.")
        int_term = frozenset(self.mapping.parity_mapping[qubit])
        if (int_term in self.constructed_hamiltonian_bp) and not allow_overwrite:
            raise ValueError(f"The term {int_term} already exists in the constructed Hamiltonian.")
        if int_term == frozenset():
            raise ValueError("The RZ gate is being applied to a qubit carrying no parity information.")
        self.constructed_hamiltonian_bp[int_term] = interaction

    def uncompute_parities(self) -> None:
        """Adds CNOT gates to the circuit synthesis to uncompute the parity mapping back to ``self.initial_mapping``.

        Currently, the uncomputing is very primitive. It just mirrors the CNOTs that were added in the first half of the
        circuit.
        """
        self.begin_uncompute()
        for step in reversed(self.compute_cnots):
            self.cnot(step.control, step.target)

    def build_qiskit_phase_separator(
        self,
        gamma: float,
        interactions: BinaryPolynomial | None = None,
        show_parities: bool = False,
        remove_cnots: bool = False,
    ) -> QuantumCircuit:
        r"""Builds the synthesized quantum circuit.

        This only builds the synthesized part of the quantum circuit. For constructing the entire QAOA circuit, use
        :meth:`build_qiskit`.

        The circuit is constructed using CNOT gates in the order in which the method :meth:`cnot` was called, acting on
        the respective qubits.

        - If ``interactions`` is not provided, the circuit inserts an RZ gate wherever it was placed using the
          :meth:`rz` method.
        - If ``interactions`` is provided, the circuit **ignores** the manually placed RZ gates (and raises a warning if
          some were manually placed). It then places RZ gates automatically whenever a qubit carries one of the parities
          from ``interactions``. A warning is raised if not all ``interactions`` are executed.

        Args:
            gamma: Global scaling factor for interaction strengths (e.g., a parameter of the QAOA ansatz).
            interactions: Optional Hamiltonian of interactions. If ``None``, uses internally constructed interactions.
                The ``varytpe`` of the interactions has to be ``"SPIN"`` and it is assumed that the quantum state
                :math:`|1\rangle` corresponds to the value -1 in the polynomial (and :math:`|0\rangle` corresponds to
                1). This comes from seeing the :class:`~dimod.higherorder.polynomial.BinaryPolynomial` as describing a
                Hamiltonian made up of sums of products of the Z Pauli gate. The reason we don't accept the input with
                ``vartype`` ``"BINARY"`` is that the conversion from ``"BINARY"`` to ``"SPIN"`` by default uses the
                opposite convention (where :math:`|1\rangle` corresponds to the value 1 in the polynomial and
                :math:`|0\rangle` to -1). This could lead to silent errors.
            show_parities: Iff set to ``True``, adds identity gates throughout the circuit whose labels show the parity
                encoded in the qubits.
            remove_cnots: Iff set to ``True``, the uncomputation is skipped and the computation happens in the opposite
                order. This saves 2-qubit gates, if this is the first phase seprator applied onto the initial state
                (which is presumably the :math:`|+\rangle` state).

        Returns:
            A Qiskit :class:`~qiskit.circuit.QuantumCircuit` implementing the synthesized quantum circuit.

        Raises:
            ValueError: If the input ``interactions`` has ``vartype`` set to ``"BINARY"``.
            ValueError: If ``remove_cnots`` is set to ``True``, but the circuit synthesis contains no uncomputation.

        """
        if interactions is None:
            interactions = self.constructed_hamiltonian_bp
        elif interactions.vartype == BINARY:
            raise ValueError(
                "The input `interactions` has `interactions.vartype` equal to 'BINARY'."
                "The `vartype` has to be 'SPIN', the interactions are supposed to represent a Hamiltonian."
            )
        elif len(self.constructed_hamiltonian_bp) > 0:
            warnings.warn(
                "RZ gates were manually added, but an explicit `interactions` argument was provided to `build_qiskit`. "
                "The manually added RZ gates will be ignored, and RZ gates will instead be generated from "
                "`interactions`.",
                stacklevel=2,
            )

        qc = QuantumCircuit(len(self.mapping.parity_mapping))

        unrealized_interactions = interactions.copy()
        unrealized_interactions.pop(frozenset(), None)  # Remove the constant term, if present.

        def apply_interaction(
            parity: frozenset[LogQubit] | set[LogQubit],
            qubit: HardQubit,
        ) -> None:
            parity = frozenset(parity)
            if unrealized_interactions.get(parity):  # If it exists and is nonzero.
                qc.rz(2 * unrealized_interactions[parity] * gamma, qubit)
                del unrealized_interactions[parity]

        if remove_cnots:
            if self.pre_uncomputation_mapping is None:
                raise ValueError(
                    "It's impossible to remove the uncomputation CNOTs if uncomputation wasn't even started."
                )
            current_mapping = cp.deepcopy(self.pre_uncomputation_mapping)
            cnot_list = self.compute_cnots[::-1]
        else:
            current_mapping = cp.deepcopy(self.initial_mapping)
            cnot_list = self.cnot_list

        # First do all RZ rotations that make sense without doing any CNOTs.
        for hq, parity in current_mapping.parity_mapping.items():
            if show_parities:
                qc.append(IGate(label=str(parity)), [hq])
            apply_interaction(parity, hq)

        # Now we start applying CNOTs.
        for cnot in cnot_list:
            qc.cx(cnot.control, cnot.target)
            current_mapping.cnot(cnot.control, cnot.target)

            control_parity = current_mapping.parity_mapping[cnot.control]
            target_parity = current_mapping.parity_mapping[cnot.target]
            # With CNOT, we created a new parity -> check if we want to do the corresponding RZ.
            if show_parities:
                qc.append(IGate(label=str(target_parity)), [cnot.target])
            if cnot.allow_interactions:
                apply_interaction(target_parity, cnot.target)
                apply_interaction(control_parity, cnot.control)

        if any(unrealized_interactions.values()):  # Check if any entries remain with nonzero values.
            warnings.warn(
                f"Not all interactions were realized. Remaining interactions: {unrealized_interactions}", stacklevel=2
            )

        return qc

    def build_qiskit(
        self,
        betas: list[float],
        gammas: list[float],
        interactions: BinaryPolynomial | None = None,
        measurement: bool = True,
        remove_cnots_first_layer: bool = True,
    ) -> QuantumCircuit:
        r"""Build a full QAOA circuit from the synthesized parity-based circuit construction.

        This method assembles the complete QAOA circuit by repeatedly composing the parity-based phase separator
        (constructed via CNOT and RZ operations) with the mixer (RX rotations). It assumes that all parity
        transformations have been properly uncomputed before execution, i.e., the parity mapping has returned to the
        initial configuration up to a permutation of hardware qubits.

        Args:
            betas: List of QAOA mixer angles (RX rotations).
            gammas: List of QAOA phase separator angles.
            interactions: The problem Hamiltonian as a :class:`~dimod.higherorder.polynomial.BinaryPolynomial` in SPIN
                representation.
            measurement: If ``True``, adds a measurement step.
            remove_cnots_first_layer: If ``True``, the first layer of QAOA swaps its computation and uncomputation.
                Then, the uncomputation step is completely removed, since it consists purely of CNOT gates applied
                directly onto the :math:`|+\rangle` state, which has no effect.


        Returns:
            A :class:`~qiskit.circuit.QuantumCircuit` implementing the full synthesized QAOA circuit.

        Raises:
            ValueError: If ``betas`` and ``gammas`` do not have the same length.
            ValueError: If the parity mapping has not been fully uncomputed back to the initial state (up to permutation
                of hardware qubits).

        """
        if len(betas) != len(gammas):
            raise ValueError("The lengths of ``gammas`` and ``betas`` need to be the same!")

        if interactions is None:
            interactions = self.constructed_hamiltonian_bp

        if not (
            (self.mapping.parity_transform_matrix.sum(axis=1) == 1).all()
            and (self.mapping.parity_transform_matrix.sum(axis=0) == 1).all()
        ):
            raise ValueError(
                "The parities have not been properly uncomputed. The mapping of the parities must be equal to the "
                "initial mapping, up to permutation of the hardware qubits."
            )

        qubit_map_from_circ_synth = np.argmax(self.mapping.parity_transform_matrix, axis=1)
        current_qubit_map = np.arange(len(self.mapping.parity_mapping))
        used_hw_qubits = np.array([hw for hw, lg in self.mapping.parity_mapping.items() if lg])

        n_vars = len(set().union(*interactions))
        qc = QuantumCircuit(len(self.mapping.parity_mapping), n_vars)

        qc.h(used_hw_qubits)

        for p, (gamma, beta) in enumerate(zip(gammas, betas, strict=True)):
            if remove_cnots_first_layer and (p == 0):  # First QAOA layer.
                phase_separator_qc = self.build_qiskit_phase_separator(gamma, interactions, False, remove_cnots=True)
                # The phase separator has to be applied like this for the qubit mapping to agree after the first layer.
                qc.compose(phase_separator_qc, inplace=True, qubits=qubit_map_from_circ_synth[current_qubit_map])
            else:
                phase_separator_qc = self.build_qiskit_phase_separator(gamma, interactions, False, remove_cnots=False)
                qc.compose(phase_separator_qc, inplace=True, qubits=current_qubit_map)

            current_qubit_map = qubit_map_from_circ_synth[current_qubit_map]

            qc.rx(2 * beta, current_qubit_map[used_hw_qubits])

        if measurement:
            sorted_vars = [
                next(iter(log_qb)) for _, log_qb in sorted(self.initial_mapping.parity_mapping.items()) if log_qb
            ]
            qc.barrier()
            for hw_qb in current_qubit_map[used_hw_qubits]:
                qc.measure(
                    hw_qb, sorted_vars.index(next(iter(self.initial_mapping.parity_mapping[current_qubit_map[hw_qb]])))
                )

        return qc
