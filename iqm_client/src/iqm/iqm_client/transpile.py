# Copyright 2021-2025 IQM client developers
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
"""Transpiling circuits to IQM devices involving computational resonators.

In the IQM Star architecture, computational resonators are connected to multiple qubits.
A MOVE gate can be used to move the state of a qubit to a connected, empty computational resonator,
and back *to the same qubit*. Additionally, two-component gates like CZ can be applied
between a qubit and a connected resonator, so that effectively the qubit can be made to interact
with other qubits connected to the resonator. However, the resonators cannot be measured,
and no single-qubit gates can be applied on them.

To enable third-party transpilers to work on the IQM Star architecture, we may abstract away the
resonators and replace the real dynamic quantum architecture with a *simplified architecture*.
Specifically, this happens by removing the resonators from the architecture, and for
each resonator ``r``, and for each pair of supported native qubit-resonator (QR) gates
``(G(q1, r), MOVE(q2, r))`` adding the *fictional* gate ``G(q1, q2)`` to the simplified architecture.
This works since the fictional gate can be implemented as the QR gate sequence

::

    G(q1, q2) = [MOVE(q2, r), G(q1, r), MOVE(q2, r)].

This sequence is called a *resolution* of ``G(q1, q2)``.
Currently  we assume all the QR gates (other than MOVE) are symmetric in the sense that

::

    [MOVE(q2, r), G(q1, r), MOVE(q2, r)] = G(q1, q2) = G(q2, q1) = [MOVE(q1, r), G(q2, r), MOVE(q1, r)]

holds. This has the effect of doubling the number of possible resolutions for ``G(q1, q2)`` since
you can reverse the roles of the qubits.

Before a circuit transpiled to a simplified architecture can be executed it must be further
transpiled to the real Star architecture using :func:`transpile_insert_moves`, which will introduce
the resonators, add MOVE gates as necessary to move the states, and convert the fictional two-qubit
gates into real native gates acting on qubit-resonator pairs.

Likewise :func:`transpile_remove_moves` can be used to perform the opposite transformation,
converting a circuit valid for the real Star architecture into an equivalent circuit for the
corresponding simplified architecture, e.g. so that the circuit can be retranspiled or optimized
using third-party tools that do not support the MOVE gate.

Given a :class:`DynamicQuantumArchitecture` for a Star architecture, the corresponding simplified
version can be obtained using :func:`simplify_architecture`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from enum import Enum

from iqm.iqm_client import (
    Circuit,
    CircuitTranspilationError,
    CircuitValidationError,
    DynamicQuantumArchitecture,
    Instruction,
)
from iqm.iqm_client.models import GateImplementationInfo, GateInfo, Locus, _op_is_symmetric
from iqm.iqm_client.validation import validate_circuit_moves, validate_instruction

Resolution = tuple[str, str, str]
"""A (gate qubit, move qubit, resonator) triple that represents a resolution of a fictional
 qubit-qubit gate."""


class ExistingMoveHandlingOptions(str, Enum):
    """Options for how :func:`transpile_insert_moves` should handle existing MOVE instructions
    in the circuit.
    """

    KEEP = "keep"
    """Strict mode. The circuit, including existing MOVE instructions in it, is validated first.
    Then, any fictional two-qubit gates in the circuit are implemented with qubit-resonator gates."""
    TRUST = "trust"
    """Lenient mode. Same as KEEP, but does not validate the circuit first.
    Will attempt to fix any apparent user errors in the circuit by adding extra MOVE gates.
    """
    REMOVE = "remove"
    """Removes existing MOVE instructions from the circuit using :func:`transpile_remove_moves`, and
    then does the same as TRUST. This may produce a more optimized end result."""


def _map_loci(
    instructions: Iterable[Instruction],
    qubit_mapping: dict[str, str],
    inverse: bool = False,
) -> tuple[Instruction, ...]:
    """Map the loci of the given instructions using the given qubit mapping, or its inverse.

    Args:
        instructions: Instructions whose loci are to be mapped.
        qubit_mapping: Mapping from one set of qubit names to another. Assumed to be injective.
        inverse: Invert ``qubit_mapping`` before using it.

    Returns:
        Copies of ``instructions`` with mapped loci.

    """
    if inverse:
        qubit_mapping = {phys: log for log, phys in qubit_mapping.items()}
    return tuple(
        inst.model_copy(update={"qubits": tuple(qubit_mapping[q] for q in inst.qubits)}) for inst in instructions
    )


class _ResonatorStateTracker:
    r"""Tracks the qubit states stored in computational resonators on a Star architecture QPU
    as they are moved with MOVE gates.

    Since the MOVE gate is not defined when acting on a :math:`|11\rangle` state,
    and involves an unknown phase, it is not equivalent to a SWAP.
    The state must always be moved back to its original qubit (this reverses the unknown phase),
    and no gates may be applied on the qubit while its state is in a resonator (to avoid populating
    the :math:`|11\rangle` state).

    Also contains information on available qubit-resonator gate loci.

    Args:
        qr_gates: Mapping from qubit-resonator gate name to mapping from qubit to resonators with
            which it has the gate available.

    """

    move_gate = "move"
    """Name of the MOVE gate in the architecture."""
    qr_gate_names = frozenset(("move", "cz"))
    """Names of all arity-2 gates that *can* (in principle) be applied on a (qubit, resonator) locus
    in the real Star architecture. They *may* also have (qubit, qubit) loci available if the architecture
    allows it.
    Other arity-2 gates in the real architecture are assumed to *require* a (qubit, qubit) locus."""

    def __init__(self, qr_gates: dict[str, dict[str, set[str]]]) -> None:
        # NOTE: For now the transpilation logic assumes all qr gates other than MOVE are symmetric
        # in their effect, though not in implementation.
        for op in self.qr_gate_names:
            if op != self.move_gate and not _op_is_symmetric(op):
                raise ValueError(f"QR gate {op} is not symmetric.")

        def invert_locus_mapping(mapping: dict[str, set[str]]) -> dict[str, set[str]]:
            """Invert the give mapping of resonators to a list of connected qubits, returning
            a mapping of qubits to connected resonators (or vice versa).
            """
            inverse: dict[str, set[str]] = {}
            for r, qubits in mapping.items():
                for q in qubits:
                    inverse.setdefault(q, set()).add(r)
            return inverse

        self.move_q2r = qr_gates.pop(self.move_gate, {})
        """Mapping from qubit to resonators it can be MOVEd into."""
        self.move_r2q = invert_locus_mapping(self.move_q2r)
        """Mapping from resonator to qubits whose state can be MOVEd into it."""
        self.qr_gates_q2r = qr_gates
        """Mapping from QR gate name to mapping from qubit to resonators with which it has the gate available."""

        self.res_state_owner = {r: r for r in self.resonators}
        """Maps resonator to the QPU component whose state it currently holds."""

    @classmethod
    def from_dynamic_architecture(cls, arch: DynamicQuantumArchitecture) -> _ResonatorStateTracker:
        """Constructor to make a _ResonatorStateTracker from a dynamic quantum architecture.

        Args:
            arch: Architecture that determines the available gate loci.

        Returns:
            Tracker for ``arch``.

        """
        resonators = set(arch.computational_resonators)
        qr_gates: dict[str, dict[str, set[str]]] = {}
        for gate_name in cls.qr_gate_names:
            if gate_info := arch.gates.get(gate_name):
                qr_loci: dict[str, set[str]] = {}
                for q, r in gate_info.loci:
                    if gate_name == cls.move_gate:
                        # MOVE must have a (q, r) locus
                        if q in resonators or r not in resonators:
                            raise ValueError(f"MOVE gate locus {q, r} is not of the form (qubit, resonator)")
                    elif q in resonators:
                        # Other QR gates
                        raise ValueError(f"Gate {gate_name} locus {q, r} is not of the form (qubit, *)")
                    qr_loci.setdefault(q, set()).add(r)
                qr_gates[gate_name] = qr_loci
        return cls(qr_gates)

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> _ResonatorStateTracker:
        """Constructor to make the _ResonatorStateTracker from a circuit.

        Infers the resonator connectivity and gate loci from the given the circuit.

        Args:
            circuit: The circuit to track the qubit states on.

        Returns:
            Tracker for the architecture inferred from ``circuit``.

        """
        return cls.from_instructions(circuit.instructions)

    @classmethod
    def from_instructions(cls, instructions: Iterable[Instruction]) -> _ResonatorStateTracker:
        """Constructor to make the _ResonatorStateTracker from a sequence of instructions.

        Infers the resonator connectivity from the MOVE gates.

        Args:
            instructions: The instructions to track the qubit states on.

        Returns:
            Tracker for the architecture inferred from the given instructions.

        """
        qr_gates: dict[str, dict[str, set[str]]] = {}
        for i in instructions:
            if i.name in cls.qr_gate_names:
                q, r = i.qubits
                qr_gates.setdefault(i.name, {}).setdefault(q, set()).add(r)
        return cls(qr_gates)

    @property
    def resonators(self) -> Collection[str]:
        """Computational resonators that are being tracked."""
        return self.move_r2q.keys()

    @property
    def qubit_state_holder(self) -> dict[str, str]:
        """Qubits not found in the dict hold their own states."""
        # TODO reverses res_state_owner, maybe this is the one we need to track instead?
        resonators = self.resonators
        # qubits whose states are in resonators
        return {c: r for r, c in self.res_state_owner.items() if c not in resonators}

    def apply_move(self, qubit: str, resonator: str) -> None:
        """Record changes to qubit state location when a MOVE gate is applied.

        Args:
            qubit: The moved qubit.
            resonator: The moved resonator.

        Raises:
            CircuitTranspilationError: MOVE is not allowed, either because the resonator does not exist,
                the MOVE gate is not available between this qubit-resonator pair, or the resonator is
                currently holding the state of a different qubit.

        """
        if (
            resonator in self.resonators
            and qubit in self.move_r2q[resonator]
            and (owner := self.res_state_owner[resonator]) in [qubit, resonator]
        ):
            self.res_state_owner[resonator] = qubit if owner == resonator else resonator
        else:
            raise CircuitTranspilationError(f"MOVE locus {qubit, resonator} is not allowed.")

    def create_move_instructions(
        self,
        qubit: str,
        resonator: str,
    ) -> Iterable[Instruction]:
        """MOVE instruction(s) to move the state of the given qubit into the given resonator,
        or back to the qubit.

        If the resonator has the state of another qubit in it, or if qubit's state is in another resonator,
        restore them first using additional MOVE instructions.

        Applies the returned MOVE instructions on the tracker state.

        Args:
            qubit: The qubit
            resonator: The resonator

        Yields:
            1, 2 or 3 MOVE instructions.

        """
        # if the resonator has another qubit's state in it, restore it
        owner = self.res_state_owner[resonator]
        if owner not in [qubit, resonator]:
            locus = (owner, resonator)
            self.apply_move(*locus)
            yield Instruction(name=self.move_gate, qubits=locus, args={})

        # if the qubit does not hold its own state, restore it, unless it's in the resonator
        # find where the qubit state is (it can be in at most one resonator)
        res = [r for r, c in self.res_state_owner.items() if c == qubit]  # TODO not efficient
        if res and (holder := res[0]) != resonator:
            locus = (qubit, holder)
            self.apply_move(*locus)
            yield Instruction(name=self.move_gate, qubits=locus, args={})

        # move qubit state to resonator, or back to the qubit
        locus = (qubit, resonator)
        self.apply_move(*locus)
        yield Instruction(name=self.move_gate, qubits=locus, args={})

    def restore_as_move_instructions(
        self,
        resonators: Iterable[str] | None = None,
    ) -> list[Instruction]:
        """MOVE instructions that move the states held in the given resonators back to their qubits.

        Applies the returned MOVE instructions on the tracker state.

        Args:
            resonators: Resonators that (may) hold qubit states that should be moved back to the qubits.
                If ``None``, the states in all known resonators will be returned to the qubits.

        Returns:
            MOVE instructions needed to move the states out of ``resonators`` into
            the qubits they belong to.

        """
        if resonators is None:
            resonators = self.resonators

        instructions: list[Instruction] = []
        for r in resonators:
            q = self.res_state_owner[r]
            # if the state in r does not belong to r, restore it to its owner
            if q != r:
                locus = (q, r)
                instructions.append(Instruction(name=self.move_gate, qubits=locus, args={}))
                self.apply_move(*locus)
        return instructions

    def resonators_holding_qubits(self, qubits: Iterable[str]) -> list[str]:
        """Return the resonators that are holding the state of one of the given qubits.

        Args:
            qubits: The qubits to check.

        Returns:
            Resonators that hold the state of one of ``qubits``.

        """
        # a resonator can only hold the state of a connected qubit, or its own state
        # TODO needs to be made more efficient once we have lots of resonators
        return [r for r, q in self.res_state_owner.items() if q != r and q in qubits]

    def map_resonators_in_locus(self, locus: Iterable[str]) -> Locus:
        """Map any resonators in the given instruction locus into the QPU components whose state is
        currently stored in that resonator.

        If the resonator contains no qubit state at the moment, it is not changed.
        Non-resonator components in the locus are also unchanged.

        Args:
            locus: Instruction locus to map.

        Returns:
            The mapped locus.

        """
        return tuple(self.res_state_owner.get(q, q) for q in locus)

    def find_resolutions(self, inst: Instruction) -> list[Resolution]:
        """Find all the possible resolutions for the given fictional qubit-qubit gate.

        Given a fictional gate G acting on qubits (a, b), finds all resonators r for which the current DQA
        has either G(a, r) and MOVE(b, r), or G(b, r) and MOVE(a, r) available. See :mod:`~iqm.iqm_client.transpile`.

        If ``inst`` is not a known QR gate, returns an empty list.

        Args:
            inst: Circuit instruction applying the fictional qubit-qubit gate G.

        Returns:
            All (gate qubit, move qubit, resonator) triples that can be used to implement ``inst``.

        """
        if (gate_q2r := self.qr_gates_q2r.get(inst.name)) is None:
            return []

        def get_resonators(g: str, m: str) -> set[str]:
            """Resonators r for which we have G(g, r) and MOVE(m, r) available."""
            return gate_q2r.get(g, set()) & self.move_q2r.get(m, set())

        # G is assumed symmetric, hence we may reverse the locus order for more resolutions
        a, b = inst.qubits
        return [(a, b, r) for r in get_resonators(a, b)] + [(b, a, r) for r in get_resonators(b, a)]

    def find_best_resolution(self, inst: Instruction, lookahead: Iterable[Instruction]) -> Resolution | None:
        """Find the best resolution for the fictional qubit-qubit gate instruction ``inst``
        using the available native qubit-resonator gates.

        Given a resolution (g, m, r) for ``inst``, it can be implemented as G(g, r) with
        additional MOVE gates applied first to make sure the state of the qubit m is in r,
        and g is holding its own state.

        Does not change the internal state of the tracker.

        Args:
            inst: instruction to implement
            lookahead: upcoming instructions, to be taken into consideration

        Returns:
            Best resolution for implementing ``inst``, or None iff no resolution could be found.

        """
        resolutions = self.find_resolutions(inst)
        if not resolutions:
            return None

        # We could sequence n ops of lookahead using recursion and use sequence lenght as badness,
        # but that would scale exponentially in n. Instead we do some heuristics:
        # Look ahead until we find the instructions that target the locus qubits next, and include
        # their costs in the badness calculation.
        followers: dict[str, Instruction] = {}
        for follower in lookahead:
            if len(followers) == 2:
                break  # found a follower for both locus qubits
            for q in follower.qubits:
                if q in inst.qubits:
                    followers.setdefault(q, follower)

        def get_badness(res: Resolution, g_holder: str, m_holder: str, r_owner: str) -> int:
            """Badness of the given resolution for implementing a fictional qubit-qubit gate.

            ``g_holder``, ``m_holder`` and ``r_owner`` represent the current QPU state.
            """
            g, m, r = res
            badness = 0
            if g_holder != g:
                badness += 1  # need to move the state back to g from a resonator
            if m_holder == r:
                pass  # m qubit state already in r
            else:
                if r_owner != r:
                    badness += 1  # resonator has some other qubit's state in it, and it must be restored
                if m_holder == m:
                    badness += 1  # need to move the m state to r
                else:
                    badness += 2  # m state is in another resonator, need 2 moves to get it to r
            return badness

        def follower_badness(res: Resolution) -> int:
            """Badness for implementing the follower instruction(s), starting from
            the situation after implementing ``inst`` itself using ``res``.
            """
            g, m, r = res
            g_follower = followers.get(g)
            m_follower = followers.get(m)

            if g_follower == m_follower and g_follower is not None:
                # >=2q gate. 2q gate has the same or reversed locus (no matter since QR gates are assumed symmetric!)
                if g_follower.name == inst.name:
                    # same gate => same resolution works, free
                    return 0
                follower_resolutions = self.find_resolutions(g_follower)
                if follower_resolutions:
                    # different resolvable QR gate
                    if res in follower_resolutions:
                        # same resolution works, free
                        return 0
                    # same resolution not ok, find the cheapest one
                    return min(get_badness(f_res, g, r, m) for f_res in follower_resolutions)
                # not a resolvable QR gate, ignore
                return 0

            badness: int = 0
            if g_follower:
                follower_resolutions = self.find_resolutions(g_follower)
                if follower_resolutions:
                    # g_follower is a 2q gate (sharing g only) that we can resolve
                    if any(f_res[2] != r for f_res in follower_resolutions):
                        # follower can be implemented using a different resonator
                        pass
                    else:
                        # follower needs to use same resonator but different move qubit
                        badness += 2
                else:
                    # Not a resolvable QR gate. For 1q gates on g, qubit state is already there.
                    pass

            if m_follower:
                follower_resolutions = self.find_resolutions(m_follower)
                if follower_resolutions:
                    # m_follower is a 2q gate (sharing m only) that we can resolve
                    if any(f_res[1:] == (m, r) for f_res in follower_resolutions):
                        # follower can be implemented using m as the move qubit and r as the resonator
                        pass
                    elif any(f_res[2] != r for f_res in follower_resolutions):
                        # follower can be implemented using a different resonator, m state must be restored
                        badness += 1
                    else:
                        # follower needs to use same resonator but different move qubit
                        badness += 2
                elif len(m_follower.qubits) == 1:
                    # Not a resolvable QR gate. For 1q gates on m, state must be restored.
                    badness += 1

            return badness

        options = []
        for res in resolutions:
            g, m, r = res
            # badness is the number of extra native instructions we need to implement ``inst``
            # and its followers using this resolution.
            # implementing ``inst`` itself, starting from the current tracker state
            badness: int = get_badness(
                res,
                self.qubit_state_holder.get(g, g),
                self.qubit_state_holder.get(m, m),
                self.res_state_owner[r],
            )

            # implementing the gate's followers, starting from the situation after implementing ``inst`` itself
            badness += follower_badness(res)

            if badness == 0:
                # badness cannot be lower than 0, so this is already an optimal resolution
                return res
            options.append((res, badness))

        # return the best option
        return min(options, key=lambda x: x[1])[0]

    def get_sequence(self, resolution: Resolution, inst: Instruction) -> list[Instruction]:
        """Apply a fictional two-qubit gate using a sequence of native qubit-resonator gates.

        See :mod:`~iqm.iqm_client.transpile`.
        Given the resolution (g, m, r), the gate is implemented as G(g, r), with additional MOVE gates
        applied first to make sure the state of the qubit m is in r, and g is holding its own state.

        Modifies the tracker state to reflect the returned sequence.

        Args:
            resolution: Defines a native gate sequence for implementing ``inst``.
            inst: Fictional qubit-qubit gate as an instruction.

        Returns:
            Sequence of real qubit-resonator gates implementing ``inst``.

        """
        g, m, r = resolution
        seq: list[Instruction] = []
        # does m state need to be moved to the resonator?
        m_holder = self.qubit_state_holder.get(m, m)
        if m_holder != r:
            seq += self.create_move_instructions(m, r)
        # does g state need to be moved to g?
        g_holder = self.qubit_state_holder.get(g, g)
        if g_holder != g:
            seq += self.restore_as_move_instructions([g_holder])
        # apply G(g, r)
        seq.append(inst.model_copy(update={"qubits": (g, r)}))
        return seq

    def insert_moves(
        self,
        instructions: Sequence[Instruction],
        arch: DynamicQuantumArchitecture,
    ) -> list[Instruction]:
        """Convert a simplified architecture circuit into a equivalent Star architecture circuit with
        resonators and MOVE gates.

        Inserts MOVE gates into the circuit and changes the existing instruction loci as needed,
        while updating the state of the tracker object.

        Can also handle circuits that mix the simplified and real architectures.

        Args:
            instructions: The instructions in the circuit, using physical qubit names.
            arch: Real Star quantum architecture we transpile to.

        Raises:
            CircuitTranspilationError: Raised when the circuit contains invalid gates that cannot be
                transpiled using this method.

        Returns:
            Real Star architecture equivalent of ``circuit`` with MOVEs and resonators added.

        """
        # This method can handle real single- and two-qubit gates, real q-r gates including MOVE,
        # and fictional two-qubit gates which it decomposes into real q-r gates.
        new_instructions: list[Instruction] = []

        for idx, inst in enumerate(instructions):
            locus = inst.qubits
            try:
                validate_instruction(architecture=arch, instruction=inst)
                # inst can be applied as is on locus, but we may first need to use MOVEs to make
                # sure the locus qubits contain their states

                if inst.name == self.move_gate:
                    # apply the requested MOVE, closing interfering MOVE sandwiches first
                    new_instructions += self.create_move_instructions(*locus)
                    continue

                # are some of the locus qubits' states currently in a resonator?
                if res_match := self.resonators_holding_qubits(locus):
                    # Some locus qubits do not hold their states, which need to be restored before applying the gate.
                    # NOTE: as a consequence, a barrier closes a MOVE sandwich.
                    new_instructions += self.restore_as_move_instructions(res_match)
                new_instructions.append(inst)

            except CircuitValidationError as e:
                # inst can not be applied to this locus as is
                if inst.name not in self.qr_gates_q2r or any(c in self.resonators for c in locus):
                    raise CircuitTranspilationError(e) from e

                resolution = self.find_best_resolution(inst, instructions[idx + 1 :])
                if resolution is None:
                    raise CircuitTranspilationError(
                        f"Unable to find native gate sequence to enable fictional gate {inst.name} at {locus}."
                        " Try routing the circuit to the simplified architecture first."
                    ) from e

                # implement G using the sequence
                new_instructions += self.get_sequence(resolution, inst)

        return new_instructions


def simplify_architecture(
    arch: DynamicQuantumArchitecture,
    *,
    remove_resonators: bool = True,
) -> DynamicQuantumArchitecture:
    """Converts the given IQM Star quantum architecture into the equivalent simplified quantum architecture.

    See :mod:`iqm.iqm_client.transpile` for the details.

    Adds fictional gates, abstracts away their gate implementations.
    Returns ``arch`` itself if it does not contain computational resonators (in which case nothing will change).

    Args:
        arch: quantum architecture to convert
        remove_resonators: iff False, return the union of the simplified and real architectures

    Returns:
        equivalent quantum architecture with fictional gates

    """
    # NOTE: assumes all qubit-resonator gates have the locus order (q, r)
    if not arch.computational_resonators:
        return arch

    r_set = frozenset(arch.computational_resonators)
    q_set = frozenset(arch.qubits)

    moves: dict[str, set[str]] = {}  # maps resonator r to qubits q for which we have MOVE(q, r) available
    for q, r in arch.gates["move"].loci if "move" in arch.gates else []:
        if q not in q_set or r not in r_set:
            raise ValueError(f"MOVE locus {q, r} is not of the form (qubit, resonator)")
        moves.setdefault(r, set()).add(q)

    def simplify_gate(gate_name: str, gate_info: GateInfo) -> GateInfo:
        """Convert the loci of the given gate"""
        new_loci: dict[str, tuple[Locus, ...]] = {}  # mapping from implementation to its new loci
        # loci for fictional gates, a set because multiple resonators can produce the same fictional locus
        fictional_loci: set[Locus] = set()

        for impl_name, impl_info in gate_info.implementations.items():
            kept_impl_loci: list[Locus] = []  # these real loci we keep for this implementation
            for locus in impl_info.loci:
                if len(locus) == 2:
                    # two-component op
                    q1, r = locus
                    if q1 not in q_set:
                        raise ValueError(f"Unexpected '{gate_name}' locus: {locus}")

                    if r in r_set:
                        if not remove_resonators:
                            kept_impl_loci.append(locus)
                        # involves a resonator, for each G(q1, r), MOVE(q2, r) pair add G(q1, q2) to the simplified arch
                        for q2 in moves.get(r, []):
                            if q1 != q2:
                                fictional_loci.add((q1, q2))
                    else:
                        # does not involve a resonator, keep
                        kept_impl_loci.append(locus)
                # other arities: keep as long as it does not involve a resonator
                elif not set(locus) & r_set:
                    kept_impl_loci.append(locus)
            new_loci[impl_name] = tuple(kept_impl_loci)

        # implementation info is lost in the simplification
        if fictional_loci:
            new_loci["__fictional"] = tuple(fictional_loci)

        return GateInfo(
            implementations={impl_name: GateImplementationInfo(loci=loci) for impl_name, loci in new_loci.items()},
            default_implementation=gate_info.default_implementation,
            override_default_implementation={
                locus: impl_name
                for locus, impl_name in gate_info.override_default_implementation.items()
                if all(c not in r_set for c in locus)
            },
        )

    # create fictional gates, remove real gate loci that involve a resonator
    new_gates: dict[str, GateInfo] = {}
    for gate_name, gate_info in arch.gates.items():
        if gate_name == "move":
            # MOVE gates do not have fictional versions
            if not remove_resonators:
                # keep the gate_info as is
                new_gates[gate_name] = gate_info
            continue
        new_gates[gate_name] = simplify_gate(gate_name, gate_info)

    return DynamicQuantumArchitecture(
        calibration_set_id=arch.calibration_set_id,
        qubits=arch.qubits,
        computational_resonators=[] if remove_resonators else arch.computational_resonators,
        gates=new_gates,
    )


def transpile_insert_moves(
    circuit: Circuit,
    arch: DynamicQuantumArchitecture,
    *,
    existing_moves: ExistingMoveHandlingOptions = ExistingMoveHandlingOptions.KEEP,
    qubit_mapping: dict[str, str] | None = None,
    restore_states: bool = True,
) -> Circuit:
    """Convert a simplified architecture circuit into an equivalent Star architecture circuit with
    resonators and MOVE gates, if needed.

    In the typical use case ``circuit`` has been transpiled to a simplified architecture
    where the resonators have been abstracted away, and this function converts it into
    the corresponding Star architecture circuit.

    It can also handle the case where ``circuit`` already contains MOVE gates and resonators,
    which are treated according to ``existing_moves``, followed by the conversion
    of the two-qubit gates that are not supported by the Star architecture.

    The function does nothing if ``arch`` does not support MOVE gates.

    Args:
        circuit: The circuit to convert.
        arch: Real Star architecture of the target device.
        existing_moves: Specifies how to deal with existing MOVE instructions in ``circuit``, if any.
        qubit_mapping: Mapping of logical qubit names to physical qubit names.
            Can be set to ``None`` if ``circuit`` already uses physical qubit names.
        restore_states: Iff True, all qubit states held in resonators are returned to their qubits
            at the end of the circuit (i.e. all MOVE sandwiches are closed), even when there
            is no computational reason to do so.

    Returns:
        Equivalent Star architecture circuit with MOVEs and resonators added.

    """
    move_gate = _ResonatorStateTracker.move_gate

    # see if the circuit already contains some MOVEs/resonators
    circuit_has_moves = any(i for i in circuit.instructions if i.name == move_gate)
    # can we use MOVEs?
    if move_gate not in arch.gates:
        if circuit_has_moves:
            raise ValueError("Circuit contains MOVE instructions, but the architecture does not support them.")
        # nothing to do (do not validate the circuit)
        return circuit

    tracker = _ResonatorStateTracker.from_dynamic_architecture(arch)

    # add missing QPU components to the mapping (mapped to themselves)
    if qubit_mapping is None:
        qubit_mapping = {}
    for c in set(arch.components) - set(qubit_mapping.values()):
        qubit_mapping[c] = c

    if existing_moves == ExistingMoveHandlingOptions.KEEP:
        # convert to physical qubit names
        phys_instructions = _map_loci(circuit.instructions, qubit_mapping)
        try:
            validate_circuit_moves(
                arch,
                Circuit(name=circuit.name, instructions=phys_instructions, metadata=circuit.metadata),
            )

        except CircuitValidationError as e:
            raise CircuitTranspilationError(e) from e
    else:
        if circuit_has_moves and existing_moves == ExistingMoveHandlingOptions.REMOVE:
            # convert the circuit into a pure simplified architecture circuit
            circuit = transpile_remove_moves(circuit)

        # convert to physical qubit names
        phys_instructions = _map_loci(circuit.instructions, qubit_mapping)

    new_instructions = tracker.insert_moves(phys_instructions, arch)

    if restore_states:
        new_instructions += tracker.restore_as_move_instructions()

    # convert back to logical qubit names
    return Circuit(
        name=circuit.name,
        instructions=_map_loci(new_instructions, qubit_mapping, inverse=True),
        metadata=circuit.metadata,
    )


def transpile_remove_moves(circuit: Circuit) -> Circuit:
    """Convert a Star architecture circuit involving resonators and MOVE gates into an equivalent
    simplified achitecture circuit without them.

    The method assumes that in ``circuit`` a MOVE gate is always used to move a qubit state into a
    resonator before any other gates act on the resonator. If this is not the case, this function
    will not work as intended.

    Args:
        circuit: Star architecture circuit from which resonators and MOVE gates should be removed.

    Returns:
        Equivalent simplified architecture circuit without resonators and MOVEs.

    """
    tracker = _ResonatorStateTracker.from_circuit(circuit)
    new_instructions = []
    for inst in circuit.instructions:
        if inst.name == tracker.move_gate:
            # update the state tracking, drop the MOVE
            tracker.apply_move(*inst.qubits)
        else:
            # map the instruction locus
            new_qubits = tracker.map_resonators_in_locus(inst.qubits)
            new_instructions.append(
                Instruction(name=inst.name, implementation=inst.implementation, qubits=new_qubits, args=inst.args)
            )
    return Circuit(name=circuit.name, instructions=tuple(new_instructions), metadata=circuit.metadata)
