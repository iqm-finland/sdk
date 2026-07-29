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

"""Module containing the implementation of the parity twine network strategies based on :cite:`Dreier_2025`."""

from __future__ import annotations

import copy as cp
from itertools import pairwise
from typing import TypeAlias, cast
import warnings

from dimod import BinaryQuadraticModel
from iqm.qaoa.transpiler.quantum_hardware import QPU, HardQubit, LogQubit
from iqm.qaoa.transpiler.routing import BaseMapping, BaseRouting
from iqm.qaoa.transpiler.sparse.two_color_mapper import _greedy_longest_path_with_backtracking
from qiskit import QuantumCircuit

# Type aliases relevant to PTN to make typing a bit more self-documenting.
LineQubit: TypeAlias = int
r"""
Labels of the lines in the PTN circuit.

The :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`\s and their parities get mapped onto :class:`LineQubit`\s
and these in turn get mapped onto :class:`~iqm.qaoa.transpiler.quantum_hardware.HardQubit`\s.
"""
Parity: TypeAlias = set[LogQubit]
"""
Represents the problem qubits whose parity is encoded in one :class:`LineQubit`.
"""
PTNLayer: TypeAlias = list[tuple[LineQubit, LineQubit]]
"""
Represents a Parity Twine Chain (PTC), see fig. 2.(b) in :cite:`Dreier_2025`.
"""


class LineMappingPTN(BaseMapping):
    r"""The mapping class for line PTN routing algorithm.

    The mapping instance for line PTN routing has the responsibility to keep track of two important attributes:
    :attr:`hard2line` and :attr:`line_of_parities`. The former keeps track of mapping hardware qubits on the QPU into
    :class:`LineQubit`\s of the PTN. The latter keeps track what parity is mapped onto which :class:`LineQubit`.

    Args:
        qpu: The QPU onto which we do the routing.
        problem_bqm: The BQM of the problem, only necessary to extract variable names.
        line_of_hw_qubits: A chain of HW qubits on the ``qpu``. If not provided, it is found with a greedy algorithm.

    Raises:
        ValueError: If the provided ``line_of_hw_qubits`` contains duplicates.
        ValueError: If the provided ``line_of_hw_qubits`` is too short.
        ValueError: If the provided ``line_of_hw_qubits`` is not a chain on the QPU (i.e., two consecutive HW qubits
            are not connected on the QPU).

    Warnings:
        UserWarning: If the provided ``line_of_hw_qubits`` is longer than the number of problem variables. In that case
            it is truncated.

    """

    def __init__(
        self,
        qpu: QPU,
        problem_bqm: BinaryQuadraticModel,
        line_of_hw_qubits: list[HardQubit] | None = None,
    ) -> None:
        super().__init__(qpu, problem_bqm)

        num_vars = problem_bqm.num_variables

        if line_of_hw_qubits is None:
            line_of_hw_qubits = _greedy_longest_path_with_backtracking(qpu.hardware_graph)[:num_vars]
        else:
            if len(line_of_hw_qubits) != len(set(line_of_hw_qubits)):
                raise ValueError("The provided line of HW qubits contains duplicates.")
            if len(line_of_hw_qubits) < num_vars:
                raise ValueError(
                    f"The provided list of HW qubits {line_of_hw_qubits} is shorter "
                    f"than the number of problem variables {num_vars}."
                )
            elif len(line_of_hw_qubits) > num_vars:
                line_of_hw_qubits = line_of_hw_qubits[:num_vars]
                warnings.warn(
                    f"The provided line of HW qubits {line_of_hw_qubits} is longer than the number of problem "
                    f"variables. Only the first {num_vars} HW qubits will be used.",
                    stacklevel=2,
                )

            for qb0, qb1 in pairwise(line_of_hw_qubits):
                if not qpu.hardware_graph.has_edge(qb0, qb1):
                    raise ValueError("The provided line of HW qubits contains consecutive un-connected qubits.")

        self._line2hard: list[HardQubit] = line_of_hw_qubits
        self._line_of_parities: list[Parity] = [{cast(int, qb)} for qb in problem_bqm.variables]

        hard2line_used_qbs = {hw_qb: line_qb for line_qb, hw_qb in enumerate(self._line2hard)}
        # Build the full dictionary with default ``None`` for missing qubits.
        self._hard2line = {hw_qb: hard2line_used_qbs.get(hw_qb, None) for hw_qb in qpu.qubits}

    @property
    def hard2line(self) -> dict[HardQubit, LineQubit | None]:
        """The mapping from hardware qubits to line qubits in the PTN circuit."""
        return self._hard2line

    @property
    def line_of_parities(self) -> list[Parity]:
        """The line of parity information carried by the qubits of the PTN circuit."""
        return self._line_of_parities

    def update(self, layer: PTNLayer) -> None:
        """Update the mapping from the input ``layer``.

        Goes through the ``layer`` and applies each DCNOT therein. The ``layer`` represents a Parity Twine Chain, i.e.,
        one chain of DCNOTs going through the line qubits.

        Args:
            layer: The layer to be applied to the mapping, referred to as PTC (Parity Twine Chain).

        """
        for dcnot in layer:
            self.dcnot(*dcnot)

    def cnot(self, qb0: LineQubit, qb1: LineQubit) -> None:
        """Applies the CNOT gate to the mapping at line qubits ``qb0`` and ``qb1``.

        The gate acts on ``qb1``, controlled by ``qb0``. This modifies ``self._line_of_parities`` in-place.

        Args:
            qb0: The control :class:`LineQubit` of the CNOT gate.
            qb1: The target :class:`LineQubit` of the CNOT gate.

        """
        self._line_of_parities[qb1] ^= self._line_of_parities[qb0]

    def dcnot(self, qb0: LineQubit, qb1: LineQubit) -> None:
        """Applies the DCNOT gate to the mapping at line qubits ``qb0`` and ``qb1``.

        A DCNOT gate is made up of two CNOT gates. Note that the first CNOT gate acts on ``qb0``, controlled by ``qb1``.
        The second CNOT gate acts on ``qb1``, controlled by ``qb0``.

        Args:
            qb0: The first :class:`LineQubit` on which the DCNOT gate acts.
            qb1: The second :class:`LineQubit` on which the DCNOT gate acts.

        """
        self.cnot(qb0=qb1, qb1=qb0)
        self.cnot(qb0=qb0, qb1=qb1)


class BaseRoutingPTN(BaseRouting):
    """Base class for the Parity Twine Network routing (PTN)."""

    def __init__(
        self, problem_bqm: BinaryQuadraticModel, qpu: QPU, initial_mapping: LineMappingPTN | None = None
    ) -> None:
        super().__init__(problem_bqm, qpu)

        if initial_mapping is None:
            self.initial_mapping = LineMappingPTN(qpu, problem_bqm)
        else:
            self.initial_mapping = initial_mapping

        self.mapping = cp.deepcopy(self.initial_mapping)

        self._layers: list[list[tuple[LineQubit, LineQubit]]]

    @property
    def layers(self) -> list[list[tuple[LineQubit, LineQubit]]]:
        """The list of layers of the routing object.

        A layer here corresponds to a Parity Twine chain and therefore it can NOT be executed in one circuit layer.
        The purpose of having layers like this is mostly debugging and structuring the code.
        """
        return self._layers


class LineRoutingPTN(BaseRoutingPTN):
    """Subclass implementing the PTN line strategy."""

    def __init__(
        self, problem_bqm: BinaryQuadraticModel, qpu: QPU, initial_mapping: LineMappingPTN | None = None
    ) -> None:
        super().__init__(problem_bqm, qpu, initial_mapping)

        self._layers = self._build_triangle(self.problem.num_variables)

    def _build_triangle(self, n: int) -> list[list[tuple[LogQubit, LogQubit]]]:
        first_layer = [(i, i + 1) for i in range(n - 1)]
        return [first_layer[: len(first_layer) - k] for k in range(len(first_layer))]

    def _apply_ptc(self, line_qb0: LineQubit, line_qb1: LineQubit) -> None:
        for step in range(line_qb0, line_qb1):
            self.mapping.dcnot(step, step + 1)

    def build_qiskit(self, betas: list[float], gammas: list[float], measurement: bool = True) -> QuantumCircuit:
        r"""Build the QAOA circuit from the routing (``self``) in :mod:`qiskit`.

        The :class:`~iqm.qaoa.transpiler.ptn.LineRoutingPTN` (``self``) contains all the information needed to create
        the phase separator part of the QAOA circuit. This method builds the rest of the circuit from it, i.e.:

        1. It initializes the qubits in the :math:`| + >` state by applying the Hadamard gate to all of them.
        2. It applies local fields (*RZ* gates).
        3. It applies the PTN layers of DCNOT, mapping the qubit parities to the line qubits, applying interactions
           (*RZ* gates) in between the layers.
        4. It applies the driver.
        5. It repeats steps 2-5 until it uses up all ``betas`` and ``gammas``.
        6. It applies the measurements and barrier before them.

        Args:
            betas: The QAOA parameters to be used in the driver (*RX* gate).
            gammas: The QAOA parameters to be used in the phase separator (*RZ* and *RZZ* gates).
            measurement: Should the circuit contain a layer of measurements or not?

        Returns:
            A complete QAOA :class:`~qiskit.circuit.QuantumCircuit`.

        Raises:
            ValueError: If lengths of ``gammas`` and ``betas`` are not the same.

        """
        if len(betas) != len(gammas):
            raise ValueError("The lengths of ``gammas`` and ``betas`` need to be the same!")
        p = len(betas)

        mapping = cp.deepcopy(self.initial_mapping)
        layers = cp.deepcopy(self.layers)

        # Build the quantum circuit on line qubits and eventually map it onto hardware qubits.
        # Building the circuit on line qubits is very intuitive.
        qc_line_qbs = QuantumCircuit(len(self.mapping._line2hard), len(self.mapping._line2hard))

        # Prepare uniform superposition.
        qc_line_qbs.h(range(len(self.mapping._line2hard)))

        for qaoa_layer in range(p):
            # Phase separator.
            for line_qb, log_qb_set in enumerate(mapping._line_of_parities):
                (log_qb,) = log_qb_set
                local_term = self.problem.get_linear(log_qb)
                if local_term != 0:
                    qc_line_qbs.rz(2 * gammas[qaoa_layer] * local_term, line_qb)

            for ptn_layer in layers:
                mapping.update(ptn_layer)
                for dcnot in ptn_layer:
                    qc_line_qbs.cx(dcnot[1], dcnot[0])
                    qc_line_qbs.cx(dcnot[0], dcnot[1])
                    v1, v2 = tuple(mapping._line_of_parities[dcnot[0]])
                    int_strength = self.problem.get_quadratic(v1, v2, default=0.0)
                    if int_strength != 0:
                        qc_line_qbs.rz(2 * gammas[qaoa_layer] * int_strength, dcnot[0])

            # Restore variables.
            for line_qb in range(len(mapping._line2hard) - 1, 0, -1):
                qc_line_qbs.cx(line_qb, line_qb - 1)
                mapping.cnot(line_qb, line_qb - 1)

            for line_qb in range(len(mapping._line2hard) - 1, -1, -1):
                qc_line_qbs.rx(2 * betas[qaoa_layer], line_qb)

            mapping._line_of_parities = mapping._line_of_parities[::-1]
            qc_line_qbs = qc_line_qbs.reverse_bits()

        if measurement:
            qc_line_qbs.barrier()
            for line_qb, log_qb_set in enumerate(mapping._line_of_parities):
                (log_qb,) = log_qb_set
                qc_line_qbs.measure(line_qb, log_qb)

        # Below we create the version of the quantum circuit on the hardware qubits.
        qc_hard_qbs = QuantumCircuit(len(mapping.hard_qbs), qc_line_qbs.num_clbits)
        qc_hard_qbs.compose(qc_line_qbs, qubits=mapping._line2hard, inplace=True)

        return qc_hard_qbs


def ptn_router(problem_bqm: BinaryQuadraticModel, qpu: QPU, strategy: str = "Line") -> BaseRoutingPTN:
    """Construct a routing object for the ParityTwineNetwork (PTN).

    Depending on the chosen strategy, this function returns an appropriate
    subclass of `BaseRoutingPTN` configured with the given problem BQM.

    Args:
        problem_bqm: The binary quadratic model representing the optimization problem.
        qpu: QPU instance.
        strategy: The routing strategy to apply. Currently supported: "Line".

    Returns:
        A `BaseRoutingPTN` subclass implementing the requested routing strategy.

    Raises:
        ValueError: If an unsupported strategy is provided.

    """
    if strategy == "Line":
        return LineRoutingPTN(problem_bqm, qpu=qpu, initial_mapping=None)

    else:
        raise ValueError(f"Unknown ParityTwineNetwork strategy: {strategy}")
