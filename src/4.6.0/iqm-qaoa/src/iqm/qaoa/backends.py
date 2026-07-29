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
"""Backends for calculating expectation values and generating samples from QAOA instances.

The backends are divided into two main categories:

- **EstimatorBackends**: Used for calculating expectation values of Hamiltonians and arbitrary ZZ...Z terms.
- **SamplerBackends**: Used for generating samples.

**Key Features:**

- **Dispatcher Methods**: The :meth:`~iqm.qaoa.backends.EstimatorBackend.estimate` and
  :meth:`~iqm.qaoa.backends.SamplerBackend.sample` methods use ``@singledispatchmethod`` to dynamically route calls to
  the appropriate implementation based on the type of the input QAOA object (e.g.,
  :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA`, :class:`~iqm.qaoa.hubo_qaoa.HUBOQAOA`).
- **Extensibility**: New backends can be added by subclassing :class:`EstimatorBackend` or :class:`SamplerBackend` and
  implementing the required methods.
"""

from __future__ import annotations

from functools import reduce, singledispatchmethod
from math import prod
import operator
from typing import TYPE_CHECKING, Any
import warnings

from dimod import BinaryQuadraticModel
from iqm.qaoa.circuits import (
    TranspilerOption,
    qiskit_circuit,
    qiskit_circuit_hubo,
    quimb_tn,
    quimb_tn_hubo,
    transpiled_circuit,
)
from iqm.qaoa.hubo_qaoa import HUBOQAOA
from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qaoa.transforming_functions import ham_bp_to_ham_operator, ham_bqm_to_ham_operator, positions_to_pauli_string
from iqm.qaoa.transpiler.quantum_hardware import LogQubit
from iqm.qiskit_iqm.iqm_provider import IQMProvider
import numpy as np
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
import quimb as qu

if TYPE_CHECKING:
    from iqm.qaoa.generic_qaoa import QAOA


class EstimatorBackend:
    """The template class for estimator backends, i.e., those calculating the expected value of the Hamiltonian."""

    @singledispatchmethod
    def estimate(self, qaoa_object: QAOA) -> float:
        """Estimate the expected value of the Hamiltonian.

        This method uses :func:`~functools.singledispatchmethod` to dispatch to a type-specific implementation based on
        the runtime type of ``qaoa_object``. If no registered implementation matches, it falls back to
        :meth:`_estimate_unsupported`. Subclasses should override the type-specific private methods (e.g.,
        :meth:`_estimate_qubo`) rather than this one.

        The input ``qaoa_object`` includes the training parameters (:attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`), which
        are typically used in estimation of the energy.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose energy is to be estimated.

        Returns:
            The estimated expected value of the Hamiltonian with the quantum state implied by the QAOA object.

        Raises:
            TypeError: If no registered implementation exists for the type of ``qaoa_object``.

        """
        return self._estimate_unsupported(qaoa_object)

    def _estimate_unsupported(self, qaoa: QAOA) -> float:
        """The fallback method for :meth:`estimate` of backends subclassed from :class:`EstimatorBackend`.

        This is intended to be called when an estimator doesn't support estimating the input ``qaoa_object``.
        """
        raise TypeError(
            f"The estimator {self.__class__.__name__} does not support estimating expectation values for"
            f" {type(qaoa).__name__}."
        )

    def _estimate_qubo(self, qaoa_object: QUBOQAOA) -> float:
        """Calculate the Hamiltonian expectation value for a QUBOQAOA object.

        If not overwritten in a subclass, this defaults to calling :meth:`_estimate_unsupported`.
        """
        return self._estimate_unsupported(qaoa_object)

    def _estimate_hubo(self, qaoa_object: HUBOQAOA) -> float:
        """Calculate the Hamiltonian expectation value for a HUBOQAOA object.

        If not overwritten in a subclass, this defaults to calling :meth:`_estimate_unsupported`.
        """
        return self._estimate_unsupported(qaoa_object)

    # Register hooks.
    # To change behavior, override the private methods (e.g., _estimate_qubo) in subclasses.
    @estimate.register
    def _(self, qaoa_object: QUBOQAOA) -> float:
        return self._estimate_qubo(qaoa_object)

    @estimate.register
    def _(self, qaoa_object: HUBOQAOA) -> float:
        return self._estimate_hubo(qaoa_object)

    @singledispatchmethod
    def estimate_correlations_z(
        self, qaoa_object: QAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""Estimate the expected value of products of Z operators on ``target_qubits``.

        This method uses :func:`~functools.singledispatchmethod` to dispatch to a type-specific implementation based on
        the runtime type of ``qaoa_object``. If no registered implementation matches, it falls back to
        :meth:`_estimate_unsupported`. Subclasses should override the type-specific private methods rather than this
        one.

        The input ``qaoa_object`` includes the training parameters (:attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`), which
        are used in estimation of the correlations. Some estimators (subclasses of :class:`EstimatorBackend`) may
        only be able to estimate the expectation values of at most quadratic products of Z's.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act. For example if one is interested in
                :math:`\langle Z_1 Z_4 Z_5 \rangle`, then ``target_qubits == {1, 4, 5}``. If one is interested in
                multiple different correlations, they may set ``target_qubits`` as a list of sets and get out a list of
                correlations. This is likely to be more efficient than repeatedly calling
                :meth:`estimate_correlations_z` with each one set of qubits at a time.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        Raises:
            TypeError: If no registered implementation exists for the type of ``qaoa_object``.

        """
        return self._estimate_unsupported(qaoa_object)

    def _estimate_correlations_qubo(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        """Calculate the ZZ...Z correlations for a QUBOQAOA object.

        If not overwritten in a subclass, this defaults to calling :meth:`_estimate_unsupported`.
        """
        return self._estimate_unsupported(qaoa_object)

    def _estimate_correlations_hubo(
        self, qaoa_object: HUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        """Calculate the ZZ...Z correlations for a HUBOQAOA object.

        If not overwritten in a subclass, this defaults to calling :meth:`_estimate_unsupported`.
        """
        return self._estimate_unsupported(qaoa_object)

    # Register hooks.
    # To add or change behavior, override the private methods (e.g., _estimate_correlations_qubo) in subclasses.
    @estimate_correlations_z.register
    def _(self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]) -> float | list[float]:
        return self._estimate_correlations_qubo(qaoa_object, target_qubits)

    @estimate_correlations_z.register
    def _(self, qaoa_object: HUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]) -> float | list[float]:
        return self._estimate_correlations_hubo(qaoa_object, target_qubits)


class SamplerBackend:
    """The template for sampler backends, i.e., those returning samples from the QAOA."""

    @singledispatchmethod
    def sample(self, qaoa_object: QAOA, shots: int) -> dict[str, int]:
        """Sample bitstrings from the QAOA circuit.

        This method uses :func:`~functools.singledispatchmethod` to dispatch to a type-specific implementation based on
        the runtime type of ``qaoa_object``. If no registered implementation matches, it falls back to
        :meth:`_sample_unsupported`. Subclasses should override the type-specific private methods rather than this one.

        Args:
            qaoa_object: A :class:`~iqm.qaoa.generic_qaoa.QAOA` object to be sampled from.
            shots: The number of individual samples to take.

        Returns:
            A dictionary of samples. The keys are bitstrings and the values are their counts (which should add up to
            ``shots``)

        Raises:
            TypeError: If no registered implementation exists for the type of ``qaoa_object``.

        """
        return self._sample_unsupported(qaoa_object, shots)

    def _sample_unsupported(self, qaoa: QAOA, shots: int) -> dict[str, int]:
        """The fallback method for :meth:`sample` of backends subclassed from :class:`SamplerBackend`.

        This is intended to be called when an estimator doesn't support estimating the input ``qaoa_object``.
        """
        raise TypeError(f"The sampler {self.__class__.__name__} does not support sampling from {type(qaoa).__name__}.")

    def _sample_qubo(self, qaoa_object: QUBOQAOA, shots: int) -> dict[str, int]:
        """Sample from QUBOQAOA object.

        If not overwritten in a subclass, this defaults to calling :meth:`_sample_unsupported`.
        """
        return self._sample_unsupported(qaoa_object, shots)

    def _sample_hubo(self, qaoa_object: HUBOQAOA, shots: int) -> dict[str, int]:
        """Sample from HUBOQAOA objects.

        If not overwritten in a subclass, this defaults to calling :meth:`_sample_unsupported`.
        """
        return self._sample_unsupported(qaoa_object, shots)

    # Register hooks.
    # To add or change behavior, override the private methods (e.g., _sample_qubo) in subclasses.
    @sample.register
    def _(self, qaoa_object: QUBOQAOA, shots: int) -> dict[str, int]:
        return self._sample_qubo(qaoa_object, shots)

    @sample.register
    def _(self, qaoa_object: HUBOQAOA, shots: int) -> dict[str, int]:
        return self._sample_hubo(qaoa_object, shots)


def _validate_and_normalize_target_qubits(target_qubits: set[LogQubit] | list[set[LogQubit]]) -> list[set[LogQubit]]:
    """Validates and normalizes the variable ``target_qubits``, an input to :meth:`estimate_correlations_z`.

    Does the following two steps:
    1. Checks that ``target_qubits`` is the correct type. That is, either a set of
       :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit` or a list of sets of
       :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`.
    2. In case that ``target_qubits`` is a list of sets of :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`,
       return it. If it is just a set of :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`, returns a
       single-element list containing ``target_qubits`` so that the output of this function is always
       ``list[set[LogQubit]]``.

    Args:
        target_qubits: The variable to be validated and normalized (representing the qubits whose Z-correlations we're
            interested in).

    Returns:
        Normalized ``target_qubits``.

    Raises:
        TypeError: If the input is not the expected type ``set[LogQubit] | list[set[LogQubit]]``.

    """
    if isinstance(target_qubits, set) and all(isinstance(q, LogQubit) for q in target_qubits):
        return [target_qubits]

    elif isinstance(target_qubits, list) and all(
        isinstance(s, set) and all(isinstance(q, LogQubit) for q in s) for s in target_qubits
    ):
        return target_qubits
    else:
        raise TypeError(
            f"Invalid type for target_qubits: {target_qubits!r}.Expected set[LogQubit] or list[set[LogQubit]]."
        )


class EstimatorSingleLayer(EstimatorBackend):
    """Analytical estimator for :math:`p=1` QAOA using the closed-form expressions from :cite:`Ozaeta_2020`."""

    def _estimate_qubo(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian for :math:`p=1` QAOA.

        The function calculates the energy (exp. val. of the Hamiltonian) by adding the expectation values of its
        individual terms expressed through equation (12) in :cite:`Ozaeta_2020`. The calculation includes a constant
        term (coming from the translation of a QUBO problem to a Hamiltonian).

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        Raises:
            ValueError: If the provided :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object has more than 1 layer.

        """
        if qaoa_object.num_layers != 1:
            raise ValueError(f"The number of layers is not 1, but {qaoa_object.num_layers}")

        g, b = qaoa_object.angles  # QAOA angles gamma and beta.
        h_bqm = qaoa_object.hamiltonian_bqm

        energy = 0.0  # To be incremented by the exp. val. of the individual terms in the two following for loops.

        # Linear terms.
        for qb in h_bqm.variables:
            # The expectation value of :math:`\langle Z \rangle` is offloaded into a helper function.
            energy += self._expval_z(qb, g, b, h_bqm) * h_bqm.get_linear(qb)

        # Quadratic terms.
        for i, j in h_bqm.quadratic:
            # The expectation value of :math:`\langle ZZ \rangle` is offloaded into a helper function.
            energy += self._expval_zz(i, j, g, b, h_bqm) * h_bqm.get_quadratic(i, j)

        # Constant offset.
        energy += h_bqm.offset
        return energy

    def _expval_z(self, qb: LogQubit, g: float, b: float, h_bqm: BinaryQuadraticModel) -> float:
        r"""Expectation value of a Z operator on the qubit ``qb``.

        Matches the first term of eq. 12, except for the factor :math:`h_i` (the local field), which is excluded here.

        Args:
            qb: The qubit on which we want to calculate :math:`\langle Z \rangle`.
            g: The gamma angle parameter of the QAOA.
            b: The beta angle parameter of the QAOA.
            h_bqm: The BQM carrying the information about the optimization problem instance.

        Returns:
            The expectation value of :math:`\langle Z \rangle` on the qubit ``qb``.

        """
        hi = h_bqm.get_linear(qb)
        nn = {x[0] for x in h_bqm.iter_neighborhood(qb)}  # The set of nearest neighbours of ``qb``.
        prod_cos = np.prod([np.cos(2 * g * h_bqm.get_quadratic(qb, n)) for n in nn])
        return np.sin(2 * b) * np.sin(2 * g * hi) * prod_cos

    def _expval_zz(self, i: LogQubit, j: LogQubit, g: float, b: float, h_bqm: BinaryQuadraticModel) -> float:
        r"""Expectation value of the operator ZZ acting on qubits ``i`` and ``j``.

        Matches the second term of eq. 12, except for the interaction strength factor :math:`J_{ij}`, which is excluded
        here.

        Args:
            i: One of the qubits on which we calculate :math:`\langle ZZ \rangle`.
            j: The other one of the qubits on which we calculate :math:`\langle ZZ \rangle`.
            g: The gamma angle parameter of the QAOA.
            b: The beta angle parameter of the QAOA.
            h_bqm: The BQM carrying the information about the optimization problem instance.

        Returns:
            The expectation value of :math:`\langle ZZ \rangle` on the qubits ``i`` and ``j``.

        """
        hi = h_bqm.get_linear(i)
        hj = h_bqm.get_linear(j)
        jij = h_bqm.get_quadratic(i, j)

        # NN = nearest neighbours.
        nn_i = {x[0] for x in h_bqm.iter_neighborhood(i)} - {j}  # The NN of i, excluding j.
        nn_j = {x[0] for x in h_bqm.iter_neighborhood(j)} - {i}  # The NN of j, excluding i.
        nn_only_i = nn_i - nn_j - {j}  # The nodes which are NN of i, but not NN of j (or j itself)
        nn_only_j = nn_j - nn_i - {i}  # The nodes which are NN of j, but not NN of i (or i itself)
        nn_both = nn_j - nn_only_j  # The nodes which are NN of both i and j

        # The first product on the first line of expval_cij formula.
        prod_nn_i = np.prod([np.cos(2 * g * h_bqm.get_quadratic(i, k)) for k in nn_i])
        # The second product on the first line of expval_cij formula.
        prod_nn_j = np.prod([np.cos(2 * g * h_bqm.get_quadratic(j, k)) for k in nn_j])
        # The first product on the second line of expval_cij formula.
        prod_only_i = np.prod([np.cos(2 * g * h_bqm.get_quadratic(i, k)) for k in nn_only_i])
        # The second product on the second line of expval_cij formula.
        prod_only_j = np.prod([np.cos(2 * g * h_bqm.get_quadratic(j, k)) for k in nn_only_j])
        # The first product on the last line of expval_cij formula.
        prod_both_plus = np.prod(
            [np.cos(2 * g * (h_bqm.get_quadratic(i, k) + h_bqm.get_quadratic(j, k))) for k in nn_both]
        )
        # The second product on the last line of expval_cij formula.
        prod_both_minus = np.prod(
            [np.cos(2 * g * (h_bqm.get_quadratic(i, k) - h_bqm.get_quadratic(j, k))) for k in nn_both]
        )

        # The entire first line of the expval_cij formula, except for the :math:`J_{ij}` factor.
        first_part = (
            0.5
            * np.sin(4 * b)
            * np.sin(2 * g * jij)
            * (np.cos(2 * g * hi) * prod_nn_i + np.cos(2 * g * hj) * prod_nn_j)
        )

        # The entire second line of the expval_cij formula (except for the :math:`J_{ij}` factor).
        factor1 = 0.5 * np.sin(2 * b) ** 2 * prod_only_i * prod_only_j
        # The entire last line of the expval_cij formula.
        factor2 = np.cos(2 * g * (hi + hj)) * prod_both_plus - np.cos(2 * g * (hi - hj)) * prod_both_minus

        # The expval_cij formula is the difference of the 1st line and the product of the 2nd and 3rd line.
        return first_part - factor1 * factor2

    def _estimate_correlations_qubo(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        This works only if the set(s) in ``target_qubits`` are of size at most 2. In case of a set of two qubits without
        an interaction in the BQM, it adds an interaction of strength 0, so that they become neighbors in the BQM.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        Raises:
            ValueError: If the number of layers of the QAOA is not 1.
            ValueError: If the weight of the operator whose exp. value we are interested in (i.e., the number of qubits
                it affects) is more than 2.

        """
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)

        if qaoa_object.num_layers != 1:
            raise ValueError(f"The number of layers is not 1, but {qaoa_object.num_layers}")
        g, b = qaoa_object.angles

        # The variable to be returned.
        list_of_correlations: list[float] = []

        for qubit_set in target_qubits:
            if len(qubit_set) == 0:
                result = 0.0
            elif len(qubit_set) == 1:
                qb = next(iter(qubit_set))
                result = self._expval_z(qb, g=g, b=b, h_bqm=qaoa_object.hamiltonian_bqm)
            elif len(qubit_set) == 2:  # noqa: PLR2004
                qbs = list(qubit_set)
                aux_bqm = qaoa_object.hamiltonian_bqm.copy()
                # If there's no interaction between `qbs`, add it to make the method `_expval_zz` work.
                aux_bqm.add_quadratic(qbs[0], qbs[1], 0)
                result = self._expval_zz(qbs[0], qbs[1], g=g, b=b, h_bqm=aux_bqm)
            else:
                raise ValueError("The ``EstimatorSingleLayer`` can only calculate expectation values of Z or ZZ.")
            list_of_correlations.append(result)

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations


class EstimatorStateVector(EstimatorBackend):
    """Exact estimator using Qiskit statevector simulation.

    Supports both QUBOQAOA and HUBOQAOA with arbitrary number of layers, but scales exponentially with the number of
    qubits.
    """

    def _estimate_qubo(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian from running state-vector simulation in :mod:`qiskit`.

        Builds a :class:`~qiskit.circuit.QuantumCircuit` for the QAOA and runs the statevector simulation of the
        circuit, calculating the expectation value of the energy from the statevector. The calculation includes
        a constant term (coming from the translation of a QUBO problem to a Hamiltonian).

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        """
        qc = qiskit_circuit(qaoa_object, measurements=False)
        statevector = Statevector.from_instruction(qc)
        statevector = statevector.reverse_qargs()
        observable = ham_bqm_to_ham_operator(qaoa_object.hamiltonian_bqm)
        expectation_value = statevector.expectation_value(observable)
        return expectation_value.real

    def _estimate_correlations_qubo(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        Using statevector simulator, calculating any expectation value exactly is relatively straightforward.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        """
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)

        qc = qiskit_circuit(qaoa_object, measurements=False)
        statevector = Statevector.from_instruction(qc).reverse_qargs()

        # The variable to be returned.
        list_of_correlations: list[float] = []

        for qubit_set in target_qubits:
            # We off-load creating the operator whose exp. value we are interested in.
            qubit_index_set = {qaoa_object.hamiltonian_bqm.variables.index(qb) for qb in qubit_set}
            observable = SparsePauliOp(
                positions_to_pauli_string(qubit_index_set, qaoa_object.num_qubits), np.array(1.0)
            )
            expectation_value = statevector.expectation_value(observable)
            list_of_correlations.append(expectation_value.real)

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations

    def _estimate_hubo(self, qaoa_object: HUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian from running state-vector simulation in :mod:`qiskit`.

        Builds a :class:`~qiskit.circuit.QuantumCircuit` for the QAOA and runs the statevector simulation of the
        circuit, calculating the expectation value of the energy from the statevector.

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.hubo_qaoa.HUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        """
        qc = qiskit_circuit_hubo(qaoa_object, measurements=False)
        statevector = Statevector.from_instruction(qc)
        statevector = statevector.reverse_qargs()
        observable = ham_bp_to_ham_operator(qaoa_object.hamiltonian_bp)
        expectation_value = statevector.expectation_value(observable)
        return expectation_value.real

    def _estimate_correlations_hubo(
        self, qaoa_object: HUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        Using statevector simulator, calculating any expectation value exactly is relatively straightforward.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.hubo_qaoa.HUBOQAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        """
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)

        qc = qiskit_circuit_hubo(qaoa_object, measurements=False)
        statevector = Statevector.from_instruction(qc).reverse_qargs()

        # The variable to be returned.
        list_of_correlations: list[float] = []

        for qubit_set in target_qubits:
            # We off-load creating the operator whose exp. value we are interested in.
            qubit_index_set = {ind for ind, var in enumerate(qaoa_object.problem.sorted_vars) if var in qubit_set}
            observable = SparsePauliOp(
                positions_to_pauli_string(qubit_index_set, qaoa_object.num_qubits), np.array(1.0)
            )
            expectation_value = statevector.expectation_value(observable)
            list_of_correlations.append(expectation_value.real)

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations


class EstimatorFromSampler(EstimatorBackend):
    """The estimator class for calculating the expectation value using counts obtained from a sampler.

    Takes an instance of a subclass of :class:`SamplerBackend` and uses it to generate samples from the QAOA.
    These energy of these samples is then calculated classically and averaged-out to produce an estimate of
    the expectation value of the Hamiltonian. If ``cvar`` is provided, the estimator returns not the average of
    the energies, but its CVaR at the ``cvar`` threshold.

    Args:
        sampler: The sampler to produce the samples.
        shots: The number of shots that should be produced with the sampler.
        cvar: The threshold used to calculate CVaR (if provided).

    Raises:
        ValueError: If ``cvar`` is provided, but it's not between 0 (excluded) and 1 (included).

    """

    def __init__(self, sampler: SamplerBackend, shots: int, cvar: float | None = None) -> None:
        super().__init__()
        self.sampler = sampler
        self.shots = shots
        if cvar is not None:
            if not 0 < cvar <= 1:
                raise ValueError(
                    f"The provided ``cvar`` must be between 0 and 1 (0 excluded, 1 included). It is {cvar}"
                )
            self.cvar = cvar
        else:
            self.cvar = 1  # CVaR threshold of 1 corresponds to normal average.

    def _estimate_qubo(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian by sampling from the QAOA circuit.

        Uses the sampler provided at initialization to sample from the QAOA circuit and then calculates the expectation
        value from the counts.

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The average energy of the sampled bitstrings (to serve as estimation of the expectation value).

        """
        counts = self.sampler.sample(qaoa_object, self.shots)
        return qaoa_object.problem.cvar(counts, self.cvar)

    def _estimate_correlations_qubo(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        The correlations are picked out from the counts. Each bitstring contributes to the exp. value as follows:
        1. The positions in the bitstrings corresponding to ``target_qubits`` are located.
        2. The values at the picked positions are transformed as `"0" -> 1` and `"1" -> -1`.
        3. These values are multiplied together.
        4. The results for all bitstrings are averaged-out (weighted by their corresponding counts).

        Examples
        --------
        +---------------+---------------------+----------------------------------------+
        | Bitstring     | ``target_qubits``   | Contribution of this bitstring         |
        +===============+=====================+========================================+
        |``"011100001"``| :math:`\{3, 6, 8\}` | :math:`(-1)\cdot(1)\cdot(-1) = 1`      |
        +---------------+---------------------+----------------------------------------+
        |``"011100001"``|  :math:`\{0, 1\}`   | :math:`(1)\cdot(-1) = -1`              |
        +---------------+---------------------+----------------------------------------+

        Args:
            qaoa_object: The :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        """  # noqa: D416  # Silence warnings from building docs.
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)
        target_qubits_idx = [
            {qaoa_object.hamiltonian_bqm.variables.index(qb) for qb in qb_set} for qb_set in target_qubits
        ]
        counts = self.sampler.sample(qaoa_object, self.shots)
        return self._z_cor_from_counts(target_qubits_idx, counts)

    def _z_cor_from_counts(self, target_qubits_idx: list[set[int]], counts: dict[str, int]) -> float | list[float]:
        """Common logic for :meth:`estimate_correlations_z_qubo` and :meth:`estimate_correlations_z_hubo`."""
        # The variable to be returned.
        list_of_correlations: list[float] = []
        for qubit_set_idx in target_qubits_idx:
            cum_sum: float = 0
            number_of_measurements = 0

            for bin_str, counter in counts.items():
                # Contribution of one bitstring (multiplied by the respective count).
                cum_sum += prod(1 if bin_str[qb_idx] == "0" else -1 for qb_idx in qubit_set_idx) * counter
                number_of_measurements += counter

            if number_of_measurements == 0:
                raise ValueError("There are no counts. The expected value can't be averaged.")

            list_of_correlations.append(cum_sum / number_of_measurements)

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations

    def _estimate_hubo(self, qaoa_object: HUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian by sampling from the QAOA circuit.

        Uses the sampler provided at initialization to sample from the QAOA circuit and then calculates the expectation
        value from the counts.

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.hubo_qaoa.HUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The average energy of the sampled bitstrings (to serve as estimation of the expectation value).


        """
        counts = self.sampler.sample(qaoa_object, self.shots)
        return qaoa_object.problem.cvar(counts, self.cvar)

    def _estimate_correlations_hubo(
        self, qaoa_object: HUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        The correlations are picked out from the counts. Each bitstring contributes to the exp. value as follows:
        1. The positions in the bitstrings corresponding to ``target_qubits`` are located.
        2. The values at the picked positions are transformed as `"0" -> 1` and `"1" -> -1`.
        3. These values are multiplied together.
        4. The results for all bitstrings are averaged-out (weighted by their corresponding counts).

        Examples
        --------
        +---------------+---------------------+----------------------------------------+
        | Bitstring     | ``target_qubits``   | Contribution of this bitstring         |
        +===============+=====================+========================================+
        |``"011100001"``| :math:`\{3, 6, 8\}` | :math:`(-1)\cdot(1)\cdot(-1) = 1`      |
        +---------------+---------------------+----------------------------------------+
        |``"011100001"``|  :math:`\{0, 1\}`   | :math:`(1)\cdot(-1) = -1`              |
        +---------------+---------------------+----------------------------------------+

        Args:
            qaoa_object: The :class:`~iqm.qaoa.hubo_qaoa.HUBOQAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        """  # noqa: D416  # Silence warnings from building docs.
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)

        target_qubits_idx = [
            {ind for ind, var in enumerate(qaoa_object.problem.sorted_vars) if var in qubit_set}
            for qubit_set in target_qubits
        ]

        counts = self.sampler.sample(qaoa_object, self.shots)
        return self._z_cor_from_counts(target_qubits_idx, counts)


class EstimatorQUIMB(EstimatorBackend):
    """The estimator class for calculating the expectation value using the tensor network package :mod:`quimb`."""

    CRIT_DEG = 3  # The maximum degree for which QUIMB runs somewhat tolerably fast.

    def _estimate_qubo(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian by contracting the RCC tensor networks in :mod:`quimb`.

        Uses :func:`~iqm.qaoa.circuits.quimb_tn` to build a :class:`~quimb.tensor.circuit.Circuit`. This object
        represents the QAOA circuit, so it can be used to calculate expectation values (using the function
        :meth:`~quimb.tensor.circuit.Circuit.local_expectation`). The local expectation values are added to get
        the expectation value of the full Hamiltonian. The calculation includes a constant term (coming from
        the translation of a QUBO problem to a Hamiltonian).

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        """
        degrees_arr = qaoa_object.hamiltonian_bqm.degrees(array=True)
        if isinstance(degrees_arr, np.ndarray) and np.mean(degrees_arr) > self.CRIT_DEG:
            warnings.warn(
                f"The average degree is higher than {self.CRIT_DEG}, the Quimb-based estimator might be very slow.",
                stacklevel=2,
            )
        energy = 0

        tn = quimb_tn(qaoa_object)

        for q1, q2 in qaoa_object.hamiltonian_bqm.quadratic:
            to_measure = qu.pauli("Z") & qu.pauli("Z")
            energy += tn.local_expectation(
                to_measure,
                (qaoa_object.hamiltonian_bqm.variables.index(q1), qaoa_object.hamiltonian_bqm.variables.index(q2)),
            ) * qaoa_object.hamiltonian_bqm.get_quadratic(q1, q2)
        for q1 in qaoa_object.hamiltonian_bqm.variables:
            to_measure = qu.pauli("Z")
            energy += tn.local_expectation(
                to_measure, qaoa_object.hamiltonian_bqm.variables.index(q1)
            ) * qaoa_object.hamiltonian_bqm.get_linear(q1)

        # The energy should already be real.
        return energy.real + qaoa_object.hamiltonian_bqm.offset

    def _estimate_correlations_qubo(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        The correlations are calculated natively for QUIMB, as a contraction of tensor networks, very similarly to how
        the expectation value of the Hamiltonian is estimated in :meth:`estimate`.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        """
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)

        if (
            isinstance(degrees_arr := qaoa_object.hamiltonian_bqm.degrees(array=True), np.ndarray)
            and np.mean(degrees_arr) > self.CRIT_DEG
        ):
            warnings.warn(
                f"The average degree is higher than {self.CRIT_DEG}, the Quimb-based estimator might be very slow.",
                stacklevel=2,
            )

        tn = quimb_tn(qaoa_object)

        list_of_correlations: list[float] = []
        for qubit_set in target_qubits:
            qubit_idx_set = {qaoa_object.hamiltonian_bqm.variables.index(qb) for qb in qubit_set}
            # Construct ``qu.pauli("Z") & qu.pauli("Z") & ... & qu.pauli("Z")`` correct number of times.
            to_measure = reduce(operator.and_, (qu.pauli("Z") for _ in range(len(qubit_set))))
            correlation = tn.local_expectation(to_measure, qubit_idx_set)

            list_of_correlations.append(correlation.real)  # The correlation should already be real.

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations

    def _estimate_hubo(self, qaoa_object: HUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian by contracting the RCC tensor networks in :mod:`quimb`.

        Uses :func:`~iqm.qaoa.circuits.quimb_tn_hubo` to build a :class:`~quimb.tensor.circuit.Circuit`. This object
        represents the QAOA circuit, so it can be used to calculate expectation values (using the function
        :meth:`~quimb.tensor.circuit.Circuit.local_expectation`). The local expectation values are added to get
        the expectation value of the full Hamiltonian. The calculation includes a constant term (coming from
        the translation of a HUBO problem to a Hamiltonian).

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.hubo_qaoa.HUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        """
        energy = 0

        tn = quimb_tn_hubo(qaoa_object)

        for qubit_set, coeff in qaoa_object.hamiltonian_bp.items():
            if not qubit_set:  # Skip the constant term.
                continue
            qubit_idx_set = {qaoa_object.problem.sorted_vars.index(qb) for qb in qubit_set}
            # Construct ``qu.pauli("Z") & qu.pauli("Z") & ... & qu.pauli("Z")`` correct number of times.
            to_measure = reduce(operator.and_, (qu.pauli("Z") for _ in range(len(qubit_set))))
            correlation = tn.local_expectation(to_measure, qubit_idx_set)

            energy += correlation * coeff

        return energy.real + qaoa_object.hamiltonian_bp.get(frozenset(), 0)  # Add the constant term, if present.


class SamplerRandomBitstrings(SamplerBackend):
    """A sampler that ignores the QAOA and just produces random bitstrings of the correct length."""

    def sample(self, qaoa_object: QAOA, shots: int, seed: int | None = None) -> dict[str, int]:
        """Produce random bitstrings to act as samples from the QAOA.

        The ``qaoa_object`` is used only to get the number of qubits (which corresponds to the length of
        the bitstrings). The number of uniformly random bitstrings produced is ``shots`` and they are arranged in
        a dictionary just like counts from a :mod:`qiskit` measurement.

        This method overrides the dispatching method :meth:`~iqm.qaoa.backends.SamplerBackend.sample` of
        :class:`~iqm.qaoa.backends.SamplerBackend` because there is no point in doing any kind of dispatching here.

        Args:
            qaoa_object: The QAOA object, only used to get the number of qubits.
            shots: The number of random strings to generate.
            seed: The seed to be used in the random bitstring generation. Optional parameter.

        Returns:
            A dictionary whose keys are the produced random bitstrings and values their frequencies in the random set.

        """
        return _random_sample_generator(shots, qaoa_object.num_qubits, seed)


def _random_sample_generator(shots: int, n: int, seed: int | None) -> dict[str, int]:
    """Generate random samples.

    Args:
        shots: The number of total counts of bitstrings to generate.
        n: The length of the bitstrings.
        seed: An optional seed to make the randomness deterministic.

    Returns:
        A dictionary whose keys are the random bitstrings generated and values their counts. The sum of the values
        should equal to ``shots``.

    """
    counts: dict[str, int] = {}

    rng = np.random.default_rng(seed)
    for _ in range(shots):
        bitstring = "".join(map(str, rng.integers(0, 2, size=n)))
        if bitstring in counts:
            counts[bitstring] += 1
        else:
            counts[bitstring] = 1
    return counts


class SamplerSimulation(SamplerBackend):
    """A sampler that simulates the QAOA circuit in :mod:`qiskit`.

    Some simulators may need the circuit to be transpiled, so optionally a :class:`~iqm.qaoa.circuits.TranspilerOption`
    may be provided.

    Args:
        simulator: A simulator which the simulation can be run on.
        transpiler: A transpilation (routing) strategy to use, if applicable.

    """

    # The type hint suggests that `simulator` can be any `BackendV2`, but it should be a simulator, not a real QC.
    # `BackendV2` is the nearest common ancestor of `AerSimulator` and `IQMFakeBackend`, which are the two main
    # backends that we might want use here, so it's used as a type hint.
    def __init__(
        self, simulator: BackendV2 | None = None, transpiler: TranspilerOption | None = None, **kwargs: Any
    ) -> None:
        if simulator is None:
            simulator = AerSimulator(method="statevector")
        self.simulator = simulator
        self.transpiler = transpiler
        self.transpile_kwargs = kwargs

    def _sample_qubo(self, qaoa_object: QUBOQAOA, shots: int) -> dict[str, int]:
        """Samples from the QAOA using a simulation.

        The dictionary of counts is obtained from `qiskit` and then the bitstrings are **reversed**, so they don't use
        the `qiskit` convention of the first bit being on the right of the bitstring.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object, to be sampled from.
            shots: The number of samples (measurements) to take.

        Returns:
            A dictionary whose keys are the measured bitstrings and values their frequencies in the results.

        """
        qc = transpiled_circuit(
            qaoa_object, backend=self.simulator, transpiler=self.transpiler, **self.transpile_kwargs
        )
        job = self.simulator.run(qc, shots=shots)
        counts_from_job = job.result().get_counts()
        # Qiskit somehow reverses the order of the bitstrings.
        counts_correctly_ordered = {key[::-1]: value for key, value in counts_from_job.items()}
        return counts_correctly_ordered


class SamplerResonance(SamplerBackend):
    """A sampler that runs the circuit on IQM Resonance and returns the result.

    Args:
        token: The API token to be used to connect to IQM Resonance.
        server_url: The URL to the quantum computer (defaults to Garnet).
        transpiler: The transpiling strategy to be used when building the quantum circuit for the QC. Defaults to
            `TranspilerOption.SPARSE`
        kwargs: The keyword arguments to be passed to the inner call to Qiskit
            :func:`~qiskit.provider.transpiler.transpile`.

    """

    def __init__(
        self,
        token: str,
        server_url: str = "https://resonance.iqm.tech/garnet",
        transpiler: TranspilerOption = TranspilerOption.SPARSE,
        **kwargs: Any,
    ) -> None:
        self.iqm_backend = IQMProvider(server_url, token=token).get_backend()
        self.token = token
        self.transpiler = transpiler
        self.transpile_kwargs = kwargs

    def _sample_qubo(self, qaoa_object: QUBOQAOA, shots: int) -> dict[str, int]:
        """Samples from the QAOA on a quantum computer via IQM Resonance.

        First, it creates a :class:`~qiskit.circuit.QuantumCircuit` (using a custom transpilation approach) and then
        sends it to IQM Resonance. The dictionary of counts is obtained from Qiskit and then the bitstrings are
        **reversed**, so they don't use the Qiskit convention of the first bit being on the right of the bitstring.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object, to be sampled from.
            shots: The number of samples (measurements) to take.

        Returns:
            A dictionary whose keys are the measured bitstrings and values their frequencies in the results.

        """
        qc = transpiled_circuit(qaoa_object, self.iqm_backend, transpiler=self.transpiler, **self.transpile_kwargs)
        job = self.iqm_backend.run(qc, shots=shots)
        counts_from_job = job.result().get_counts()
        # Qiskit somehow reverses the order of the bitstrings.
        counts_correctly_ordered = {key[::-1]: value for key, value in counts_from_job.items()}
        return counts_correctly_ordered
