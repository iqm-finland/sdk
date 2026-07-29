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
"""The module for various backends for calculating the expectation value / samples from QAOA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from math import prod
import operator
import random
from typing import TYPE_CHECKING, Any
import warnings

from dimod import BinaryQuadraticModel
from iqm.qaoa.circuits import TranspilerOption
from iqm.qaoa.transpiler.quantum_hardware import LogQubit
import numpy as np
from qiskit.providers import BackendV2
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

with warnings.catch_warnings():
    # Importing quimb raises an annoying warning about different hyper-optimizers
    warnings.filterwarnings("ignore", category=UserWarning)
    import quimb as qu

from iqm.qaoa.circuits import qiskit_circuit, quimb_tn, transpiled_circuit
from iqm.qaoa.transforming_functions import ham_graph_to_ham_operator
from iqm.qiskit_iqm.iqm_provider import IQMProvider

if TYPE_CHECKING:
    from iqm.qaoa.qubo_qaoa import QUBOQAOA


class EstimatorBackend(ABC):
    """The :class:`~abc.ABC` for estimator backends, i.e., those calculating the expected value of the Hamiltonian."""

    # Temporarily restricted to QUBOQAOA, even though it should theoretically accept QAOA too, to avoid mypy problems.
    @abstractmethod
    def estimate(self, qaoa_object: QUBOQAOA) -> float:
        """The abstract method for :meth:`estimate` of backends subclassed from :class:`EstimatorBackend`.

        The input ``qaoa_object`` includes the training parameters (:attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`), which
        are typically used in estimation of the energy.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose energy is to be estimated.

        Returns:
            The estimated expected value of the Hamiltonian with the quantum state implied by the QAOA object.

        """

    @abstractmethod
    def estimate_correlations_z(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The abstract method for estimating the exp. value of products of Z operators on ``target_qubits``.

        The input ``qaoa_object`` includes the training parameters (:attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`), which
        are used in estimation of the correlations. Some estimators (subclasses of :class:`EstimatorBackend`) may only
        be able to estimate the expectation values of at most quadratic products of Z's.


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

        """
        raise NotImplementedError


class SamplerBackend(ABC):
    """The :class:`~abc.ABC` for sampler backends, i.e., those returning samples from the QAOA."""

    # Temporarily restricted to QUBOQAOA, even though it should theoretically accept QAOA too, to avoid mypy problems.
    @abstractmethod
    def sample(self, qaoa_object: QUBOQAOA, shots: int) -> dict[str, int]:
        """The abstract method for :meth:`sample` of backends subclassed from :class:`SamplerBackend`.

        Args:
            qaoa_object: A :class:`~iqm.qaoa.generic_qaoa.QAOA` object to be sampled from.
            shots: The number of individual samples to take.

        Returns:
            A dictionary of samples. The keys are bitstrings and the values are their counts (which should add up to
            ``shots``)

        """


def _validate_and_normalize_target_qubits(target_qubits: set[LogQubit] | list[set[LogQubit]]) -> list[set[LogQubit]]:
    """Validates and normalizes the variable ``target_qubits``, an input to :meth:`estimate_correlations_z`.

    Does the following two steps:
    1. Checks that ``target_qubits`` is the correct type. That is, either a set of
       :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit` (an alias for integer) or a list of sets of
       :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`.
    2. In case that ``target_qubits`` is a list of sets of :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`,
       return it. If it is just a set of :class:`~iqm.qaoa.transpiler.quantum_hardware.LogQubit`, returns a
       single-element list containing ``target_qubits``so that the output of this function is always
       ``list[set[LogQubit]]``.

    Args:
        target_qubits: The variable to be validated and normalized (representing the qubits whose Z-correlations we're
            interested in)

    Returns:
        Normalized ``target_qubits``

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
        raise TypeError(f"Invalid type for target_qubits: {target_qubits!r}. Expected set[int] or list[set[int]].")


def _operator_z_terms(qubits: set[LogQubit], num_qubits: int) -> SparsePauliOp:
    """Create a :class:`~qiskit.quantum_info.SparsePauliOp` with Z operators on the specified qubits.

    Args:
        qubits: Set of qubit indices where Z should be applied.
        num_qubits: Total number of qubits in the system.

    Returns:
        The Pauli operator with Z on the specified qubits and I (identity) elsewhere, with coefficient 1.0.

    """
    # Build the Pauli string
    pauli_str_list = ["I"] * num_qubits
    for q in qubits:
        if q < 0 or q >= num_qubits:
            raise ValueError(f"Qubit index {q} out of bounds for {num_qubits} qubits.")
        pauli_str_list[q] = "Z"

    pauli_str = "".join(pauli_str_list)
    pauli_list = PauliList([pauli_str])
    coeffs = [1.0]

    return SparsePauliOp(pauli_list, coeffs)


class EstimatorSingleLayer(EstimatorBackend):
    """The estimator class for calculating the expectation value analytically (for :math:`p=1` QAOA)."""

    def estimate(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian for :math:`p=1` QAOA.

        The function calculates the energy (exp. val. of the Hamiltonian) by adding the expectation values
        of its individual terms expressed through equation (12) in :cite:`Ozaeta_2020`. The calculation includes a
        constant term (coming from the translation of a QUBO problem to a Hamiltonian).

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        Raises:
            ValueError: If the provided :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA` object has more than 1 layer.
            TypeError: If the variables in the ``qaoa_object.hamiltonian_bqm`` are not labelled by integers. Using
                :class:`QUBOQAOA` correctly should automatically lead to its attribute
                :attr:`~iqm.qaoa.qubo_qaoa.QUBOQAOA.hamiltonian_bqm` having its variables labelled by integers.

        """
        if qaoa_object.num_layers != 1:
            raise ValueError(f"The number of layers is not 1, but {qaoa_object.num_layers}")

        g, b = qaoa_object.angles  # QAOA angles gamma and beta.
        h_bqm = qaoa_object.hamiltonian_bqm

        energy = 0.0  # To be incremented by the exp. val. of the individual terms in the two following for loops.

        # Linear terms.
        for qb in h_bqm.variables:
            if not isinstance(qb, int):  # If used as intended, this should never happen.
                raise TypeError("The variables in the ``qaoa_object.hamiltonian_bqm`` have to be integers.")
            # The expectation value of :math:`\langle Z \rangle` is offloaded into a helper function.
            energy += self._expval_z(qb, g, b, h_bqm) * h_bqm.get_linear(qb)

        # Quadratic terms.
        for i, j in h_bqm.quadratic:
            if not (isinstance(i, int) and isinstance(j, int)):  # If used as intended, this should never happen.
                raise TypeError("The variables in the ``qaoa_object.hamiltonian_bqm`` have to be integers.")
            # The expectation value of :math:`\langle ZZ \rangle` is offloaded into a helper function.
            energy += self._expval_zz(i, j, g, b, h_bqm) * h_bqm.get_quadratic(i, j)

        # Constant offset
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

    def estimate_correlations_z(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        This works only if the set(s) in ``target_qubits`` are of size at most 2. In case of a set of two qubits, it
        adds an interaction of strength 0 between them, so that they are neighboring in the BQM.

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
    """The estimator class for calculating the expectation value using statevector simulation."""

    def estimate(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian from running state-vector simulation in :mod:`qiskit`.

        Builds a :class:`~qiskit.circuit.QuantumCircuit` for the QAOA and runs the statevector simulation of
        the circuit, calculating the expectation value of the energy from the statevector. The calculation includes
        a constant term (coming from the translation of a QUBO problem to a Hamiltonian).

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.generic_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.

        Returns:
            The expectation value of the energy of the QAOA state using :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles`.

        """
        qc = qiskit_circuit(qaoa_object, measurements=False)
        statevector = Statevector.from_instruction(qc)
        statevector = statevector.reverse_qargs()
        observable = ham_graph_to_ham_operator(qaoa_object.hamiltonian_graph)
        expectation_value = statevector.expectation_value(observable) + qaoa_object.hamiltonian_bqm.offset
        return expectation_value.real

    def estimate_correlations_z(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        Using statevector simulator, calculating any expectation value exactly is relatively straightforward.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose correlations are to be estimated.
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
            observable = _operator_z_terms(qubit_set, qaoa_object.num_qubits)
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
        ValueError: If ``cvar`` is provided, but it's not between 0 and 1.

    """

    def __init__(self, sampler: SamplerBackend, shots: int, cvar: float | None = None) -> None:
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

    def estimate(self, qaoa_object: QUBOQAOA, **kwargs: Any) -> float:
        """Calculates the expectation value of the Hamiltonian by sampling from the QAOA circuit.

        Uses the sampler provided at initialization to sample from the QAOA circuit and then calculates the expectation
        value from the counts.

        Args:
            qaoa_object: The instance of :class:`~iqm.qaoa.generic_qaoa.QUBOQAOA` whose expectation value is to be
                calculated.
            **kwargs: Keyword arguments passed through to the :meth:`~iqm.qaoa.backends.SamplerBackend.sample`. In
                practice, this is often just ``seed_transpiler`` for the samplers which allow input seed to derandomize
                the circuit transpilation.

        Returns:
            The average energy of the sampled docstrings (to serve as estimation of the expectation value).

        """
        counts = self.sampler.sample(qaoa_object, self.shots, **kwargs)
        return qaoa_object.problem.cvar(counts, self.cvar)

    def estimate_correlations_z(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]], **kwargs: Any
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
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose correlations are to be estimated.
            target_qubits: The set of qubits on which the operators act, or a list thereof.
            **kwargs: Keyword arguments passed through to the :meth:`~iqm.qaoa.backends.SamplerBackend.sample`. In
                practice, this is often just ``seed_transpiler`` for the samplers which allow input seed to derandomize
                the circuit transpilation.

        Returns:
            The estimated expected value of product of Z operators on given ``target_qubits``. Or a list of those, if
            ``target_qubits`` was given as a list.

        """  # noqa: D416  # Silence warnings from building docs.
        # Validate input and normalize it so that it's always a list of sets of qubits (possibly a short list).
        target_qubits = _validate_and_normalize_target_qubits(target_qubits)
        counts = self.sampler.sample(qaoa_object, self.shots, **kwargs)

        # The variable to be returned.
        list_of_correlations: list[float] = []
        for qubit_set in target_qubits:
            cum_sum: float = 0
            number_of_measurements = 0

            for bin_str, counter in counts.items():
                # Contribution of one bitstring (multiplied by the respective count).
                cum_sum += prod(1 if bin_str[qb] == "0" else -1 for qb in qubit_set) * counter
                number_of_measurements += counter

            if number_of_measurements == 0:
                raise ValueError("There are no counts. The expected value can't be averaged.")

            list_of_correlations.append(cum_sum / number_of_measurements)

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations


class EstimatorQUIMB(EstimatorBackend):
    """The estimator class for calculating the expectation value using the tensor network package :mod:`quimb`."""

    CRIT_DEG = 3  # The maximum degree for which QUIMB runs somewhat tolerably fast.

    def estimate(self, qaoa_object: QUBOQAOA) -> float:
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
            energy += tn.local_expectation(to_measure, (q1, q2)) * qaoa_object.hamiltonian_bqm.get_quadratic(q1, q2)
        for q1 in qaoa_object.hamiltonian_bqm.variables:
            to_measure = qu.pauli("Z")
            energy += tn.local_expectation(to_measure, (q1)) * qaoa_object.hamiltonian_bqm.get_linear(q1)

        # The energy should already be real.
        return energy.real + qaoa_object.hamiltonian_bqm.offset

    def estimate_correlations_z(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float | list[float]:
        r"""The method for estimating the exp. value of products of Z operators on ``target_qubits``.

        The correlations are calculated natively for QUIMB, as a contraction of tensor networks, very similarly to how
        the expectation value of the Hamiltonian is estimated in :meth:`estimate`.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QAOA` object whose correlations are to be estimated.
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
            # Construct ``qu.pauli("Z") & qu.pauli("Z") & ... & qu.pauli("Z")`` correct number of times.
            to_measure = reduce(operator.and_, (qu.pauli("Z") for _ in range(len(qubit_set))))
            correlation = tn.local_expectation(to_measure, qubit_set)

            list_of_correlations.append(correlation.real)  # The correlation should already be real.

        # If there's just one correlation, don't return the list, just return the correlation.
        if len(list_of_correlations) == 1:
            return list_of_correlations[0]
        else:
            return list_of_correlations


class SamplerRandomBitstrings(SamplerBackend):
    """A sampler that ignores the QAOA and just produces random bitstrings of the correct length."""

    def sample(self, qaoa_object: QUBOQAOA, shots: int) -> dict[str, int]:
        """Produce random bitstrings to act as samples from the QAOA.

        The ``qaoa_object`` is used only to get the number of qubits (which corresponds to the length of
        the bitstrings). The number of uniformly random bitstrings produced is ``shots`` and they are arranged in
        a dictionary just like counts from a :mod:`qiskit` measurement.

        Args:
            qaoa_object: The QAOA object, only used to get the number of qubits.
            shots: The number of random strings to generate.

        Returns:
            A dictionary whose keys are the produced random bitstrings and values their frequencies in the random set.

        """
        counts: dict[str, int] = {}
        for _ in range(shots):
            bitstring = "".join(random.choice(["0", "1"]) for _ in range(qaoa_object.num_qubits))
            if bitstring in counts:
                counts[bitstring] += 1
            else:
                counts[bitstring] = 1
        return counts


class SamplerSimulation(SamplerBackend):
    """A sampler that simulates the QAOA circuit in :mod:`qiskit`.

    Some simulators may need the circuit to be transpiled, so optionally a string describing the transpiler can be
    provided.

    Args:
        simulator: A simulator, (currently) assumed to be an object of class :class:`~qiskit_aer.AerSimulator`.
        transpiler: A transpilation (routing) strategy to use, if applicable.

    """

    # The type hint suggests that `simulator` can be any `BackendV2`, but it should be a simulator, not a real QC.
    # `BackendV2` is the nearest common ancestor of `AerSimulator` and `IQMFakeBackend`, which are the two main
    # backends that we might want use here, so it's used as a type hint.
    def __init__(
        self,
        simulator: BackendV2 | None = None,
        transpiler: TranspilerOption | None = None,
    ) -> None:
        if simulator is None:
            simulator = AerSimulator(method="statevector")
        self.simulator = simulator
        self.transpiler = transpiler

    def sample(self, qaoa_object: QUBOQAOA, shots: int, **kwargs: Any) -> dict[str, int]:
        """Samples from the QAOA using a simulation.

        The dictionary of counts is obtained from `qiskit` and then the bitstrings are **reversed**, so they don't use
        the `qiskit` convention of the first bit being on the right of the bitstring.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QUBOQAOA` object, to be sampled from.
            shots: The number of samples (measurements) to take.
            **kwargs: Extra keyword arguments for constructing the transpiled circuit before the simulation. Mostly
                intended for ``seed_transpiler``, a random seed for the transpilation.

        Returns:
            A dictionary whose keys are the measured bitstrings and values their frequencies in the results.

        """
        qc = transpiled_circuit(qaoa_object, backend=self.simulator, transpiler=self.transpiler, **kwargs)
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

    """

    def __init__(
        self,
        token: str,
        server_url: str = "https://resonance.iqm.tech/garnet",
        transpiler: TranspilerOption = TranspilerOption.SPARSE,
    ) -> None:
        self.iqm_backend = IQMProvider(server_url, token=token).get_backend()
        self.token = token
        self.transpiler = transpiler

    def sample(self, qaoa_object: QUBOQAOA, shots: int, **kwargs: Any) -> dict[str, int]:
        """Samples from the QAOA on a quantum computer via IQM Resonance.

        First, it creates a :class:`~qiskit.circuit.QuantumCircuit` (using a custom transpilation approach) and then
        sends it to IQM Resonance. The dictionary of counts is obtained from `qiskit` and then the bitstrings are
        **reversed**, so they don't use the `qiskit` convention of the first bit being on the right of the bitstring.

        Args:
            qaoa_object: The :class:`~iqm.qaoa.generic_qaoa.QUBOQAOA` object, to be sampled from.
            shots: The number of samples (measurements) to take.
            **kwargs: Extra keyword arguments for constructing the transpiled circuit before sending it to Resonance.
                Mainly intended for ``seed_transpiler``, a random seed for the transpilation.

        Returns:
            A dictionary whose keys are the measured bitstrings and values their frequencies in the results.

        """
        qc = transpiled_circuit(qaoa_object, backend=self.iqm_backend, transpiler=self.transpiler, **kwargs)
        job = self.iqm_backend.run(qc, shots=shots)
        counts_from_job = job.result().get_counts()
        # Qiskit somehow reverses the order of the bitstrings.
        counts_correctly_ordered = {key[::-1]: value for key, value in counts_from_job.items()}
        return counts_correctly_ordered
