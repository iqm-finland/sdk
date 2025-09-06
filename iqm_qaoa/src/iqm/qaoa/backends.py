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
import random
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
from qiskit.providers import BackendV2
from qiskit.quantum_info import Statevector
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


class EstimatorSingleLayer(EstimatorBackend):
    """The estimator class for calculating the expectation value analytically (for :math:`p=1` QAOA)."""

    def estimate(self, qaoa_object: QUBOQAOA) -> float:
        """Calculates the expectation value of the Hamiltonian for :math:`p=1` QAOA.

        The function calculates the energy (exp. val. of the Hamiltonian) by adding the expectation values
        of its individual terms expressed through equation (12) in :cite:`Ozaeta_2020`.
        The calculation includes a constant term (coming from the translation of a QUBO problem to a Hamiltonian).

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
        energy = 0  # To be incremented by the exp. val. of the individual terms in the two following "for" loops.
        g = qaoa_object.angles[0]  # variable gamma
        b = qaoa_object.angles[1]  # variable beta
        for node in qaoa_object.bqm.variables:
            hi = qaoa_object.bqm.get_linear(node)
            nn = {x[0] for x in qaoa_object.bqm.iter_neighborhood(node)}  # The set of nearest neighbours of "node".
            prod_cos = 1
            for n in nn:
                prod_cos *= np.cos(
                    2 * g * qaoa_object.bqm.get_quadratic(node, n)
                )  # The product in the formula for expval_ci.
            expval_ci = hi * np.sin(2 * b) * np.sin(2 * g * hi) * prod_cos
            energy += expval_ci

        for i, j in qaoa_object.bqm.quadratic:
            hi = qaoa_object.bqm.get_linear(i)
            hj = qaoa_object.bqm.get_linear(j)
            jij = qaoa_object.bqm.get_quadratic(i, j)

            # NN = nearest neighbours
            nn_i = {x[0] for x in qaoa_object.bqm.iter_neighborhood(i)} - {j}  # The NN of i, excluding j
            nn_j = {x[0] for x in qaoa_object.bqm.iter_neighborhood(j)} - {i}  # The NN of j, excluding i
            nn_only_i = nn_i - nn_j - {j}  # The nodes which are NN of i, but not NN of j (or j itself)
            nn_only_j = nn_j - nn_i - {i}  # The nodes which are NN of j, but not NN of i (or i itself)
            nn_both = nn_j - nn_only_j  # The nodes which are NN of both i and j

            prod_nn_i = np.prod(
                [np.cos(2 * g * qaoa_object.bqm.get_quadratic(i, k)) for k in nn_i]
            )  # The first product on the first line of expval_cij formula
            prod_nn_j = np.prod(
                [np.cos(2 * g * qaoa_object.bqm.get_quadratic(j, k)) for k in nn_j]
            )  # The second product on the first line of expval_cij formula

            prod_only_i = np.prod(
                [np.cos(2 * g * qaoa_object.bqm.get_quadratic(i, k)) for k in nn_only_i]
            )  # The first product on the second line of expval_cij formula
            prod_only_j = np.prod(
                [np.cos(2 * g * qaoa_object.bqm.get_quadratic(j, k)) for k in nn_only_j]
            )  # The second product on the second line of expval_cij formula

            prod_both_plus = np.prod(
                [
                    np.cos(2 * g * (qaoa_object.bqm.get_quadratic(i, k) + qaoa_object.bqm.get_quadratic(j, k)))
                    for k in nn_both
                ]
            )  # The first product on the last line of expval_cij formula
            prod_both_minus = np.prod(
                [
                    np.cos(2 * g * (qaoa_object.bqm.get_quadratic(i, k) - qaoa_object.bqm.get_quadratic(j, k)))
                    for k in nn_both
                ]
            )  # The second product on the last line of expval_cij formula

            # The entire first line of the expval_cij formula
            first_part = (
                0.5
                * jij
                * np.sin(4 * b)
                * np.sin(2 * g * jij)
                * (np.cos(2 * g * hi) * prod_nn_i + np.cos(2 * g * hj) * prod_nn_j)
            )
            factor1 = (
                1 / 2 * jij * np.sin(2 * b) ** 2 * prod_only_i * prod_only_j
            )  # The entire second line of the expval_cij formula
            factor2 = (
                np.cos(2 * g * (hi + hj)) * prod_both_plus - np.cos(2 * g * (hi - hj)) * prod_both_minus
            )  # The entire last line of the expval_cij formula
            second_part = factor1 * factor2

            expval_cij = (
                first_part - second_part
            )  # The expval_cij formula is the difference of the 1st line and the product of the 2nd and 3rd line
            energy += expval_cij

        energy += qaoa_object.bqm.offset
        return energy


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
        expectation_value = statevector.expectation_value(observable) + qaoa_object.bqm.offset
        return expectation_value.real


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
            **kwargs: Keyword arguments passed through to the :meth:`~iqm.qaoa.backends.SamplerBackend.sample`, in
                practice, this is just ``seed_transpiler`` for the samplers which allow input seed to derandomize
                the circuit transpilation.

        Returns:
            The average energy of the sampled docstrings (to serve as estimation of the expectation value).

        """
        counts = self.sampler.sample(qaoa_object, self.shots, **kwargs)
        return qaoa_object.problem.cvar(counts, self.cvar)


CRIT_DEG = 3  # The maximum degree for which QUIMB runs somewhat tolerably fast.


class EstimatorQUIMB(EstimatorBackend):
    """The estimator class for calculating the expectation value using the tensor network package :mod:`quimb`."""

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
        if (
            isinstance(degrees_arr := qaoa_object.bqm.degrees(array=True), np.ndarray)
            and np.mean(degrees_arr) > CRIT_DEG
        ):
            warnings.warn(
                f"The average degree is higher than {CRIT_DEG}, the :mod:`quimb`-based estimator might be very slow.",
                stacklevel=2,
            )
        energy = 0
        tn = quimb_tn(qaoa_object)
        for q1, q2 in qaoa_object.bqm.quadratic:
            to_measure = qu.pauli("Z") & qu.pauli("Z")
            energy += tn.local_expectation(to_measure, (q1, q2)) * qaoa_object.bqm.get_quadratic(q1, q2)
        for q1 in qaoa_object.bqm.variables:
            to_measure = qu.pauli("Z")
            energy += tn.local_expectation(to_measure, (q1)) * qaoa_object.bqm.get_linear(q1)
        return energy.real + qaoa_object.bqm.offset


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
        transpiler: A string describing the transpilation method to use (if applicable).

    """

    # The type hint suggests that `simulator` can be any `BackendV2`, but it should be a simulator, not a real QC.
    # `BackendV2` is the nearest common ancestor of `AerSimulator` and `IQMFakeBackend`, which are the two main
    # backends that we might want use here, so it's used as a type hint.
    def __init__(
        self,
        simulator: BackendV2 | None = None,
        transpiler: str | None = None,
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
            "SparseTranspiler"

    """

    def __init__(
        self,
        token: str,
        server_url: str = "https://cocos.resonance.meetiqm.com/garnet",
        transpiler: str = "SparseTranspiler",
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
