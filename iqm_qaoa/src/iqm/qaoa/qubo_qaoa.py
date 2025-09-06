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
"""A module for the QUBOQAOA class.

The class :class:`QUBOQAOA` mostly serves to store the parameters (angles) of the QAOA circuit and to facilitate various
operations on it. The training of the QAOA circuit is done here. For constructing circuits representing the QAOA, use
functions from the module :mod:`~iqm.qaoa.circuits`. For sampling from the QAOA and calculating expectation values, use
the backend classes from the module :mod:`~iqm.qaoa.backends`.
"""

from collections.abc import Sequence
from typing import Any, Literal

from dimod import BinaryQuadraticModel, to_networkx_graph
from iqm.applications.qubo import ConstrainedQuadraticInstance, QUBOInstance
from iqm.qaoa.backends import EstimatorBackend, EstimatorSingleLayer, EstimatorStateVector, SamplerBackend
from iqm.qaoa.generic_qaoa import QAOA
import networkx as nx
import numpy as np
from scipy.optimize import minimize


class QUBOQAOA(QAOA):
    """The class for QAOA with quadratic unconstrained binary (QUBO) cost function.

    The class inherits a lot of functionality from its parent :class:`iqm.qaoa.generic_qaoa.QAOA`. One new addition is
    the attribute :attr:`bqm` which stores the coefficient of the problem Hamiltonian. The same data in the form of
    :class:`~networkx.Graph` is :attr:`hamiltonian_graph`.

    Args:
        problem: A :class:`~iqm.applications.qubo.QUBOInstance` object describing the QUBO problem to be solved.
        num_layers: The number of QAOA layers, commonly referred to as *p* in the literature.
        betas: An optional list of the initial *beta* angles of QAOA. Has to be provided together with ``gammas``.
        gammas: An optional list of the initial *gamma* angles of QAOA. Has to be provided together with ``betas``.
        initial_angles: An optional list of the initial QAOA angles as one variable. Shouldn't be provided together
            with either ``betas`` or ``gammas``.

    """

    def __init__(
        self,
        problem: QUBOInstance | ConstrainedQuadraticInstance,
        num_layers: int,
        *,
        betas: Sequence[float] | np.ndarray | None = None,
        gammas: Sequence[float] | np.ndarray | None = None,
        initial_angles: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        super().__init__(problem, num_layers, betas=betas, gammas=gammas, initial_angles=initial_angles)
        self._bqm = problem.bqm.spin

    @property
    def bqm(self) -> BinaryQuadraticModel:
        """The BQM representation of the problem, taken from the input :class:`~iqm.applications.qubo.QUBOInstance`."""
        return self._bqm

    @property
    def hamiltonian_graph(self) -> nx.Graph:
        """The graph whose edges / nodes have weights ``bias`` equal to the coefficients in the problem Hamiltonian."""
        return to_networkx_graph(self._bqm)

    @property
    def interactions(self) -> np.ndarray:
        r"""Returns an upper-triangular matrix of the *ZZ* interactions between the variables.

        If the Hamiltonian representing the problem is

        .. math:: H = \sum_{i<j} J_{ij} Z_i Z_j  + \sum_i h_i Z_i

        then this method outputs :math:`J_{ij}` as upper-triangular square matrix :class:`~numpy.ndarray`. Note that
        these are different from the off-diagonal elements of :attr:`~iqm.applications.qubo.QUBOInstance.qubo_matrix` of
        the input ``problem`` because the QUBO cost function has different coefficients than the Hamiltonian.
        """
        _, (row, col, quad), *_ = self._bqm.to_numpy_vectors(sort_indices=True)
        matrix_interactions = np.zeros((self._bqm.num_variables, self._bqm.num_variables))
        matrix_interactions[row, col] = quad
        return matrix_interactions

    @property
    def local_fields(self) -> np.ndarray:
        r"""Returns a :class:`~numpy.ndarray` of the local fields of the model (*Z* coefficients).

        If the Hamiltonian representing the problem is

        .. math:: H = \sum_{i<j} J_{ij} Z_i Z_j  + \sum_i h_i Z_i

        then this method outputs :math:`h_{i}` as 1-dimensional :class:`~numpy.ndarray`. Note that these are different
        from the diagonal elements of :attr:`~iqm.applications.qubo.QUBOInstance.qubo_matrix` of the input ``problem``
        because the QUBO cost function has different coefficients than the Hamiltonian.
        """
        loc_fields, _, *_ = self._bqm.to_numpy_vectors(sort_indices=True)
        return loc_fields

    def train(
        self,
        estimator: EstimatorBackend | None = None,
        min_method: Literal[
            "Nelder-Mead",
            "nelder-mead",
            "Powell",
            "powell",
            "CG",
            "cg",
            "BFGS",
            "bfgs",
            "Newton-CG",
            "newton-cg",
            "L-BFGS-B",
            "l-bfgs-b",
            "TNC",
            "tnc",
            "COBYLA",
            "cobyla",
            "COBYQA",
            "cobyqa",
            "SLSQP",
            "slsqp",
            "Trust-Constr",
            "trust-constr",
            "Dogleg",
            "dogleg",
            "Trust-NCG",
            "trust-ncg",
            "Trust-Exact",
            "trust-exact",
            "Trust-Krylov",
            "trust-krylov",
        ] = "COBYLA",
        **kwargs: Any,
    ) -> None:
        """The function that performs the training of the angles.

        The training modifies :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles` in-place using
        the :func:`~scipy.optimize.minimize` function from :mod:`scipy`. The training uses the provided ``estimator``.

        Args:
            estimator: An estimator :class:`~iqm.qaoa.backends.EstimatorBackend` to be used to calculating expectation
                values for the minimization.
            min_method: The minimization method passed to the :func:`~scipy.optimize.minimize` function.
            **kwargs: The keyword arguments to pass to the ``estimator``'s
                :meth:`~iqm.qaoa.backends.EstimatorBackend.estimate`.

        """
        if estimator is None:
            if self._num_layers == 1:
                estimator = EstimatorSingleLayer()
            else:
                estimator = EstimatorStateVector()

        def function_to_minimize(local_angles: np.ndarray) -> float:
            """Auxiliary function to be used in minimization.

            Takes an input, sets :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles` to it and then calls the energy function.
            The :func:`~scipy.optimize.minimize` function from :mod:`scipy` needs a callable with one input. The energy
            functions don't have input. Instead they use :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles` (and possibly other
            attributes of ``self``) which they have access to. To allow us to use :func:`~scipy.optimize.minimize`, we
            need to define this function.

            Args:
                local_angles: A :class:`~numpy.ndarray` of the angles to try.

            Returns:
                The energy from ``estimator`` using the input angles.

            """
            self._angles = local_angles
            return estimator.estimate(self, **kwargs)

        solution = minimize(function_to_minimize, x0=self.angles, method=min_method)
        self._angles = solution.x
        self._trained = True

    # This method is temporarily moved here from QAOA since SamplerBackend was temporarily restricted to only accept
    # QUBOQAOA
    def sample(self, sampler: SamplerBackend, shots: int = 20000, **kwargs: Any) -> dict[str, int]:
        """The method for taking samples (i.e., measurement results) from the QAOA circuit.

        Takes a :class:`~iqm.qaoa.backends.SamplerBackend` and uses it to get ``shots`` samples. The backend is
        responsible for building the quantum circuit and taking the measurements (or obtaining the samples some other
        way), using information from the :class:`QAOA` object that is passed to its method
        :meth:`~iqm.qaoa.backends.SamplerBackend.sample`.

        Args:
            sampler: The sampler to use to generate samples. The sampler is an instance of a subclass of
                :class:`~iqm.qaoa.backends.SamplerBackend` with a :meth:`~iqm.qaoa.backends.SamplerBackend.sample`
                method of the appropriate signature.
            shots: The number of shots to be taken.
            **kwargs: Extra arguments to pass to the sampler. Mostly intended for ``seed_transpiler`` for samplers that
                include :mod:`qiskit` transpilation.

        Returns:
            A dictionary whose keys are bitstrings representing the samples and whose values are their respective
            frequencies, so that the sum of the values of the dictionary equals to ``shots``.

        """
        return sampler.sample(self, shots, **kwargs)

    # This method is temporarily moved here from QAOA since EstimatorBackend was temporarily restricted to only accept
    # QUBOQAOA
    def estimate(self, estimator: EstimatorBackend, **kwargs: Any) -> float:
        """The method for taking estimates of the expected value of the Hamiltonian from the QAOA circuit.

        Takes a :class:`~iqm.qaoa.backends.EstimatorBackend` and uses it to get estimates of the expected value.
        The backend takes all the necessary information from the :class:`QAOA` object that is passed to its method
        :meth:`~iqm.qaoa.backends.EstimatorBackend.estimate`.

        Args:
            estimator: The estimator used to get the expected value. The estimator is an instance of a subclass of
                :class:`~iqm.qaoa.backends.EstimatorBackend` with a method
                :meth:`~iqm.qaoa.backends.EstimatorBackend.estimate` of the appropriate signature.
            **kwargs: Optional keyword arguments to be passed to the ``estimator``. Mostly relevant for when
                the estimators is :class:`~iqm.qaoa.backends.EstimatorFromSampler`, the sampler is
                :class:`~iqm.qaoa.backends.SamplerResonance` and we want to specify parameters of the transpilation on
                the Resonance QPU.

        Returns:
            An estimate of the expectation value fo the Hamiltonian. Not normalized in any way.

        """
        return estimator.estimate(self, **kwargs)
