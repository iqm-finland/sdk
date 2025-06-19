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
"""A module containing the generic QAOA abstract base class."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy

from iqm.applications.applications import ProblemInstance
from iqm.qaoa.backends import EstimatorBackend, SamplerBackend
import numpy as np


class QAOA(ABC):
    """The most generic QAOA abstract base class.

    This abstract base class contains methods such as :meth:`_internal_angle_logic` or :meth:`linear_ramp_schedule`
    that can be used by any type / flavor of QAOA.

    Args:
        problem: The :class:`~iqm.applications.applications.ProblemInstance` to be solved by the QAOA.
        num_layers: The number of QAOA layers.
        betas: An optional list of the initial *beta* angles of QAOA. Has to be provided together with ``gammas``.
        gammas: An optional list of the initial *gamma* angles of QAOA. Has to be provided together with ``betas``.
        initial_angles: An optional list of the initial QAOA angles as one variable. Shouldn't be provided together
            with either ``betas`` or ``gammas``.

    """

    def __init__(
        self,
        problem: ProblemInstance,
        num_layers: int,
        *,
        betas: Sequence[float] | np.ndarray | None = None,
        gammas: Sequence[float] | np.ndarray | None = None,
        initial_angles: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if not isinstance(num_layers, int) or isinstance(num_layers, bool):
            raise TypeError(f"The number of QAOA layers must be an integer, got {type(num_layers).__name__}.")
        if num_layers <= 0:
            raise ValueError(f"The number of QAOA layers must be positive, not {num_layers}.")
        self._num_layers = num_layers

        self._angles = self._internal_angle_logic(betas, gammas, initial_angles)
        self._trained = False

        # A deep copy of the input ``problem``, to make sure that it's safe against accidental changes to ``problem``.
        self._problem = deepcopy(problem)
        self._num_qubits = self._problem.dim

    def _internal_angle_logic(
        self,
        betas: Sequence[float] | np.ndarray | None = None,
        gammas: Sequence[float] | np.ndarray | None = None,
        initial_angles: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Internal method to guarantee that angles are assigned correctly.

        This method takes the inputs and uses them to produce an output array of angles, using the following rules
        (raising the appropriate error if the rules are broken):

        - EITHER ``initial_angles`` should be provided OR ``betas`` and ``gammas``, not both.
        - If provided, ``betas`` and ``gammas`` should have their lengths equal to the number of layers
            :attr:`_num_layers`. They are merged like a zip into one set of angles to be returned.
        - If provided, the length of ``initial_angles`` should be twice the number of layers :attr:`_num_layers`. It
            is returned as a :class:`~np.ndarray`.
        - If neither of the input parameters is provided, the returned set of angles is all zeros (of the appropriate
            length).

        Args:
            betas: The beta angles of a QAOA circuit.
            gammas: The gamma angles of a QAOA circuit.
            initial_angles: The angles of a QAOA circuit in one list / array.

        Returns:
            A numpy array of angles, determined from the logic above.

        Returns:
            ValueError: If the length of the provided angles is not twice the number of QAOA layers.
            ValueError: If incorrect combination of input parameters is provided (see docstring above).

        """
        # If neither ``initial_angles``, ``betas`` nor ``gammas`` is provided, initialize the angles as all zeros.
        if betas is None and gammas is None and initial_angles is None:
            angles = np.zeros(self._num_layers * 2, dtype=float)
        # If only ``initial_angles`` is provided, use it to initialize the angles:
        elif betas is None and gammas is None and initial_angles is not None:
            if len(initial_angles) != 2 * self._num_layers:
                raise ValueError(
                    f"The length of provided angles ({len(initial_angles)}) doesn't equal twice"
                    f" the number of QAOA layers ({self._num_layers})."
                )
            angles = np.array(initial_angles).flatten()  # Flattened for if for some reason it's not flat.
        # If both ``betas`` and ``gammas`` are provided (not ``initial_angles``), use them to initialize the angles.
        elif betas is not None and gammas is not None and initial_angles is None:
            if len(gammas) != len(betas):
                raise ValueError(
                    f"The lengths of provided ``betas`` ({len(betas)}) and ``gammas`` ({len(gammas)}) aren't equal."
                )
            if len(gammas) != self._num_layers:
                raise ValueError(
                    f"The length of the provided ``gammas`` and ``betas`` ({len(gammas)}) needs to equal"
                    f" the number of the QAOA layers ({self._num_layers})."
                )
            angles = np.empty(2 * len(betas), dtype=float)
            # Angles are of the form [gamma1, beta1, gamma2, beta2, gamma3, ...]
            angles[0::2] = gammas
            angles[1::2] = betas
        # Otherwise, there was a user input error (one of three possibilities)
        elif betas is not None and gammas is None:
            raise ValueError("The beta angles were provided, but the gamma angles were not provided.")
        elif betas is None and gammas is not None:
            raise ValueError("The gamma angles were provided, but the beta angles were not provided.")
        else:
            raise ValueError(
                "The ``initial_angles`` were provided together with ``gammas`` and ``betas``. "
                "Please provide EITHER ``initial_angles`` OR ``betas`` and ``gammas``."
            )
        return angles

    @property
    def trained(self) -> bool:
        """A boolean flag indicating whether the QAOA has been trained at all or not."""
        return self._trained

    @property
    def num_layers(self) -> int:
        """The number of QAOA layers.

        At first this is set to the value given at initialization, but it may be modified later (which has an effect on
        :attr:`angles`).
        """
        return self._num_layers

    @num_layers.setter
    def num_layers(self, new_num_layers: int) -> None:
        """The setter for :attr:`_num_layers`.

        The internal attribute :attr:`angles` is adjusted accordingly:

        - If the new number of layers is larger, the angles are padded with zeros.
        - If the new number of layers is smaller, the angles are truncated.
        - If the new number of layers is the same as the old number of layers, nothing happens.

        The internal attribute :self:`_trained` is set to ``False``, unless the number of layers remains the same.
        """
        if not isinstance(new_num_layers, int) or isinstance(new_num_layers, bool):
            raise TypeError(f"The number of QAOA layers must be an integer, got {type(new_num_layers).__name__}.")
        if new_num_layers <= 0:
            raise ValueError(f"The number of QAOA layers must be positive, not {new_num_layers}.")

        if new_num_layers < self._num_layers:
            self._angles = np.resize(self._angles, 2 * new_num_layers)
            self._trained = False
        elif new_num_layers > self.num_layers:
            self._angles = np.pad(
                self._angles, (0, 2 * (new_num_layers - self.num_layers)), mode="constant", constant_values=0
            )
            self._trained = False

        self._num_layers = new_num_layers

    @property
    def problem(self) -> ProblemInstance:
        """The problem instance associated with the QAOA."""
        return self._problem

    @property
    def num_qubits(self) -> int:
        """The number of qubits, equal to the number of problem variables if no special encoding is used."""
        return self._num_qubits

    @property
    def betas(self) -> np.ndarray:
        """The beta angles in the QAOA, controlling the mixer Hamiltonian terms."""
        return self._angles[1::2]

    @betas.setter
    def betas(self, new_betas: Sequence[float] | np.ndarray) -> None:
        """Setter for the :meth:`betas`."""
        self._angles = self._internal_angle_logic(new_betas, self.gammas, None)
        self._trained = False

    @property
    def gammas(self) -> np.ndarray:
        """The gamma angles in the QAOA, controlling the problem Hamiltonian terms."""
        return self._angles[::2]

    @gammas.setter
    def gammas(self, new_gammas: Sequence[float] | np.ndarray) -> None:
        """Setter for the :meth:`gammas`."""
        self._angles = self._internal_angle_logic(self.betas, new_gammas, None)
        self._trained = False

    @property
    def angles(self) -> np.ndarray:
        """The angles in the QAOA, including :attr:`betas` and :attr:`gammas` in one :class:`~numpy.ndarray`."""
        return self._angles

    @angles.setter
    def angles(self, new_angles: Sequence[float] | np.ndarray) -> None:
        """Setter for the :meth:`gammas`."""
        self._angles = self._internal_angle_logic(None, None, new_angles)
        self._trained = False

    def sample(self, sampler: SamplerBackend, shots: int = 20000) -> dict[str, int]:
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

        Returns:
            A dictionary whose keys are bitstrings representing the samples and whose values are their respective
            frequencies, so that the sum of the values of the dictionary equals to ``shots``.

        """
        return sampler.sample(self, shots)

    def estimate(self, estimator: EstimatorBackend) -> float:
        """The method for taking estimates of the expected value of the Hamiltonian from the QAOA circuit.

        Takes a :class:`~iqm.qaoa.backends.EstimatorBackend` and uses it to get estimates of the expected value.
        The backend takes all the necessary information from the :class:`QAOA` object that is passed to its method
        :meth:`~iqm.qaoa.backends.EstimatorBackend.estimate`.

        Args:
            estimator: The estimator used to get the expected value. The estimator is an instance of a subclass of
                :class:`~iqm.qaoa.backends.EstimatorBackend` with a method
                :meth:`~iqm.qaoa.backends.EstimatorBackend.estimate` of the appropriate signature.

        Returns:
            An estimate of the expectation value fo the Hamiltonian. Not normalized in any way.

        """
        return estimator.estimate(self)

    def linear_ramp_schedule(self, delta_beta: float, delta_gamma: float) -> None:
        """The "linear ramp schedule" for setting the QAOA angles.

        Formulas adapted from :cite:`Montanez-Barrera_2024`. It can be used either instead of training the QAOA or as
        a starting set of angles. The above work uses ``delta_beta`` and ``delta_gamma`` values around 0.5, but the best
        choice for these values depends on the problem Hamiltonian.

        Args:
            delta_beta: The maximum beta angle.
            delta_gamma: The maximum gamma angle.

        """
        for i in range(self.num_layers):
            gamma = (i + 1) / self.num_layers * delta_gamma
            self._angles[i] = gamma
            beta = (1 - i / self.num_layers) * delta_beta
            self._angles[i + 1] = beta

        self._trained = True

    @abstractmethod
    def train(self) -> None:
        """The function that performs the training of the angles."""
