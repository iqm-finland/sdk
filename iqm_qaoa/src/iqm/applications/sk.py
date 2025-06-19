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
"""Module containing the SK model problem instance class.

Example:

    .. code-block:: python

        from iqm.applications.sk import sk_iterator

        for my_instance in sk_generator(n, n_instances):  # Generates problem instances.
            my_instance.quality("111111111")

"""

from collections.abc import Iterator

from dimod import BinaryQuadraticModel
from iqm.applications.qubo import QUBOInstance
import numpy as np


class SherringtonKirkpatrick(QUBOInstance):
    """The problem class for Sherrington Kirkpatrick model.

    In this model, all qubits interact with randomly distributed interactions. The model takes an interaction matrix on
    input. The suggested usage is to generate SK problem instances using :func:`~iqm.applications.sk.sk_generator`.

    Args:
        interaction_matrix: The matrix of interactions between the spins in the SK model.

    """

    def __init__(self, interaction_matrix: np.ndarray) -> None:
        self._bqm = BinaryQuadraticModel(interaction_matrix, vartype="SPIN").binary
        super().__init__(self._bqm)


def sk_generator(
    n: int, n_instances: int, distribution: str = "gaussian", seed: int = 1337
) -> Iterator[SherringtonKirkpatrick]:
    """The generator function for generating random SK model problem instances.

    The generator yields :class:`SherringtonKirkpatrick` model problem instances using random ``interaction_matrix``,
    created according to the input parameters.

    * 'gaussian' -> Gaussian distribution with mean 0 and standard deviation 1.
    * 'rademacher' -> Value +1 with probability 0.5 and value -1 with probability 0.5.
    * 'uniform' -> Uniform distribution between 0 and 1.

    Args:
        n: The number of qubits in the problem instance, also the ``intraction_matrix`` dimensions.
        n_instances: The number of SK model instances to generate.
        distribution: A string describing the distribution of the elements in the ``interaction_matrix``.
            Possible distributions include 'gaussian' (also known as 'normal'), 'rademacher' and 'uniform'.
        seed: Optional random seed for generating the problem instances.

    Returns:
        An iterator of :class:`SherringtonKirkpatrick` objects, corresponding to randomly-generated instances of
        the model.

    """
    np.random.seed(seed)
    for _ in range(n_instances):
        if distribution in ("gaussian", "normal"):
            interaction_matrix = np.triu(np.random.normal(0, 1, size=(n, n)), k=1) / np.sqrt(n)
        elif distribution == "rademacher":
            matrix = 2 * np.random.binomial(n=1, p=0.5, size=(n, n)) - np.ones(shape=(n, n))
            interaction_matrix = np.triu(matrix, k=1) / np.sqrt(n)
        elif distribution == "uniform":
            interaction_matrix = np.triu(np.random.uniform(0, 1, size=(n, n)), k=1) / np.sqrt(n)
        else:
            raise ValueError("Invalid distribution. Choose either 'gaussian', 'rademacher' or 'uniform'.")
        yield SherringtonKirkpatrick(interaction_matrix)
