# Copyright (c) 2024-2026 IQM Quantum Computers
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
"""Contains the problem instance class for HUBO problems."""

from collections.abc import Hashable

from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.vartypes import VartypeLike
from iqm.applications.applications import ProblemInstance


class HUBOInstance(ProblemInstance):
    r"""A problem instance class for generic HUBO problems.

    Internally, the HUBO instance is stored as a :class:`~dimod.higherorder.BinaryPolynomial` object. This object stores
    the problem as a collection of terms and their coefficients. Each term is a :class:`frozenset` of variables whose
    product makes up the term in the polynomial. For example, the polynomial:

    .. math::

        1.3 x y - 3.7 x

    is represented as:

    .. code-block:: python

        {frozenset({'x', 'y'}): 1.3, frozenset({'x'}): -3.7}

    Args:
        bp: The input data for creating the instance. It can be one of the following:

            - A :class:`~dimod.higherorder.BinaryPolynomial` object.
            - A dictionary mapping tuples/frozensets of variable names to coefficients. For the dictionary:

              - Keys must always be tuples or frozensets, even for single-variable terms. For example, ``("var",): 0.5``
                represents a single-variable term, while ``"var": 0.5`` would be incorrectly interpreted as a cubic term
                with variables ``"v"``, ``"a"``, ``"r"``.
              - Repeated variables in a tuple are treated based on ``vartype`` (consistent with the math:
                :math:`x^2 = x` for :math:`x \in \{0, 1\}` and :math:`x^2 = 1` for :math:`x \in \{-1, 1\}`).
              - Values are the coefficients of the corresponding terms.

        vartype: Optional variable type for interpreting the dictionary input. Defaults to ``'BINARY'``.

    Raises:
        TypeError: If the variable labels of the input are not sortable (e.g., mixing integers and strings).
        TypeError: If the input ``bp`` is a dictionary and the keys aren't tuples or frozensets.

    """

    def __init__(
        self,
        bp: BinaryPolynomial | dict[tuple[Hashable, ...] | frozenset[Hashable], float],
        vartype: VartypeLike = "BINARY",
    ) -> None:
        if isinstance(bp, BinaryPolynomial):
            intermediate_bp = bp
        else:
            bad_keys = [term for term in bp if not isinstance(term, (tuple, frozenset))]
            if bad_keys:
                raise TypeError(
                    "All keys in `bp` must be tuples or frozensets. Each key represents a term in the binary "
                    "polynomial, containing the variables involved in that term. "
                    f"Invalid keys found: {bad_keys!r}"
                )
            intermediate_bp = BinaryPolynomial(bp, vartype=vartype)

        # One of the following two is identity.
        # The private attribute ``_spin_bp`` is saved in order to avoid converting the BP as spin -> binary -> spin.
        # This conversion back and forth introduces zero terms which might mess up Qiskit circuit construction.
        self.bp = intermediate_bp.to_binary(copy=True)
        self._spin_bp = intermediate_bp.to_spin(copy=True)

        try:
            self.sorted_vars = sorted(self.bp.variables)
        except TypeError as e:
            types = {type(v).__name__ for v in self.bp.variables}
            raise TypeError(
                "Failed to sort the variable labels. All variable labels must be comparable with each other."
                f"Encountered types: {types}. "
                f"Labels: {self.bp.variables!r}"
            ) from e
        super().__init__()

    @property
    def dim(self) -> int:
        """The dimension of the problem, i.e., the number of variables in the polynomial."""
        return len(self.bp.variables)

    @property
    def average_quality(self) -> float:
        """The average quality value over all possible bitstrings.

        For HUBO problems, this is equal to the constant term in the spin/Hamiltonian formulation of the cost function.
        """
        return self._spin_bp[frozenset()]

    def quality(self, bit_str: str) -> float:
        """The 'quality' of the input bitstring.

        Calculates the value of the polynomial when the bit values from the bitstring are plugged in for the variables.
        The values in the bitstring correspond to the variables at their corresponding location in ``self.sorted_vars``.

        Args:
            bit_str: The bitstring representing the values to be plugged in for the variables.

        Raises:
            ValueError: If the length of the input bitstring does not match the number of variables of the problem.

        """
        if len(bit_str) != self.dim:
            raise ValueError(
                f"The length of the provided bitstring ({len(bit_str)}) is incorrect. "
                f"The bitstring length has to be equal to the number of problem variables: {self.dim}"
            )

        bit_str_as_sample = {var: int(bit_str[i]) for i, var in enumerate(self.sorted_vars)}
        return self.bp.energy(bit_str_as_sample)
