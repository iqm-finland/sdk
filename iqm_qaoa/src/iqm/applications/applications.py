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
"""This module contains the abstract base class providing a template for defining problem instances.

Example:

    .. code-block:: python

        from iqm.applications.applications import ProblemInstance

        class MyProblemInstance(ProblemInstance):
            @property
            def dim(self):
                # Implementation (mandatory)

            def quality(self):
                # Implementation (mandatory)

            @property
            def best_quality(self)
                # Implementation if there is a more efficient problem-instance-specific way to calculate this.

"""

from abc import ABC, abstractmethod
from itertools import product

from dimod.typing import Variable
from dimod.variables import Variables


class ProblemInstance(ABC):
    """The abstract base class for defining problem instances.

    Currently, only problems with binary variables are supported. Any possible candidate solution is therefore
    represented by a bitstring (a string of 1's and 0's).

    The abstract methods :meth:`dim` and :meth:`quality` are meant to be overridden by children classes, depending on
    how the dimension of the problem and the quality of the solution are understood.

    The upper/lower bound, average/best quality attributes start as ``None``. The first time one of them is called, it
    calls the method :meth:`initialize_properties`, which calculates all of them by brute-forcing over all bitstrings
    representing solution candidates.
    """

    def __init__(self) -> None:
        self._upper_bound: float | None = None  # The upper bound on the cost function value.
        self._lower_bound: float | None = None  # The lower bound on the cost function value.
        self._average_quality: float | None = None  # The average cost function value over all bitstrings.
        self._best_quality: float | None = None  # The best cost function value, typically equal to the lower bound.

        self._fixed_variables: dict[Variable, int] = {}
        self._original_variables: Variables

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the problem (e.g., the number of nodes in a problem graph)."""

    @abstractmethod
    def quality(self, bit_str: str) -> float:
        """Accepts a bitstring and returns its quality / energy (the smaller the better).

        Args:
            bit_str: The bitstring representing a solution candidate.

        """

    def fix_variables(self, variables: list[Variable] | dict[Variable, int]) -> None:
        """Fixes (assigns) some of the problem variables.

        Args:
            variables: Either a list of variables (which get all fixed to the value 1) or a dictionary with keys equal
                to the variables to fix and whose values are equal to the values to fix them to (either 1 or 0).

        """
        raise NotImplementedError(
            f"The problem instance class {self.__class__.__name__} does not have this method implemented."
        )

    def initialize_properties(self, max_size: int | None = 30) -> None:
        """The initialization method for upper/lower bound of the cost function and its average/best value.

        The quantities are calculated by brute force (scaling exponentially). By default, using this with problem sizes
        larger than ``max_size`` (default 30) will raise ValueError. This can be bypassed by making ``max_size`` larger
        or setting it to ``None``.

        Args:
            max_size: The maximum size of problems for which the properties may be calculated.

        Raises:
            ValueError: If :meth:`initialize_properties` was called on a :class:`ProblemInstance` object with dimension
                larger than ``max_size``.

        """
        if max_size is not None and self.dim > max_size:
            raise ValueError(
                f"The problem dimension {self.dim} exceeds the maximum of {max_size}. For large dimensions (>30),"
                f" the brute force approach of ``initialize_properties`` may be too slow. Change the ``max_size``"
                f" parameter or set it to ``None`` to bypass this error."
            )

        upper_bound: float = self.quality("0" * self.dim)
        lower_bound: float = self.quality("0" * self.dim)
        average_quality: float = 0

        for bitstr in product("01", repeat=self.dim):
            qlty = self.quality("".join(bitstr))
            upper_bound = max(upper_bound, qlty)
            lower_bound = min(lower_bound, qlty)
            average_quality += qlty

        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._average_quality = average_quality / 2**self.dim
        self._best_quality = lower_bound

    @property
    def upper_bound(self) -> float:
        """The highest quality value among all possible bitstrings.

        Can be calculated together with other bounds using the brute-force :meth:`initialize_properties`. Shouldn't be
        modified by a user.
        """
        if self._upper_bound is None:
            self.initialize_properties()
        if self._upper_bound is None:  # For if :meth:`initialize_properties` for some reason fails.
            raise ValueError("Expected 'upper_bound' to be set, but it is None.")
        return self._upper_bound

    @property
    def lower_bound(self) -> float:
        """The lowest quality value among all possible bitstrings.

        Can be calculated together with other bounds using the brute-force :meth:`initialize_properties`. Shouldn't be
        modified by a user.
        """
        if self._lower_bound is None:
            self.initialize_properties()
        if self._lower_bound is None:  # For if :meth:`initialize_properties` for some reason fails.
            raise ValueError("Expected 'lower_bound' to be set, but it is None.")
        return self._lower_bound

    @property
    def average_quality(self) -> float:
        """The average quality value over all possible bitstrings.

        Can be calculated together with bounds using the brute-force :meth:`initialize_properties`. Shouldn't be
        modified by a user. Meant to be used in comparison with QAOA results to see how much (if at all)
        the optimization improves over uniformly random guessing.
        """
        if self._average_quality is None:
            self.initialize_properties()
        if self._average_quality is None:  # For if :meth:`initialize_properties` for some reason fails.
            raise ValueError("Expected 'average_quality' to be set, but it is None.")
        return self._average_quality

    @property
    def best_quality(self) -> float:
        """The *best* quality value over all possible bitstrings.

        For minimization problems, this is equal to :attr:`lower_bound`. Like similar properties, it can be calculated
        using the brute-force :meth:`initialize_properties`. Shouldn't be modified by a user. Meant to be used in
        comparison with QAOA results to see how close the optimization gets to the ideal best solution.
        """
        if self._best_quality is None:
            self.initialize_properties()
        if self._best_quality is None:  # For if :meth:`initialize_properties` for some reason fails.
            raise ValueError("Expected 'best_quality' to be set, but it is ``None``.")
        return self._best_quality

    def quality_renormalized(self, bit_str: str) -> float:
        """Accepts a bitstring and returns renormalized quality of that bitstring.

        The quality is renormalized using :attr:`best_quality` and :attr:`average_quality`.

        * A value of 1 corresponds to the best solution.
        * A value of 0 corresponds to average quality.
        * A value above/under 0 corresponds to better/worse than average quality.

        Args:
            bit_str: The bitstring representing a solution.

        Returns:
            The renormalized quality of the solution as a float.

        """
        return (self.quality(bit_str) - self.average_quality) / (self.best_quality - self.average_quality)

    def average_quality_counts(self, counts: dict[str, int]) -> float:
        """Accepts a dictionary and returns the average quality of the keys weighted by their values.

        The input dictionary has the format of "counts" from a :mod:`qiskit` experiment. The keys are bitstrings
        (representing possible solutions) and the values are their respective counts, i.e., the number of times
        that the particular string was sampled from a QAOA run. The quality is calculated by :meth:`quality`.

        Args:
            counts: A dictionary whose keys are solution bitstrings and whose values are the respective counts.

        Returns:
            Average quality of the bitstrings, weighted by their counts.

        Raises:
            ValueError: If the number of measurements in ``counts`` is 0 (e.g., if it's an empty dictionary).

        """
        avg_quality: float = 0
        number_of_measurements = 0
        for bin_str, counter in counts.items():
            avg_quality += self.quality(bin_str) * counter
            number_of_measurements += counter
        if number_of_measurements == 0:
            raise ValueError("There are no counts. The quality can't be averaged.")
        return avg_quality / number_of_measurements

    def average_quality_renormalized(self, counts: dict[str, int]) -> float:
        """Accepts a dictionary and returns the renormalized quality of the keys weighted by their values.

        Calculates the average weighted quality using :meth:`average_quality_counts` and renormalizes it using
        :attr:`best_quality` and :attr:`average_quality`.

        * A value of 1 corresponds to the best solution.
        * A value of 0 corresponds to average quality.
        * A value above/under 0 corresponds to better/worse than average quality.

        Args:
            counts: A dictionary whose keys are solution bitstrings and whose values are the respective counts.

        Returns:
            Average quality of the bitstrings, weighted by their counts and renormalized.

        """
        return (self.average_quality_counts(counts) - self.average_quality) / (self.best_quality - self.average_quality)

    def restore_fixed_variables(self, counts: dict[str, int]) -> dict[str, int]:
        """Postprocessing method for restoring fixed variables to the measurement bitstrings.

        When variables are fixed, the number of variables of the (remaining) problem is reduced. When the problem is
        solved (e.g., by a quantum computer), the solutions doesn't include the fixed variables. This method takes
        a dictionary of solutions (e.g., the counts from a quantum computer) and modifies the keys (bitstrings) by
        inserting the fixed variables where they belong.

        Args:
            counts: A dictionary whose keys are bitstrings (solutions) and whose values are integers (their respective
                frequencies)

        Returns:
            The input dictionary corrected by inserting the fixed variables into the keys, where they belong.

        """
        new_counts: dict[str, int] = {}

        for bit_str, count in counts.items():
            bit_str_list = list(bit_str)

            # Insert removed characters at their respective positions
            for variable, value in sorted(self._fixed_variables.items()):
                variable_index = self._original_variables.index(variable)
                bit_str_list.insert(variable_index, str(value))

            new_counts["".join(bit_str_list)] = count

        return new_counts

    def local_bitflip_bitstring(self, bit_str: str) -> str:
        """Post-processing which takes a bitstring and replaces it with its lowest-energy unit Hamming distance
        neighbor.

        Takes the solution bitstring and then iteratively swaps each bit in it. The function returns the lowest-energy
        bitstring from all of these bitstrings (including the original bitstring).

        Args:
            bit_str: The bitstring to be replaced by its lowest-energy unit Hamming distance neighbor.

        Returns:
            The replaced bitstring.

        """
        best_quality = self.quality(bit_str)
        best_bitstring = bit_str

        for i, bit in enumerate(bit_str):
            if bit == "0":
                new_bit_str = bit_str[:i] + "1" + bit_str[i + 1 :]
            else:
                new_bit_str = bit_str[:i] + "0" + bit_str[i + 1 :]
            # Lower "quality" = better solution
            new_quality = self.quality(new_bit_str)
            if new_quality < best_quality:
                best_quality = new_quality
                best_bitstring = new_bit_str

        return best_bitstring

    # pylint: disable=anomalous-backslash-in-string
    def local_bitflip_postprocessing(self, counts: dict[str, int]) -> dict[str, int]:
        r"""Postprocessing method for checking a unit Hamming distance neighborhood of the dictionary of counts.

        When implemented naively, the time complexity of this scales cubically :math:`\mathcal{O}(n^3)` in the number
        of variables (linear from iterating over them and quadratic from calculating the energy), but some computation
        might be saved in the calculation of the energy because it's repeatedly calculated for very similar bitstrings.

        Args:
            counts: A dictionary whose keys are bitstrings (solutions) and whose values are integers (their respective
                frequencies)

        Returns:
            The input dictionary modified by replacing each bitstring by its lowest-energy neigbor.

        """
        new_counts: dict[str, int] = {}
        for bit_str, count in counts.items():
            new_bit_str = self.local_bitflip_bitstring(bit_str)
            if new_bit_str in new_counts:
                new_counts[new_bit_str] += count
            else:
                new_counts[new_bit_str] = count
        return new_counts

    def percentile_counts(
        self, counts: dict[str, int], quantile: float, best_percentile: bool = True
    ) -> dict[str, int]:
        """A method that selects only the best / worst ``quantile`` of given ``counts``, measured by :meth:`quality`.

        The quantile is weighted by the frequencies (counts) of the bitstrings. If multiple bitstrings around
        the ``quantile`` have the same quality, the order is selected arbitrarily (or rather, based on how the built-in
        ``sorted`` function sorts them). If a bitstring has counts that cross the ``quantile``, its counts in the output
        are adjusted to match the ``quantile`` exactly (at least rounded to the nearest integer).

        Args:
            counts: The input dictionary of counts.
            quantile: The quantile of counts to be selected.
            best_percentile: Boolean saying whether the "best" (lowest quality) or the "worst" (highest) bitstrings
                should be selected.

        Returns:
            A dictionary of counts, with only the best / worse bitstrings selected.

        Raises:
            ValueError: If the quantile is not between 0 and 1 (included).

        """
        if not 0 <= quantile <= 1:
            raise ValueError("The quantile has to be a number between 0 and 1 (included).")

        total_counts = sum(counts.values())
        to_select = int(total_counts * quantile)  # The number of counts to be included in ``selected_counts``.

        list_of_bitstrings = list(counts.keys())
        sorted_bitstrings = sorted(list_of_bitstrings, key=self.quality, reverse=not best_percentile)

        selected_counts: dict[str, int] = {}
        # Iterate over all bitstrings (sorted) and keep adding them to ``selected_counts`` as long as needed.
        for bitstring in sorted_bitstrings:
            if sum(selected_counts.values()) + counts[bitstring] >= to_select:
                if to_select - sum(selected_counts.values()) > 0:
                    selected_counts[bitstring] = to_select - sum(selected_counts.values())
                break
            selected_counts[bitstring] = counts[bitstring]

        return selected_counts

    def cvar(self, counts: dict[str, int], quantile: float = 0.05) -> float:
        """Calculates the Conditional Value at Risk (CVaR) of the given dictionary of counts at the given quantile.

        The CVaR is the average of the worst-case ``quantile`` of the data. In the case of training QAOA, it's often
        used to calculate the average of the best ``quantile`` of samples.

        Args:
            counts: The given dictionary of counts.
            quantile: The given quantile. Since it's common to calculate the CVaR at 5%, that's the default value for
                this variable

        Returns:
            The CVaR of the counts.

        """
        return self.average_quality_counts(self.percentile_counts(counts, quantile, True))
