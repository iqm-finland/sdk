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
"""Contains the TreeQAOA class, built for using the tree schedule.

Also contains a little helper function ``_find_nearest``.
"""

from collections.abc import Sequence
import re
import warnings

from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qaoa.tree_data_linear import angles
import numpy as np
from scipy.interpolate import interp1d


def _find_nearest(array: Sequence[int | float], value: int | float) -> tuple[int, int | float]:
    """Finds the index and value of the entry in a given ``array`` that is closest to a specified target ``value``.

    Args:
        array: The array of numeric values to search.
        value: The target value to find the nearest entry to.

    Returns:
        A tuple. The second element is the entry in ``array`` closest to ``value``. The first element is its index
        in ``array``.

    """
    array_as_np = np.asarray(array)
    idx = int((np.abs(array_as_np - value)).argmin())  # Cast as integer to satisfy type checker.
    return idx, array_as_np[idx]


# pylint: disable=too-many-locals
class TreeQAOA(QUBOQAOA):
    """The class for tree QAOA with QUBO cost function.

    The class inherits everything from :class:`~iqm.qaoa.qubo_qaoa.QUBOQAOA`, but adds one more method
    :meth:`set_tree_angles`, which sets :attr:`~iqm.qaoa.generic_qaoa.QAOA.angles` according to the tree schedule.
    The tree schedule uses QAOA angles precalculated numerically for a class of high-girth regular graph problems with
    uniform Hamiltonian. For more details see :cite:`Wybo_2024`.

    Args:
        problem: A :class:`~iqm.applications.qubo.QUBOInstance` object describing the QUBO problem to be solved.
        num_layers: The number of QAOA layers, commonly referred to as *p* in the literature.
        betas: An optional list of the initial *beta* angles of QAOA. Has to be provided together with ``gammas``.
        gammas: An optional list of the initial *gamma* angles of QAOA. Has to be provided together with ``betas``.
        initial_angles: An optional list of the initial QAOA angles as one variable. Shouldn't be provided together
            with either ``betas`` or ``gammas``.

    """

    def set_tree_angles(self) -> None:
        """A method for setting :attr:`iqm.qaoa.generic_qaoa.QAOA.angles` according to the tree schedule.

        The tree schedule is designed for problems with uniform 1-body local field, unit 2-body interactions and
        uniform node degree. Therefore, when given a generic problem, the algorithm first calculates the average
        degree ``d_av``, the average local field ``h_av`` and the average interaction strength ``j_av`` (which is used
        to renormalize the local field). The angles are then looked up in the saved files for the nearest implemented
        degree and local field. For :math:`p > 6` QAOA, the angles are interpolated from the calculated angles for
        :math:`p = 6`. The method doesn't output anything, but it modifies :attr:`iqm.qaoa.generic_qaoa.QAOA.angles`
        in-place.

        Raises:
            ValueError: If the Hamiltonian contains ferromagnetic (i.e., negative) interactions between qubits. The
                tree schedule shouldn't be used for such cases.

        """
        g = self.hamiltonian_graph  # From problem instance
        d_av = g.number_of_edges() * 2 / g.number_of_nodes()

        j_av = 0.0
        for i, j in g.edges():
            if g[i][j]["bias"] < 0:
                raise ValueError("Only anti-ferromagnetic (i.e., positive) interactions are allowed.")
            j_av += g[i][j]["bias"]
        j_av /= g.number_of_edges()  # Renormalize problem s.t. j_av = 1 --> is equivalent to rescaling gammas
        if abs(j_av) < 1e-4:
            warnings.warn(f"Warning: The average interaction strength {j_av} is very close to zero.", UserWarning)

        h_av = 0.0
        for i in g.nodes():
            h_av += g.nodes[i]["bias"]
        h_av /= g.number_of_nodes()

        pattern = re.compile(r"^RESULTS_D(\d+)$")
        # Extract integers from matching variable names
        implemented_ds = [
            int(match.group(1)) for each_variable in dir(angles) if (match := pattern.match(each_variable))
        ]

        d_round = round(d_av)  # Round input to integer

        implemented_hs = [0] + list(np.linspace(0.1, d_round + 1, 15))

        _, h_close = _find_nearest(implemented_hs, h_av / j_av)

        print("QAOA depth p = ", self.num_layers)
        print(
            "Tree angles for d = ",
            d_round,
            " rounded from ",
            d_av,
            ",[input] and h = ",
            h_close,
            " near ",
            h_av / j_av,
            " [input] (rescaled by average j_av = ",
            j_av,
            ")",
        )

        if d_round not in implemented_ds:
            raise ValueError(
                "For this regularity the tree calculation has not been performed yet. Implemented regularities are: "
                + " ".join(str(x) for x in implemented_ds)
            )
        if h_av / j_av > (d_round + 2) or h_av / j_av < 0:
            warnings.warn(
                "For this onsite field strength, the tree calculation has not been performed yet. "
                "Angles may not give good performance."
            )

        results = getattr(angles, f"RESULTS_D{d_round - 1:d}")

        # Get angles:
        if self.num_layers < 7:
            # Just take the angles obtained by the calculation
            self._angles[0::2] = np.array(results[h_close]["gammas"][self.num_layers - 1]) / (
                np.sqrt(d_round - 1) * j_av
            )
            self._angles[1::2] = -np.array(results[h_close]["betas"][self.num_layers - 1])
        elif self.num_layers > 6:
            # Interpolate between the tree angles at p = 6, x-range for these objects is [0,1]

            ps_norm = (np.arange(6) + 0.5) / 6  # The QAOA layers normalized to the interval [0, 1]
            interpolate = [0] + list(ps_norm) + [1]  # Adding edges of the interval to the list.

            # We need a value for the edges of the interval (0 and 1).
            # Those are chosen somewhat arbitrarily here. But it works. Trust Elisabeth.
            data_g = (
                [results[h_close]["gammas"][i][0] / 2]
                + results[h_close]["gammas"][i]
                + [results[h_close]["gammas"][i][-1] * 1.1]
            )
            data_b = (
                [results[h_close]["betas"][i][0] * 1.1]
                + results[h_close]["betas"][i]
                + [results[h_close]["betas"][i][-1] / 2]
            )
            f_gamma = interp1d(interpolate, data_g)
            f_beta = interp1d(interpolate, data_b)

            self._angles[0::2] = np.array(f_gamma((np.arange(self.num_layers) + 0.5) / self.num_layers)) / (
                np.sqrt(d_round - 1) * j_av
            )
            self._angles[1::2] = -np.array(f_beta((np.arange(self.num_layers) + 0.5) / self.num_layers))
        else:
            raise ValueError(f"Unexpected value for p: {self.num_layers}")

        self._trained = True
