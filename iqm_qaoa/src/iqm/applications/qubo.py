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
"""Contains problem instance classes for a generic QUBO problem and a genetic constrained QUBO problem.

While objects of these classes can be instantiated, it's generally not recommended. For problems with any added
structure, it is recommended to use sub-classes (children) of these classes, e.g.,
:class:`~iqm.applications.maxcut.MaxCutInstance`, :class:`~iqm.applications.mis.MISInstance` or
:class:`~iqm.applications.sk.SherringtonKirkpatrick`.

Example:

    .. code-block:: python

        from iqm.applications.qubo import QUBOInstance
        from iqm.applications.qubo import ConstrainedQuadraticInstance

        # Defining a QUBO instance from a ``numpy`` array of interactions.
        my_qubo_instance = QUBOInstance(my_square_np_array)
        my_qubo_instance.fix_variables([0, 2, 4])  # Fixes the variables labelled by 0, 2 and 4 to the value '1'.

        # For defining your own custom constrained quadratic binary subclass.
        class MyCustomConstrainedInstance(ConstrainedQuadraticInstance):
            def constraints_checker(self):
                # Implementation

"""

from itertools import product
from typing import Any

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, to_networkx_graph
from dimod.sym import Sense
from dimod.typing import Variable
from dimod.utilities import new_variable_label
from iqm.applications.applications import ProblemInstance
import networkx as nx
import numpy as np


class QUBOInstance(ProblemInstance):
    """A problem instance class for generic QUBO problems.

    This problem instance class should only be used for problems without any structure beyond having a QUBO
    cost function. Problems with constraints or more structure (such as maxcut, MIS, ...) should use sub-classes
    (children) of :class:`QUBOInstance`.

    The class is initialized with a ``qubo_object`` variable, which stores the QUBO parameters. Valid ``qubo_object``
    is either a 2D square :class:`~numpy.ndarray`, :class:`~networkx.Graph` or
    :class:`~dimod.BinaryQuadraticModel`. In the case of a :class:`~networkx.Graph`, the interactions need to be
    represented as the ``bias`` parameter of nodes / edges (treated as 0 if not present). Regardless of the type of
    ``qubo_object``, the :meth:`__init__` method internally saves the problem description as :attr:`bqm`, which is a
    :class:`~dimod.BinaryQuadraticModel`.

    Args:
        qubo_object: Either a square :class:`~numpy.ndarray`, a :class:`~networkx.Graph` or
            a :class:`~dimod.BinaryQuadraticModel` describing the QUBO problem.

    """

    def __init__(self, qubo_object: np.ndarray | nx.Graph | BinaryQuadraticModel) -> None:
        if isinstance(qubo_object, np.ndarray) and qubo_object.ndim == 2:
            self._bqm = BinaryQuadraticModel(qubo_object, vartype="BINARY")
        elif isinstance(qubo_object, nx.Graph):
            self._bqm = BinaryQuadraticModel(qubo_object.number_of_nodes(), vartype="BINARY")
            for node in qubo_object.nodes:
                self._bqm.add_linear(node, qubo_object.nodes[node].get("bias", 0))
            for u, v, data in qubo_object.edges(data=True):
                self._bqm.add_quadratic(u, v, data.get("bias", 0))
        elif isinstance(qubo_object, BinaryQuadraticModel):
            self._bqm = qubo_object
        else:
            raise ValueError(
                "The input is not a valid QUBO object. Valid objects are: 2D numpy array, networkx graph or dimod BQM."
            )
        super().__init__()
        self._original_variables = self._bqm.variables.copy()

    @property
    def dim(self) -> int:
        """The dimension of the problem (i.e., the number of binary variables)."""
        return self._bqm.num_variables

    # pylint: disable=anomalous-backslash-in-string
    @property
    def qubo_matrix(self) -> np.ndarray:
        r"""The QUBO matrix of the problem instance.

        The matrix is obtained from the internal variable :attr:`bqm`. If the QUBO cost function of the problem
        variables :math:`x_i \in \{0, 1\}` is described as:

        .. math:: C = \sum_{i<j} x_i Q_{ij} x_j + \sum_{i} x_i Q_{ii} x_i

        Then the output of this method is the square matrix :math:`Q_{ij}` as :class:`~numpy.ndarray`.

        - The diagonal entries corresponds to the local fields acting on the variables.
        - The entries above the diagonal correspond to the interactions between variables.
        - The entries below the diagonal are empty.
        """
        matrix = np.zeros((self._bqm.num_variables, self._bqm.num_variables))
        lin, (row, col, quad), *_ = self._bqm.to_numpy_vectors(sort_indices=True)
        np.fill_diagonal(matrix, lin)
        matrix[row, col] = quad
        return matrix

    @property
    def qubo_graph(self) -> nx.Graph:
        """The QUBO graph of the problem instance.

        The nodes / edges of the graph have a ``bias`` parameter containing the local field / interaction strength
        of the corresponding variable(s). Variable pairs without interaction aren't connected by edges in the graph.
        """
        return to_networkx_graph(self._bqm)

    @property
    def bqm(self) -> BinaryQuadraticModel:
        """The :class:`~dimod.BinaryQuadraticModel` representation of the problem instance.

        This variable is defined in :meth:`__init__` and is used throughout the class to calculate such quantities as
        :attr:`dim` or :attr:`qubo_matrix` lazily.
        """
        return self._bqm

    def fix_variables(self, variables: list[Variable] | dict[Variable, int]) -> None:
        """Fixes (assigns) some of the problem variables.

        Warning: For problems that come from a graph (such as maxcut), this doesn't change the original graph,
        only the derived QUBO formulation (i.e., the BQM variable)!

        Args:
            variables: Either a list of variables (which get all fixed to the value 1) or a dictionary with keys equal
                to the variables to fix and whose values are equal to the values to fix them to (either 1 or 0).

        Raises:
            ValueError: If one of the variables has already been fixed previously to a different value.

        """
        if isinstance(variables, list):
            variables = {var: 1 for var in variables}
        for var in variables.keys():
            if var in self._fixed_variables and self._fixed_variables[var] != variables[var]:
                raise ValueError(
                    f"The variable {var} has been fixed previously to {self._fixed_variables[var]}, "
                    f"but now it's attempted to be fixed to {variables[var]}."
                )
        self._fixed_variables.update(variables)
        self._bqm.fix_variables(variables)

    def quality(self, bit_str: str) -> float:
        """Accepts a bitstring (representing a solution) and returns that solution's quality / energy.

        Args:
            bit_str: The bitstring whose quality is being calculated.

        Returns:
            The energy of the input bitstring.

        """
        sol_vector = np.array([int(bit) for bit in bit_str])
        energy = self._bqm.energy(sol_vector)
        return float(energy)


# pylint: disable=too-many-instance-attributes
class ConstrainedQuadraticInstance(ProblemInstance):
    """A class for constrainted quadratic binary problems.

    The class saves the problem as a :class:`~dimod.ConstrainedQuadraticModel` and uses this object for its
    various methods. When the problem needs to be transformed into a QUBO, a private method :meth:`_recalculate_bqm`
    is used to return a :class:`~dimod.BinaryQuadraticModel` formulation of the problem, making the constraints
    soft and penalizing their breaking with :attr:`penalty`.

    Args:
        cqm: The problem encoded as a :class:`~dimod.ConstrainedQuadraticModel`, passed over from a subclass.
        penalty: The numerical penalty incurred by violating each constraint of the problem, to be used when
            the problem is transformed into a BQM.

    """

    def __init__(self, cqm: ConstrainedQuadraticModel, penalty: float = 1.0) -> None:
        super().__init__()
        self._cqm = cqm
        self._penalty = penalty
        self._bqm = BinaryQuadraticModel(vartype="BINARY")
        self._original_variables = self._cqm.variables.copy()

    @property
    def cqm(self) -> ConstrainedQuadraticModel:
        """The :class:`~dimod.ConstrainedQuadraticModel` representation of the problem instance."""
        return self._cqm

    @property
    def dim(self) -> int:
        """The dimension of the problem (i.e., the number of binary variables)."""
        return self._cqm.num_variables()

    @property
    def bqm(self) -> BinaryQuadraticModel:
        """The BQM representation of the problem, penalizing constraint violation.

        The problem represented as :class:`~dimod.BinaryQuadraticModel`. This is calculated using
        :meth:`_recalculate_bqm` by inserting the constraints as penalties to the cost function.
        """
        self._bqm = self._recalculate_bqm()
        return self._bqm

    def _recalculate_bqm(self) -> BinaryQuadraticModel:
        bqm_to_return = BinaryQuadraticModel(vartype="BINARY")

        # add the variables
        for v in self._cqm.variables:
            bqm_to_return.add_variable(v)

        # objective, we know it's always a BQM
        for v in self._cqm.objective.variables:
            bqm_to_return.add_linear(v, self._cqm.objective.get_linear(v))
        for u, v, bias in self._cqm.objective.iter_quadratic():
            bqm_to_return.add_quadratic(u, v, bias)

        for constraint in self._cqm.constraints.values():
            lhs = constraint.lhs
            rhs = constraint.rhs
            sense = constraint.sense

            if not lhs.is_linear():
                raise ValueError("CQM must not have any quadratic constraints.")

            if sense is Sense.Eq:
                bqm_to_return.add_linear_equality_constraint(
                    ((v, lhs.get_linear(v)) for v in lhs.variables),
                    self._penalty,
                    lhs.offset - rhs,
                )
            elif sense is Sense.Ge:
                bqm_to_return.add_linear_inequality_constraint(
                    ((v, lhs.get_linear(v)) for v in lhs.variables),
                    self._penalty,
                    new_variable_label(),
                    constant=lhs.offset,
                    lb=rhs,
                    ub=np.iinfo(np.int64).max,
                )
            elif sense is Sense.Le:
                bqm_to_return.add_linear_inequality_constraint(
                    ((v, lhs.get_linear(v)) for v in lhs.variables),
                    self._penalty,
                    new_variable_label(),
                    constant=lhs.offset,
                    lb=np.iinfo(np.int64).min,
                    ub=rhs,
                )

        return bqm_to_return

    def fix_variables(self, variables: list[Variable] | dict[Variable, int]) -> None:
        """Fixes (assigns) some of the problem variables.

        This method is not implemented in :class:`ConstrainedQuadraticInstance` because in the general case, fixing one
        variable might have implications for the other variables (implied by the constraints). Therefore, this method
        needs to be implemented in a subclass of :class:`ConstrainedQuadraticInstance`, if it's needed.

        Args:
            variables: Either a list of variables (which get all fixed to the value 1) or a dictionary with keys equal
                to the variables to fix and whose values are equal to the values to fix them to (either 1 or 0).

        """
        raise NotImplementedError(
            f"The problem instance class {self.__class__.__name__} does not have this method implemented."
        )

    @property
    def penalty(self) -> float:
        """The penalty for breaking the constraints."""
        return self._penalty

    @penalty.setter
    def penalty(self, new_penalty: float) -> None:
        """Whenver ``penalty`` is changed, the problem model needs to be changed correspondingly."""
        self._penalty = new_penalty
        self._bqm = self._recalculate_bqm()

    @property
    def qubo_matrix(self) -> np.ndarray:
        """The QUBO matrix of the problem instance.

        The matrix is obtained from the internal attribute :attr:`bqm`.

        - The i,i diagonal entry corresponds to the local field acting on the i-th variable.
        - The i,j entry above the diagonal corresponds to the interaction between the i-th and j-th variables.
        - The entries below the diagonal are empty.
        """
        matrix = np.zeros((self._bqm.num_variables, self._bqm.num_variables))
        lin, (row, col, quad), *_ = self._bqm.to_numpy_vectors(sort_indices=True)
        np.fill_diagonal(matrix, lin)
        matrix[row, col] = quad
        return matrix

    @property
    def qubo_graph(self) -> nx.Graph:
        """The QUBO graph of the problem instance.

        The nodes / edges of the graph have a ``bias`` parameter containing the local field / interaction strength
        of the corresponding variable(s). Variables without interaction aren't connected by edges in the graph.
        """
        return to_networkx_graph(self._bqm)

    def constraints_checker(self, bit_str: str) -> bool:
        """Checks whether the constrains of the problem are satisfied.

        Args:
            bit_str: A bitstring representing a solution.

        Returns:
            Bool indicating whether the solution satisfies the constraints or not.

        """
        bit_str_as_array = np.array([int(bit) for bit in bit_str])
        return self._cqm.check_feasible(bit_str_as_array)

    def loss(self, bit_str: str) -> float:
        """The loss function calculated for a given solution.

        It is equivalent to the quality of the solution, but with extra penalties for breaking the constraints.

        Args:
            bit_str: A bitstring representing a solution.

        Returns:
            The loss of the solution.

        """
        bit_str_as_array = np.array([int(bit) for bit in bit_str])
        return float(self.bqm.energy(bit_str_as_array))

    def quality(self, bit_str: str) -> float:
        """The quality function overridden for constrainted problems.

        For solutions violating the constraints, the "quality" isn't defined. If a user asks for the quality of
        the solution, this function first checks if the constraints are satisfied, prints out a warning if they
        aren't, and then returns the loss function.

        Args:
            bit_str: A bitstring representing a solution.

        Returns:
            The loss of the solution and a printed warning if the solution violates the constraints.

        """
        if not self.constraints_checker(bit_str):
            print("Constraint(s) violated, quality isn't defined, returning the loss function with default parameters")
        return self.loss(bit_str)

    def initialize_properties(self, max_size: int | None = 30) -> None:
        """The initialization method for upper/lower bound, average/best quality.

        This is the method from the parent class :class:`~iqm.applications.applications.ProblemInstance`, overridden so
        that the bruteforce search only includes solutions which satisfy the constraints.

        Args:
            max_size: The maximum size of problems for which the properties may be calculated.

        Raises:
            ValueError: If :meth:`initialize_properties` was called on a :class:`ConstrainedQuadraticInstance`
                object with dimension larger than ``max_size``.

        """
        if max_size is not None and self.dim > max_size:
            raise ValueError(
                f"The problem dimension {self.dim} exceeds the maximum of {max_size}. For large dimensions (>30),"
                f" the brute force approach of initialize_properties may be too slow. Change the ``max_size``"
                f" parameter or set it to ``None`` to bypass this error."
            )

        upper_bound: float = self.quality("0" * self.dim)
        lower_bound: float = self.quality("0" * self.dim)
        average_quality: float = 0
        number_of_passing_solutions = 0

        for bitstr in product("01", repeat=self.dim):
            if self.constraints_checker("".join(bitstr)):
                qlty = self.quality("".join(bitstr))
                upper_bound = max(upper_bound, qlty)
                lower_bound = min(lower_bound, qlty)
                average_quality += qlty
                number_of_passing_solutions += 1

        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._average_quality = average_quality / number_of_passing_solutions
        self._best_quality = lower_bound

    def fix_constraint_violation(self, counts: dict[str, int]) -> dict[str, int]:
        """Post-processing which takes a dictionary of counts and changes the bitstrings in it in some minimal way so
        that they all satisfy the constraints.

        Iterates through the dictionary ``counts`` and for each key calls :meth:`fix_constraint_violation_bitstring`.
        In case that multiple bitstrings get mapped to the same bitstring, their respective frequencies (values) are
        added.

        Args:
            counts: The dictionary of bitstrings with their frequencies as values.

        Returns:
            The input dictionary modified so that the keys now satisfy the problem constraints.

        """
        new_counts: dict[str, int] = {}
        for bit_str, count in counts.items():
            new_bit_str = self.fix_constraint_violation_bitstring(bit_str)
            if new_bit_str in new_counts:
                new_counts[new_bit_str] += count
            else:
                new_counts[new_bit_str] = count
        return new_counts

    def fix_constraint_violation_bitstring(self, bit_str: str) -> str:
        """Post-processing which takes a solution and changes it in some minimal way so that it satisfies the
        constraints.

        This is not possible to do generally, so this method is not implemented for :class:ConstrainedQuadraticInstance`
        and it needs to be defined for the individual subclasses of :class:`ConstrainedQuadraticInstance` (if it's
        possible to do). If the input bitstring ``bit_str`` already satisfies the problem constraints, it should be
        returned unchanged.

        Args:
            bit_str: The bitstring to be modified to satisfy the constraints.

        Returns:
            The bitstring modified in some minimal way to satisfy the constraints.

        """
        raise NotImplementedError(
            f"The problem instance class {self.__class__.__name__} does not have this method implemented."
        )

    def satisfy_constraints(self, counts: dict[str, int]) -> dict[str, int]:
        """Post-processing method that takes a dictionary and removes the bitstrings which don't satisfy the
        constraints.

        Args:
            counts: A dictionary of counts, with solution strings as keys and their frequencies as values.

        Returns:
            The same dictionary as inputted, except with removed entries whose keys don't satisfy the problem
            constraints.

        """
        counts_to_keep = {str: counts[str] for str in counts.keys() if self.constraints_checker(str)}

        return counts_to_keep


def relabel_graph_nodes(graph: nx.Graph) -> tuple[nx.Graph, dict[Any, int], dict[int, Any]]:
    """Map original node labels of the :class:`~networkx.Graph` to new ones between 0 and `graph.number_of_nodes` - 1.

    Creates two dictionaries that keep track of the mapping between original labels and new labels numbered between 0
    and `graph.number_of_nodes` - 1.

    Args:
        graph: The graph whose nodes should be relabeled.

    Returns:
        A tuple containing the input graph with relabeled nodes and two dictionaries containing the mapping from old
        labels to new and vice versa.

    """
    orig_to_new_labels = {orig: new for new, orig in enumerate(graph.nodes())}
    new_to_orig_labels = dict(enumerate(graph.nodes()))
    if set(graph.nodes) != set(range(len(graph))):
        re_labeled_graph = nx.relabel_nodes(graph, orig_to_new_labels)
    else:
        re_labeled_graph = graph
    return re_labeled_graph, orig_to_new_labels, new_to_orig_labels
