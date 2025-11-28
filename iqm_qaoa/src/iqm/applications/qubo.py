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

from collections.abc import Mapping
from itertools import product
from typing import TypeVar, cast

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, to_networkx_graph
from dimod.sym import Sense
from dimod.typing import Variable
from dimod.utilities import new_variable_label
from dimod.vartypes import VartypeLike
from iqm.applications.applications import ProblemInstance
from iqm.applications.graph_utils import EDGE_ATTR_PRIORITY, NODE_ATTR_PRIORITY, _get_attr_with_priority, draw_problem
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
        vartype: An optional variable type for interpreting the input :class:`~numpy.ndarray` or
            :class:`~networkx.Graph` as :class:`~dimod.BinaryQuadraticModel`. The default value is 'BINARY'.
        allow_custom_var_names: Specifies whether the variable names should be renamed or if their original names should
            be kept. Internally, variables are labelled by consecutive integers starting from 0, so if the variables in
            the input ``qubo_object`` have different names, this should be set to ``True``.

    """

    def __init__(
        self,
        qubo_object: np.ndarray | nx.Graph | BinaryQuadraticModel,
        vartype: VartypeLike = "BINARY",
        allow_custom_var_names: bool = False,
    ) -> None:
        if (
            isinstance(qubo_object, np.ndarray)
            and qubo_object.ndim == 2  # noqa: PLR2004
            and qubo_object.shape[0] == qubo_object.shape[1]
        ):
            self._bqm = BinaryQuadraticModel(qubo_object, vartype=vartype)
        elif isinstance(qubo_object, nx.Graph):
            # The internal ``self._graph`` just saves the input graph, including its original node names.
            self._graph = qubo_object
            self._bqm = BinaryQuadraticModel(vartype=vartype)
            for node in qubo_object.nodes:
                value = _get_attr_with_priority(qubo_object.nodes[node], NODE_ATTR_PRIORITY)

                if value is None:
                    raise ValueError(
                        f"The node {node} is missing one of the required attributes ({', '.join(NODE_ATTR_PRIORITY)})."
                    )
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        f"The local term at node {node} has a "
                        f"value of type {type(value).__name__}, expected ``float`` or ``int``."
                    )

                self._bqm.add_linear(node, value)

            for u, v, data in qubo_object.edges(data=True):
                value = _get_attr_with_priority(data, EDGE_ATTR_PRIORITY)
                if value is None:
                    raise ValueError(
                        f"The edge between nodes {u} and {v} is missing"
                        f" one of the required attributes ({', '.join(EDGE_ATTR_PRIORITY)})."
                    )

                if not isinstance(value, (float, int)):
                    raise TypeError(
                        f"The edge between nodes {u} and {v} has a "
                        f"value of type {type(value).__name__}, expected ``float`` or ``int``."
                    )

                self._bqm.add_quadratic(u, v, value)

        elif isinstance(qubo_object, BinaryQuadraticModel):
            self._bqm = qubo_object
        else:
            raise ValueError(
                "The input is not a valid QUBO object. Valid objects are: 2D numpy array, networkx graph or dimod BQM."
            )
        if allow_custom_var_names:
            self._bqm, self.orig_to_new_labels, self.new_to_orig_labels = relabel_bqm_cqm_variables(self._bqm)
        elif set(self._bqm.variables) != set(range(self._bqm.num_variables)):
            raise ValueError(
                "The variables of the problem need to be labelled by integers starting from 0. "
                "Set `allow_custom_var_names` to `True` to allow custom variable labels."
            )
        else:
            self.orig_to_new_labels = {i: i for i in range(self._bqm.num_variables)}
            self.new_to_orig_labels = {i: i for i in range(self._bqm.num_variables)}

        super().__init__()
        self._bqm = self._bqm.binary  # For consistency, we transform the BQM to its binary form.
        self._original_bqm = self._bqm.copy()  # If the original BQM without fixed variables if ever needed.

    @property
    def dim(self) -> int:
        """The dimension of the problem (i.e., the number of binary variables)."""
        return self._bqm.num_variables

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

    def fix_variables(self, variables: list[int] | dict[int, int]) -> None:
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
        self._bqm.fix_variables(cast(Mapping[Variable, int], variables))

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

    def draw_problem(
        self,
        solution_bitstring: str | None = None,
        seed: int | None = None,
        highlight_edge_by_node_count: frozenset[int] = frozenset({2}),
    ) -> None:
        """The method to draw a graph corresponding to an instance of a QUBO problem, with the solution highlighted.

        Args:
            solution_bitstring: A bitstring representing a solution to the problem. Selected nodes are represented as
                1's, the other nodes are represented as 0's. This skips nodes that have been fixed. The full bitstring
                including the fixed nodes is restored internally by calling
                :meth:`~iqm.applications.applications.ProblemInstance.restore_fixed_variables_bitstring`.
            seed: The seed to derandomize the layout of the graph.
            highlight_edge_by_node_count: Contains the numbers of highlighted nodes that an edge has to be connected to,
                in order to be highlighted itself. In other words, if ``highlight_edge_by_node_count`` is
                ``frozenset({1})``, then edges connected to 1 highlighted node will be themselves highlighted. Edges
                connecting 2 highlighted nodes (or 2 non-highlighted nodes) will not be highlighted.

        """
        if hasattr(self, "_graph"):
            graph_to_plot = self._graph
        else:
            graph_to_plot = nx.relabel_nodes(self.qubo_graph, self.new_to_orig_labels)

        if solution_bitstring is not None:
            full_bitstring = self.restore_fixed_variables_bitstring(solution_bitstring)
        else:
            full_bitstring = None
        draw_problem(
            graph_to_plot=graph_to_plot,
            orig_to_new_mapping=self.orig_to_new_labels,
            fixed_vars=frozenset(self._fixed_variables.keys()),
            bitstring=full_bitstring,
            seed=seed,
            highlight_edge_by_node_count=highlight_edge_by_node_count,
        )


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
        allow_custom_var_names: Specifies whether the variable names should be renamed or if their original names should
            be kept. Internally, variables are labelled by consecutive integers starting from 0, so if the variables in
            the input ``cqm`` have different names, this should be set to ``True``.

    """

    def __init__(
        self, cqm: ConstrainedQuadraticModel, penalty: float = 1.0, allow_custom_var_names: bool = False
    ) -> None:
        super().__init__()
        if allow_custom_var_names:
            self._cqm, self.orig_to_new_labels, self.new_to_orig_labels = relabel_bqm_cqm_variables(cqm)
        elif set(cqm.variables) != set(range(cqm.num_variables())):
            raise ValueError(
                "The variables of the problem need to be labelled by integers starting from 0. "
                "Set `allow_custom_var_names` to `True` to allow custom variable labels."
            )
        else:
            self.orig_to_new_labels = {i: i for i in range(cqm.num_variables())}
            self.new_to_orig_labels = {i: i for i in range(cqm.num_variables())}
            self._cqm = cqm

        self._penalty = penalty
        self._bqm = BinaryQuadraticModel(vartype="BINARY")

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

    def fix_variables(self, variables: list[int] | dict[int, int]) -> None:
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
        # First recalculate BQM to make sure all constraints and objectives are accounted for.
        self._bqm = self._recalculate_bqm()
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
        # First recalculate BQM to make sure all constraints and objectives are accounted for.
        self._bqm = self._recalculate_bqm()
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
        """Take a dictionary and change the bitstrings in it in some minimal way so that they satisfy the constraints.

        Iterates through the dictionary ``counts`` and for each key calls :meth:`fix_constraint_violation_bitstring`.
        In case that multiple bitstrings get mapped to the same bitstring, their respective frequencies (values) are
        added.

        .. warning::
           The bitstrings in the `counts` need to be ordered the same way as the variables of the problem. If you're
           using a dictionary of counts obtained directly from a `qiskit` experiment, you need to reverse the order of
           the bitstrings (keys of the `counts` dictionary) first.

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
        """Take a solution bitstring and change it in some minimal way so that it satisfies the constraints.

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
        """Take a dictionary of counts and removes the bitstrings which don't satisfy the constraints.

        If none of the counts satisfy the constraints, the returned dictionary will be empty.

        .. warning::
           The bitstrings in the `counts` need to be ordered the same way as the variables of the problem. If you're
           using a dictionary of counts obtained directly from a `qiskit` experiment, you need to reverse the order of
           the bitstrings (keys of the `counts` dictionary) first.

        Args:
            counts: A dictionary of counts, with solution strings as keys and their frequencies as values.

        Returns:
            The same dictionary as inputted, except with removed entries whose keys don't satisfy the problem
            constraints.

        """
        counts_to_keep = {str: counts[str] for str in counts.keys() if self.constraints_checker(str)}

        return counts_to_keep


T = TypeVar("T", BinaryQuadraticModel, ConstrainedQuadraticModel)


def relabel_bqm_cqm_variables(
    bqm_or_cqm: T,
) -> tuple[T, dict[Variable, int], dict[int, Variable]]:
    """Map original variable names of the input quadratic model to consecutive integers starting from 0.

    Creates two dictionaries that keep track of the mapping between the original variable names and the new variable
    names.

    Args:
        bqm_or_cqm: The BQM/CQM whose variable names should be changed.

    Returns:
        A tuple containing the input BQM/CQM with renamed variables and two dictionaries containing the mapping from
        the old variable names to the new ones and vice versa.

    """
    orig_to_new_names = {orig: new for new, orig in enumerate(bqm_or_cqm.variables)}
    new_to_orig_names = dict(enumerate(bqm_or_cqm.variables))
    if set(bqm_or_cqm.variables) != set(range(bqm_or_cqm.num_variables)):
        re_named_bqm_or_cqm = bqm_or_cqm.relabel_variables(orig_to_new_names, inplace=False)
    else:
        re_named_bqm_or_cqm = bqm_or_cqm
    return re_named_bqm_or_cqm, orig_to_new_names, new_to_orig_names
