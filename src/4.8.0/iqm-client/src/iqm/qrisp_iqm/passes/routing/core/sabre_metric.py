# Copyright 2026 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains the code for efficiently computing the SABRE metric.

The metric computed here deviates from the original paper in some ways:

1. The extended set are ALL remaining instructions that are not in the
   front layer.
2. The instructions from the extended set are weighted with a decay
   mechanism depending on how many layers they are away from the front layer.
   If an instruction is in layer d (the front layer has d = 1), the weight
   is exp(-(d-2)/_DECAY_FACTOR).
3. The distance measure in the metric is taken to the power of 1.1. This
   has the effect that a pair of qubits that is already close together induces
   a stronger cost reduction if they are moved even closer compared to a pair
   that is far away. This is important because the instruction DAG is based
   on the Permeability DAG, which implies that the front-layer can contain
   instructions on the same qubit twice. This could yield the situation that
   the algorithm tries to move the qubit back and forth in order to improve
   the metric via either one or the other instruction.

For computing the metric, there are two implementations:

1. :func:`compute_sabre_cost` straight up computes the cost function
2. :func:`compute_sabre_cost_change` computes the cost change induced by a given SWAP
   more efficiently by only involving the instructions that actually contribute
   to the cost.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import find_path_length
from numba import njit
import numpy as np

if TYPE_CHECKING:
    from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import CircuitDAG, QPUTopology


_DISTANCE_POWER: float = 1.1
"""This variable regulates what power the distance is taken to."""

_DECAY_FACTOR: float = 7.0
"""This variable regulates how much the weight factor decays for each layer."""

_D_POWER_ARRAY = np.arange(3000, dtype=np.float32) ** _DISTANCE_POWER
"""To improve the speed of computing the power, we precompute the powers.
This is advantageous because the algorithm has to compute the same powers
often so precomputing them can give a significant boost in performance.
"""


@njit(cache=True)
def compute_sabre_cost(
    topology: QPUTopology,
    front_layer_indices: list[int],
    extended_set: list[int],
    instruction_data: np.ndarray,
    citizen_to_address: np.ndarray,
) -> float:
    """Computes the SABRE metric with the modifications mentioned in the preamble.

    Args:
        topology: QPU topology.
        front_layer_indices:
            The list of integer indices indicating which entries of the instruction
            data are currently at the front layer.
        extended_set:
            The list of integer indices indicating which entries of the instruction
            data are currently in the extended set.
        instruction_data:
            An array describing what instructions are available globally
            (see graph_processing_tools for more details).
        citizen_to_address:
            An array indicating the current permutation of qubits.

    Returns:
        The cost of the modified SABRE metric.

    """
    # This array is used to track the current layer each qubit is in while
    # iterating through the instructions
    circuit_depth = np.zeros(len(citizen_to_address), dtype=citizen_to_address.dtype)

    # This will contain the cost of the front layer
    front_layer_cost = 0.0

    # This will contain the normalization of the front layer
    front_layer_size = 0.0

    # This will contain the cost of the extended set
    extended_set_cost = 0.0

    # This will contain the normalization of the extended set
    extended_set_size = 0.0

    # Iterate through the front layer
    for i in range(len(front_layer_indices)):
        # Determine the qubits of the required instruction
        lnk = instruction_data[front_layer_indices[i]]

        # If the qubit agree, it is a single qubit instruction, which does not
        # contribute to the cost.
        if lnk[0] == lnk[1]:
            continue

        # Determine and update the depth of the current qubits
        d = max(circuit_depth[lnk[0]], circuit_depth[lnk[1]]) + 1
        circuit_depth[lnk[0]] = d
        circuit_depth[lnk[1]] = d

        # Determine the current addresses of the involved qubits
        a_0 = citizen_to_address[lnk[0]]
        a_1 = citizen_to_address[lnk[1]]

        # Compute the path length and add to the cost
        front_layer_cost += find_path_length(a_0, a_1, topology) ** _DISTANCE_POWER

        # The normalization factor get simply incremented by 1
        front_layer_size += 1

    for extset in extended_set:
        # Same steps as for the front layer
        lnk = instruction_data[extset]

        if lnk[0] == lnk[1]:
            continue

        a_0 = citizen_to_address[lnk[0]]
        a_1 = citizen_to_address[lnk[1]]

        d = max(circuit_depth[lnk[0]], circuit_depth[lnk[1]]) + 1

        circuit_depth[lnk[0]] = d
        circuit_depth[lnk[1]] = d

        # Now we include the weighting with the decay factor
        extended_set_cost += (find_path_length(a_0, a_1, topology)) ** _DISTANCE_POWER * np.exp(
            -(d - 2) / _DECAY_FACTOR
        )
        extended_set_size += np.exp(-(d - 2) / _DECAY_FACTOR)

    cost = 0.0
    # Compute the final cost by combining the cost of the front layer and extended set
    if front_layer_size:
        cost += front_layer_cost / front_layer_size

    if extended_set_size:
        cost += 0.5 * extended_set_cost / extended_set_size

    return cost


@njit(cache=True)
def filter_involvement_list(
    involvement_list: list[np.int16], scaling_array: np.ndarray, threshold: float
) -> list[np.int16]:
    """Produce a filtered copy of an involvement list.

    Keeps only the instruction indices whose scaling_array weight is
    >= threshold.

    This is called once per ``depth_array_update_required`` cycle in
    :func:`.sabre_gen_swaps` so that the hot :func:`.compute_sabre_cost_change`
    loop never iterates over entries whose contribution has decayed to
    near-zero.  Filtering once per DAG advance is far cheaper than
    checking the threshold inside the per-swap-candidate loop.

    Args:
        involvement_list:
            A per-qubit list of instruction indices (front-layer or extended-set).
        scaling_array:
            The combined weight array (exp_depth * criticality) indexed by
            instruction index.
        threshold:
            Minimum weight for an entry to survive the filter.

    Returns:
        A new list containing only the entries with weight >= threshold.
        If *threshold* is zero the input list is returned unchanged
        (avoids an allocation).

    """
    if threshold == np.float32(0.0):
        return involvement_list

    result = [np.int16(x) for x in range(0)]
    for i in range(len(involvement_list)):
        ind = involvement_list[i]
        if scaling_array[ind] >= threshold:
            result.append(ind)

    return result


@njit(cache=True)
def compute_sabre_cost_change(  # noqa: PLR0913, PLR0912
    instruction_data: np.ndarray,
    fl_inv_0: list[int],
    fl_inv_1: list[int],
    es_inv_0: list[int],
    es_inv_1: list[int],
    old_c2a: np.ndarray,
    swap_0: int,
    swap_1: int,
    exp_depth_array: np.ndarray,
    front_layer_prefactor: float,
    extended_layer_prefactor: float,
    distance_matrix: np.ndarray,
) -> float:
    """Compute the cost change induced by a given swap.

    For this it only considers the terms that actually contribute, which
    is much more efficient.

    Instead of receiving a full new_c2a array (which requires a copy per call),
    the swap is described by the two citizen indices (swap_0, swap_1). The new
    addresses are resolved inline with a branch, avoiding an array allocation
    per swap candidate.

    The involvement lists are passed as 4 separate lists (two per layer type),
    one for each qubit of the swap. This avoids the allocation of a concatenated
    list per swap candidate.

    Parameters
    ----------
    instruction_data : np.ndarray
        An array describing what instructions are available globally
        (see graph_processing_tools for more details).
    fl_inv_0 : list[int]
        Front layer involvement list of the first swap qubit.
    fl_inv_1 : list[int]
        Front layer involvement list of the second swap qubit.
    es_inv_0 : list[int]
        Extended set involvement list of the first swap qubit.
    es_inv_1 : list[int]
        Extended set involvement list of the second swap qubit.
    old_c2a : np.ndarray
        An array indicating the current permutation of qubits.
    swap_0 : int
        The citizen index of the first qubit being swapped.
    swap_1 : int
        The citizen index of the second qubit being swapped.
    exp_depth_array : np.ndarray
        An array containing the precomputed weight factors of each instruction.
    front_layer_prefactor : float
        The precomputed factor that is used to weight the front layer part of
        the metric.
    extended_layer_prefactor : float
        The precomputed factor that is used to weight the extended set part of
        the metric.
    distance_matrix : np.ndarray
        The distance matrix from the topology tuple.

    Returns
    -------
    float
        The computed metric change.

    """
    # Pre-fetch the swapped addresses once
    addr_swap_0 = old_c2a[swap_0]
    addr_swap_1 = old_c2a[swap_1]

    front_layer_cost = np.float32(0)

    # Iterate through both front layer involvement lists
    for _list_idx in range(2):
        lst = fl_inv_0 if _list_idx == 0 else fl_inv_1
        for i in range(len(lst)):
            ind = lst[i]

            lnk0 = instruction_data[ind, 0]
            lnk1 = instruction_data[ind, 1]

            # Single-qubit gates are filtered at the source (involvement list
            # construction in sabre_workflow.py), so every entry here is a
            # two-qubit gate and no continue guard is needed.

            # Compute the old addresses
            old_a_0 = old_c2a[lnk0]
            old_a_1 = old_c2a[lnk1]

            # Compute the new addresses via inline swap resolution
            if lnk0 == swap_0:
                new_a_0 = addr_swap_1
            elif lnk0 == swap_1:
                new_a_0 = addr_swap_0
            else:
                new_a_0 = old_a_0

            if lnk1 == swap_0:
                new_a_1 = addr_swap_1
            elif lnk1 == swap_1:
                new_a_1 = addr_swap_0
            else:
                new_a_1 = old_a_1

            # Compute the old path length vs the new path length
            path_length_new = distance_matrix[new_a_0, new_a_1]
            path_length_old = distance_matrix[old_a_0, old_a_1]

            scaling = exp_depth_array[ind]

            # Update the cost
            front_layer_cost += scaling * (_D_POWER_ARRAY[path_length_new] - _D_POWER_ARRAY[path_length_old])

    extended_layer_cost = np.float32(0)

    # Iterate through both extended set involvement lists
    for _list_idx in range(2):
        lst = es_inv_0 if _list_idx == 0 else es_inv_1
        for i in range(len(lst)):
            ind = lst[i]
            lnk0 = instruction_data[ind, 0]
            lnk1 = instruction_data[ind, 1]

            old_a_0 = old_c2a[lnk0]
            old_a_1 = old_c2a[lnk1]

            if lnk0 == swap_0:
                new_a_0 = addr_swap_1
            elif lnk0 == swap_1:
                new_a_0 = addr_swap_0
            else:
                new_a_0 = old_a_0

            if lnk1 == swap_0:
                new_a_1 = addr_swap_1
            elif lnk1 == swap_1:
                new_a_1 = addr_swap_0
            else:
                new_a_1 = old_a_1

            path_length_new = distance_matrix[new_a_0, new_a_1]
            path_length_old = distance_matrix[old_a_0, old_a_1]

            # Add the scaling factor
            scaling = exp_depth_array[ind]

            extended_layer_cost += scaling * (_D_POWER_ARRAY[path_length_new] - _D_POWER_ARRAY[path_length_old])

    # Return the combined cost
    return float(front_layer_prefactor * front_layer_cost + extended_layer_prefactor * extended_layer_cost)


@njit(cache=True)
def compute_depth_array(
    front_layer_indices: list[int], extended_set: list[int], instruction_data: np.ndarray, N: tuple
) -> np.ndarray:
    """Precompute the depth array.

    Computes for a given combination of front layer and extended set.

    Parameters
    ----------
    front_layer_indices : list[int]
        A list of integers indicating which entries of instruction data
        constitute the front layer.
    extended_set : list[int]
        A list of integers indicating which entries of instruction data
        constitute the extended set.
    instruction_data : np.ndarray
        An array describing what instructions are available globally
        (see graph_processing_tools for more details).
    N : tuple
        A topology tuple ``(indptr, indices, distance_matrix, predecessors)``
        as returned by :func:`connectivity_to_topology`.  The distance
        matrix at ``N[2]`` determines the output dtype (int16 on NISQ-scale
        devices).

    Returns
    -------
    depth_array : np.ndarray
        An array indicating what layer each instruction from the front layer
        and extended set lives in.

    """
    # Inherit dtype from the distance matrix (topology[2]) so we stay narrow
    dt = N[2].dtype

    # This array tracks the current layer of each qubit
    circuit_depth = np.zeros(N[2].shape[0], dtype=dt)

    # This array contains the result
    depth_array = np.zeros(len(instruction_data), dtype=dt)

    # Iterate through both the front layer and the extended set and compute the
    # depth.
    for i in range(len(front_layer_indices)):
        lnk = instruction_data[front_layer_indices[i]]

        if lnk[0] == lnk[1]:
            depth_array[front_layer_indices[i]] = circuit_depth[lnk[0]]
            continue

        d = max(circuit_depth[lnk[0]], circuit_depth[lnk[1]]) + 1

        circuit_depth[lnk[0]] = d
        circuit_depth[lnk[1]] = d

        depth_array[front_layer_indices[i]] = d

    for i in range(len(extended_set)):
        lnk = instruction_data[extended_set[i]]

        if lnk[0] == lnk[1]:
            depth_array[extended_set[i]] = circuit_depth[lnk[0]]
            continue

        d = max(circuit_depth[lnk[0]], circuit_depth[lnk[1]]) + 1

        circuit_depth[lnk[0]] = d
        circuit_depth[lnk[1]] = d

        depth_array[extended_set[i]] = d

    return depth_array


@njit(cache=True)
def precompute_exp_depth(depth_array: np.ndarray) -> np.ndarray:
    """Precompute the decay factor from the depth array.

    Parameters
    ----------
    depth_array : np.ndarray
        The array indicating which layer each instruction is involved in.

    Returns
    -------
    np.ndarray
        The precomputed prefactors.

    """
    return np.exp(-(depth_array - 2) / _DECAY_FACTOR).astype(np.float32)


@njit(cache=True)
def compute_extended_set_prefactor(
    extended_set: list[int], instruction_data: np.ndarray, exp_depth_array: np.ndarray
) -> float:
    """Precompute the prefactor of the extended set.

    Parameters
    ----------
    extended_set : list[int]
        A list of integers indicating which entries of instruction data
        constitute the extended set.
    instruction_data : np.ndarray
        An array describing what instructions are available globally
        (see graph_processing_tools for more details).
    exp_depth_array : TYPE
        The array of precomputed prefactors.

    Returns
    -------
    float
        The prefactor.

    """
    res = 0
    for i in range(len(extended_set)):
        instruction = instruction_data[extended_set[i]]
        if instruction[0] == instruction[1]:
            continue
        res += exp_depth_array[extended_set[i]]

    if res != 0:
        return 0.5 / res
    else:
        return 0


@njit(cache=True)
def compute_front_layer_prefactor(
    front_layer_indices: list[int], instruction_data: np.ndarray, depth_array: np.ndarray
) -> float:
    """Precompute the prefactor of the front layer.

    Parameters
    ----------
    front_layer_indices : list[int]
        A list of integers indicating which entries of instruction data
        constitute the front layer.
    instruction_data : np.ndarray
        An array describing what instructions are available globally
        (see graph_processing_tools for more details).
    depth_array : np.ndarray
        The depth array indicating the layer of each instruction.

    Returns
    -------
    float
        The prefactor.

    """
    res = 0
    for i in range(len(front_layer_indices)):
        instruction = instruction_data[front_layer_indices[i]]
        if instruction[0] == instruction[1]:
            continue
        res += 1
    return 1 / res


@njit(cache=True)
def find_involvement(
    instruction_data: np.ndarray, front_layer_indices: list[int], extended_set: list[int], topology: QPUTopology
) -> tuple[list[list[np.int16]], list[list[np.int16]]]:
    """Determine involvement lists for SABRE cost-change computation.

    Describes which instruction each qubit is involved in. This is required
    in order to call the
    compute_sabre_cost_change function with the corresponding relevant instructions.

    Single-qubit gates are excluded from the involvement lists because they
    have distance 0 under any qubit placement and therefore contribute nothing
    to the SABRE cost metric.

    Parameters
    ----------
    instruction_data : np.ndarray
        An array describing what instructions are available globally
        (see graph_processing_tools for more details).
    front_layer_indices : list[int]
        A list of integers indicating which entries of instruction data
        constitute the front layer.
    extended_set : list[int]
        A list of integers indicating which entries of instruction data
        constitute the extended_set.
    topology : tuple
        A topology tuple (see graph_processing_tools for more details).

    Returns
    -------
    front_layer_res : list[list[int]]
        A list of lists indicating which front layer instructions each qubit is
        involved in.
    extended_set_res : list[list[int]]
        A list of lists indicating which extended set instructions each qubit is
        involved in.

    """
    # Create the result container, i.e. a list of empty lists.
    # Use np.int16 for involvement entries to match the narrow type
    # used throughout the hot path.
    front_layer_res = [[np.int16(x) for x in range(0)] for i in range(topology[1].shape[0])]

    # Iterate through the front layer.
    # Single-qubit gates do not contribute to the SABRE metric, so they
    # are omitted from the involvement lists.
    for i in range(len(front_layer_indices)):
        instruction = instruction_data[front_layer_indices[i]]
        if instruction[0] != instruction[1]:
            front_layer_res[instruction[0]].append(front_layer_indices[i])
            front_layer_res[instruction[1]].append(front_layer_indices[i])

    extended_set_res = [[np.int16(x) for x in range(0)] for i in range(topology[1].shape[0])]

    for i in range(len(extended_set)):
        instruction = instruction_data[extended_set[i]]
        if instruction[0] != instruction[1]:
            extended_set_res[instruction[0]].append(extended_set[i])
            extended_set_res[instruction[1]].append(extended_set[i])

    return front_layer_res, extended_set_res


@njit(cache=True)
def compute_sabre_cost_from_dag(topology: QPUTopology, instruction_dag: CircuitDAG, c2a: np.ndarray) -> float:
    """Compute the SABRE cost from a given instruction DAG.

    Evaluates under a given initial layout for a given topology.

    Parameters
    ----------
    topology : tuple
        A topology tuple (see graph_processing_tools for more details).
    instruction_dag : tuple
        An instruction DAG tuple (see graph_processing_tools for more details).
    c2a : np.ndarray
        An array describing the initial layout permutation.

    Returns
    -------
    float
        The SABRE metric.

    """
    # We determine the front layer and the extended set.
    # It is important to filter out the single qubit gates out of the front layer
    # as they cause an error.

    indptr = instruction_dag[0]
    indices = instruction_dag[1]
    instruction_data = instruction_dag[2]
    idt = instruction_data.dtype

    node_amount = len(indptr) - 1

    # Compute the in-degree of each node
    in_degree = np.zeros(node_amount, dtype=idt)

    for i in range(node_amount):
        for j in indices[indptr[i] : indptr[i + 1]]:
            # If j is less than 0, this stems from a pruned DAG and can be ignored
            # (see graph_processing_tools for more details).
            if j < 0:
                continue
            in_degree[j] += 1

    # Create a queue and enqueue all vertices with
    # indegree 0
    front_layer_indices = [np.int16(i) for i in range(node_amount) if in_degree[i] == 0]

    i = 0
    # filter single qubit gates
    while i < (len(front_layer_indices)):
        # front layer index
        fli = front_layer_indices[i]

        lnk = instruction_data[fli]

        if lnk[0] == lnk[1]:
            front_layer_indices.pop(i)

            # Update in degree array
            for j in indices[indptr[fli] : indptr[fli + 1]]:
                if j < 0:
                    continue
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    front_layer_indices.append(np.int16(j))

            i = 0
            continue

        i += 1

    extended_set = np.nonzero(in_degree)[0].astype(np.int16)

    return compute_sabre_cost(topology, front_layer_indices, extended_set, instruction_data, c2a)
