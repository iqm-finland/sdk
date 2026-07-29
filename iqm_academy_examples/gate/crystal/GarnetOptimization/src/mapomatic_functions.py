# This code is part of Mapomatic licenced under Apache 2.0. (https://github.com/qiskit-community/mapomatic)
# 
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The exact_mappings routine is more or less a direct port of the VF2Layout
# pass code from Qiskit with minor modifications to simplify and return a
# different format.  The Qiskit code is under the following license:

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from rustworkx import (
    PyGraph,
    PyDiGraph,
    vf2_mapping,
)
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
import random


def deflate_circuit(input_circ):
    """Reduce a transpiled circuit down to only active qubits.

    Args:
        input_circ (QuantumCircuit): Input circuit.

    Returns:
        QuantumCircuit: Reduced circuit.
    """
    active_qubits, active_clbits = active_bits(input_circ)

    num_reduced_qubits = len(active_qubits)
    num_reduced_clbits = len(active_clbits)

    active_qubit_map = {}
    active_bit_map = {}
    for idx, val in enumerate(
        sorted(active_qubits, key=lambda x: input_circ.find_bit(x).index)
    ):
        active_qubit_map[val] = idx
    for idx, val in enumerate(
        sorted(active_clbits, key=lambda x: input_circ.find_bit(x).index)
    ):
        active_bit_map[val] = idx

    new_qc = QuantumCircuit(num_reduced_qubits, num_reduced_clbits)
    for item in input_circ.data:
        # Find active qubits used by instruction (if any)
        used_active_set = [qubit for qubit in item[1] if qubit in active_qubits]
        # If any active qubits used, add to deflated circuit
        if any(used_active_set):
            ref = getattr(new_qc, item[0].name)
            params = item[0].params
            qargs = [
                new_qc.qubits[active_qubit_map[qubit]] for qubit in used_active_set
            ]
            cargs = [new_qc.clbits[active_bit_map[clbit]] for clbit in item[2]]
            ref(*params, *qargs, *cargs)
    new_qc.global_phase = input_circ.global_phase
    return new_qc


def active_bits(input_circ):
    """Find active bits (quantum and classical) in a transpiled circuit.

    Args:
        input_circ (QuantumCircuit): Input circuit.

    Returns:
        tuple: Tuple of sets for active qubits and active classical bits

    """
    active_qubits = set([])
    active_clbits = set([])
    for item in input_circ.data:
        if item[0].name not in ["barrier", "delay"]:
            qubits = item[1]
            for qubit in qubits:
                active_qubits.add(qubit)
            clbits = item[2]
            for clbit in clbits:
                active_clbits.add(clbit)

    return active_qubits, active_clbits


def matching_layouts(trans_circ, cmap, strict_direction=True, call_limit=int(3e7)):
    """Finds the matching layouts for the transpiled circuit and given coupling map.

    Args:
        trans_circ (QuantumCircuit): Initial transpiled circuit
        cmap (dict): Dictionary of mapping qubits to coupling map.

    Returns:
        matching_layouts (list): List of matching layouts.

    """

    qc = deflate_circuit(trans_circ)
    dag = circuit_to_dag(qc)
    qubits = dag.qubits
    qubit_indices = {qubit: index for index, qubit in enumerate(qubits)}

    interactions = []
    for node in dag.op_nodes(include_directives=False):
        len_args = len(node.qargs)
        if len_args == 2:
            interactions.append(
                (qubit_indices[node.qargs[0]], qubit_indices[node.qargs[1]])
            )
    if strict_direction:
        cm_graph = cmap.graph
        im_graph = PyDiGraph(multigraph=False)
    else:
        cm_graph = cmap.graph.to_undirected()
        im_graph = PyGraph(multigraph=False)

    cm_nodes = list(cm_graph.node_indexes())
    seed = -1
    if seed != -1:
        random.Random(seed).shuffle(cm_nodes)
        shuffled_cm_graph = type(cm_graph)()
        shuffled_cm_graph.add_nodes_from(cm_nodes)
        new_edges = [
            (cm_nodes[edge[0]], cm_nodes[edge[1]]) for edge in cm_graph.edge_list()
        ]
        shuffled_cm_graph.add_edges_from_no_data(new_edges)
        cm_nodes = [k for k, v in sorted(enumerate(cm_nodes), key=lambda item: item[1])]
        cm_graph = shuffled_cm_graph

    im_graph.add_nodes_from(range(len(qubits)))
    im_graph.add_edges_from_no_data(interactions)

    mappings = vf2_mapping(
        cm_graph,
        im_graph,
        subgraph=True,
        id_order=False,
        induced=False,
        call_limit=call_limit,
    )
    layouts = []
    for mapping in mappings:
        # Here we sort in the order that we would use
        # for intial layout
        temp_list = [None] * qc.num_qubits
        for cm_i, im_i in mapping.items():
            key = qubits[im_i]
            val = cm_nodes[cm_i]
            temp_list[qc.find_bit(key).index] = val
        layouts.append(temp_list)

    sets = []
    for mapping in layouts:
        temp = set(mapping)
        if temp not in sets:
            sets.append(temp)
    matching_layouts = [list(xx) for xx in sets]
    return matching_layouts

