import requests

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def get_calibration_data(system, token):
    """Loading the latest calibration data.

    Args:
        system (str): name of the device being used
        token (str): API token

    Returns:
        single_qubit_fidelity (dict): dictionary containing the calibration data for single qubit gates.
        two_qubit_fidelity (dict): calibration data for the two qubits gates.
        readout_fidelity (dict): calibration data for the readout operations.
    """
    two_qubit_fidelity = {}
    single_qubit_fidelity = {}
    readout_fidelity = {}

    url = (
        f"https://api.resonance.meetiqm.com/quantum-computers/v1/{system}/calibrations"
    )
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    calibration = r.json()
    for iq in calibration["calibrations"][0]["metrics"][0]["metrics"]:
        temp = list(iq.values())
        two_qubit_fidelity[str(temp[0])] = temp[1]
        two_qubit_fidelity[str([temp[0][1], temp[0][0]])] = temp[1]

    for iq in calibration["calibrations"][0]["metrics"][1]["metrics"]:
        temp = list(iq.values())
        single_qubit_fidelity[str(temp[0])] = temp[1]

    for iq in calibration["calibrations"][0]["metrics"][3]["metrics"]:
        temp = list(iq.values())
        readout_fidelity[str(temp[0])] = temp[1]

    return two_qubit_fidelity, single_qubit_fidelity, readout_fidelity


def evaluate_costs(layouts, qc_algo, backend, cal_data, opt_level):
    """Evaluate the cost for each a given layout based on the calibration data.

    Args:
        layouts (list): list of possible layouts N qubit layouts
        qc_algo (QuantumCircuit): Quantum circuit of interest pre-transpilation.
        backend (bool, optional): Backend of the device to be considered.
        cal_data (dictionary): calibration data of the given hardware
        opt_level (int): Optimization_level to be considered for transpilation

    Returns:
        cost (list): List of scores for the respective layouts.
    """
    costs = []

    for layout in layouts:
        error = 0
        fid = 1
        reduced_map = backend.coupling_map.reduce(mapping=layout)
        aux_circ = QuantumCircuit(backend.num_qubits, qc_algo.num_clbits)
        qc_tranpiled = aux_circ.compose(
            transpile(
                qc_algo,
                basis_gates=["r", "cz"],
                optimization_level=opt_level,
                coupling_map=reduced_map,
            ),
            qubits=layout,
            clbits=list(range(qc_algo.num_clbits)),
        )
        for item in qc_tranpiled._data:
            if item[0].num_qubits == 2 and item[0].name != "barrier":
                mode = "two_qubit"
                q0 = qc_tranpiled.find_bit(item[1][0]).index
                q1 = qc_tranpiled.find_bit(item[1][1]).index
                qubit_list = str(["QB" + str(q0 + 1), "QB" + str(q1 + 1)])
                fid *= cal_data[0][qubit_list]

            elif item[0].name in ["r", "x"]:
                mode = "single_qubit"
                q0 = qc_tranpiled.find_bit(item[1][0]).index
                qubit_list = "QB" + str(q0 + 1)
                fid *= cal_data[1][qubit_list]

        error = 1 - fid
        costs.append([[backend.index_to_qubit_name(index) for index in layout], error])
        costs.sort(key=lambda x: x[1])
    return costs
