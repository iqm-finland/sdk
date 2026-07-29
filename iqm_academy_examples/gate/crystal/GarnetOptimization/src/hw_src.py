import numpy as np
import sys
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates
from src.mapomatic_functions import matching_layouts
from src.utils import evaluate_costs


###################################################################################################
### QAOA circuit - complete graphs - small vertex set size - close-to-optimal ###
def hw_qaoa_circuit(num_qubits, num_layers, theta, W, w):

    qc = QuantumCircuit(num_qubits, num_qubits)
    cl_bits = np.arange(num_qubits).tolist()

    gamma = theta[:num_layers]
    beta = theta[num_layers:]

    for i in range(num_qubits):
        qc.h(i)

    for p in range(num_layers):

        ### Phase_separator
        ### two-body interactions
        if num_layers != 1:
            print("please set num_layers = 1")
            sys.exit()

        ############################################################################
        if num_qubits == 4:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[3]] * gamma[p], 0, 3)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

        ############################################################################
        elif num_qubits == 5:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[3]] * gamma[p], 0, 3)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)

            qc.cx(3, 4)
            qc.cx(4, 3)
            qc.cx(3, 4)

            cl_bits = swap_bit_indices(cl_bits, 3, 4)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[3]] * gamma[p], 0, 3)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

            qc.cx(2, 3)
            qc.cx(3, 2)
            qc.cx(2, 3)

            cl_bits = swap_bit_indices(cl_bits, 2, 3)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)

        ############################################################################
        elif num_qubits == 6:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)

            qc.cx(0, 5)
            qc.cx(5, 0)
            qc.cx(0, 5)

            cl_bits = swap_bit_indices(cl_bits, 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

            qc.cx(2, 3)
            qc.cx(3, 2)
            qc.cx(2, 3)

            cl_bits = swap_bit_indices(cl_bits, 2, 3)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)

            qc.cx(4, 5)
            qc.cx(5, 4)
            qc.cx(4, 5)

            cl_bits = swap_bit_indices(cl_bits, 4, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)

        ############################################################################
        elif num_qubits == 7:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)

            qc.cx(0, 5)
            qc.cx(5, 0)
            qc.cx(0, 5)

            cl_bits = swap_bit_indices(cl_bits, 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

            qc.cx(2, 3)
            qc.cx(3, 2)
            qc.cx(2, 3)

            cl_bits = swap_bit_indices(cl_bits, 2, 3)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)

            qc.cx(4, 5)
            qc.cx(5, 4)
            qc.cx(4, 5)

            cl_bits = swap_bit_indices(cl_bits, 4, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

            qc.cx(5, 6)
            qc.cx(6, 5)
            qc.cx(5, 6)

            cl_bits = swap_bit_indices(cl_bits, 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

        ############################################################################
        elif num_qubits == 8:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)

            qc.cx(0, 5)
            qc.cx(5, 0)
            qc.cx(0, 5)

            cl_bits = swap_bit_indices(cl_bits, 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

            qc.cx(2, 3)
            qc.cx(3, 2)
            qc.cx(2, 3)

            cl_bits = swap_bit_indices(cl_bits, 2, 3)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)

            qc.cx(4, 5)
            qc.cx(5, 4)
            qc.cx(4, 5)

            cl_bits = swap_bit_indices(cl_bits, 4, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

            qc.cx(5, 6)
            qc.cx(6, 5)
            qc.cx(5, 6)

            cl_bits = swap_bit_indices(cl_bits, 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)

            qc.cx(2, 7)
            qc.cx(7, 2)
            qc.cx(2, 7)

            cl_bits = swap_bit_indices(cl_bits, 2, 7)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

        ############################################################################
        elif num_qubits == 9:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)

            qc.cx(0, 5)
            qc.cx(5, 0)
            qc.cx(0, 5)

            cl_bits = swap_bit_indices(cl_bits, 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

            qc.cx(2, 3)
            qc.cx(3, 2)
            qc.cx(2, 3)

            cl_bits = swap_bit_indices(cl_bits, 2, 3)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)

            qc.cx(4, 5)
            qc.cx(5, 4)
            qc.cx(4, 5)

            cl_bits = swap_bit_indices(cl_bits, 4, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

            qc.cx(5, 6)
            qc.cx(6, 5)
            qc.cx(5, 6)

            cl_bits = swap_bit_indices(cl_bits, 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)

            qc.cx(2, 7)
            qc.cx(7, 2)
            qc.cx(2, 7)

            cl_bits = swap_bit_indices(cl_bits, 2, 7)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)

            qc.rzz(2 * W[cl_bits[7]][cl_bits[8]] * gamma[p], 7, 8)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[8]] * gamma[p], 1, 8)

            qc.cx(1, 8)
            qc.cx(8, 1)
            qc.cx(1, 8)

            cl_bits = swap_bit_indices(cl_bits, 1, 8)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

        ############################################################################
        elif num_qubits == 10:
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)

            qc.cx(0, 5)
            qc.cx(5, 0)
            qc.cx(0, 5)

            cl_bits = swap_bit_indices(cl_bits, 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)

            qc.cx(2, 3)
            qc.cx(3, 2)
            qc.cx(2, 3)

            cl_bits = swap_bit_indices(cl_bits, 2, 3)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)

            qc.cx(4, 5)
            qc.cx(5, 4)
            qc.cx(4, 5)

            cl_bits = swap_bit_indices(cl_bits, 4, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[4]] * gamma[p], 3, 4)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

            qc.cx(5, 6)
            qc.cx(6, 5)
            qc.cx(5, 6)

            cl_bits = swap_bit_indices(cl_bits, 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)

            qc.cx(2, 7)
            qc.cx(7, 2)
            qc.cx(2, 7)

            cl_bits = swap_bit_indices(cl_bits, 2, 7)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)

            qc.rzz(2 * W[cl_bits[7]][cl_bits[8]] * gamma[p], 7, 8)
            qc.rzz(2 * W[cl_bits[3]][cl_bits[9]] * gamma[p], 3, 9)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[8]] * gamma[p], 1, 8)

            qc.cx(1, 8)
            qc.cx(8, 1)
            qc.cx(1, 8)

            cl_bits = swap_bit_indices(cl_bits, 1, 8)
            qc.rzz(2 * W[cl_bits[7]][cl_bits[9]] * gamma[p], 7, 9)

            qc.cx(7, 9)
            qc.cx(9, 7)
            qc.cx(7, 9)

            cl_bits = swap_bit_indices(cl_bits, 7, 9)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[1]] * gamma[p], 0, 1)
            qc.rzz(2 * W[cl_bits[7]][cl_bits[8]] * gamma[p], 7, 8)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)
            qc.rzz(2 * W[cl_bits[1]][cl_bits[2]] * gamma[p], 1, 2)

            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)

            cl_bits = swap_bit_indices(cl_bits, 1, 2)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[3]] * gamma[p], 2, 3)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[7]] * gamma[p], 2, 7)

            qc.cx(2, 7)
            qc.cx(7, 2)
            qc.cx(2, 7)

            cl_bits = swap_bit_indices(cl_bits, 2, 7)
            qc.rzz(2 * W[cl_bits[2]][cl_bits[5]] * gamma[p], 2, 5)

            qc.cx(2, 5)
            qc.cx(5, 2)
            qc.cx(2, 5)

            cl_bits = swap_bit_indices(cl_bits, 2, 5)
            qc.rzz(2 * W[cl_bits[0]][cl_bits[5]] * gamma[p], 0, 5)
            qc.rzz(2 * W[cl_bits[5]][cl_bits[6]] * gamma[p], 5, 6)
            qc.rzz(2 * W[cl_bits[4]][cl_bits[5]] * gamma[p], 4, 5)

        ### local terms
        for i in range(num_qubits):
            qc.rz(2 * w[cl_bits[i]] * gamma[p], i)

        ### Mixer
        for i in range(num_qubits):
            qc.rx(2 * beta[p], i)

    qc.measure(np.arange(num_qubits).tolist(), cl_bits)

    return qc


def swap_bit_indices(vector, m, n):
    vector[m], vector[n] = vector[n], vector[m]
    return vector


def hw_qaoa_transpile(
    num_qubits, num_layers, theta, W, w, IQM_backend, calibration_data
):

    ### initial set of qubits for the hardwired transpilation
    n4 = [13, 14, 9, 8]
    n5 = [13, 14, 9, 8, 3]
    n6 = [13, 14, 9, 4, 3, 8]
    n7 = [13, 14, 9, 4, 3, 8, 7]
    n8 = [13, 14, 9, 4, 3, 8, 7, 10]
    n9 = [13, 14, 9, 4, 3, 8, 7, 10, 15]
    n10 = [13, 14, 9, 4, 3, 8, 7, 10, 15, 5]

    G_cmap_list = [n4, n5, n6, n7, n8, n9, n10]
    G_cmap = G_cmap_list[num_qubits - 4]
    cmap = IQM_backend.coupling_map
    reduced_map = cmap.reduce(mapping=G_cmap)

    qc_hw = hw_qaoa_circuit(num_qubits, num_layers, theta, W, w)
    qc_hw_transpiled = transpile(
        qc_hw, IQM_backend, optimization_level=3, coupling_map=reduced_map
    )
    qc_hw_1qopt = optimize_single_qubit_gates(qc_hw_transpiled)

    ### searching for the best layout
    ### aux_circ_trans is equivalent to qc_hw_1qopt on the complete map
    aux_circ = QuantumCircuit(IQM_backend.num_qubits, num_qubits)
    aux_circ_trans = aux_circ.compose(
        qc_hw_1qopt, qubits=G_cmap, clbits=np.arange(num_qubits).tolist()
    )

    layouts = matching_layouts(aux_circ_trans, cmap)
    scores = evaluate_costs(
        layouts, qc_hw_1qopt, IQM_backend, calibration_data, opt_level=2
    )
    layout_opt = [IQM_backend.qubit_name_to_index(name) for name in scores[0][0]]

    reduced_map_new = cmap.reduce(mapping=layout_opt)
    aux_circ_new = QuantumCircuit(IQM_backend.num_qubits, num_qubits)
    qc_hw_1qopt_opt = aux_circ_new.compose(
        transpile(
            qc_hw_1qopt, IQM_backend, optimization_level=3, coupling_map=reduced_map_new
        ),
        qubits=layout_opt,
        clbits=np.arange(num_qubits).tolist(),
    )

    return qc_hw_1qopt_opt


def obj_fun(bit_string, W, w):
    """Given a bit string bit_string, W, and w, this function returns
    the value of objective function
    Args:
        bit_string: (str) bit string
        W: a symmetric matrix (2D list)
        w: a vector (1D list)
    Returns:
        obj: (float) Objective
    """
    obj = 0
    num_qubits = len(bit_string)
    bit_string_reversed = bit_string[::-1]

    ########## this convention may be changed ##############
    b = np.array([int(bit) for bit in bit_string_reversed])
    s = 2 * b - np.ones(num_qubits)
    ########################################################

    for i in range(num_qubits - 1):
        for j in range(i + 1, num_qubits):
            obj = obj + W[i][j] * s[i] * s[j]

    for i in range(num_qubits):
        obj = obj - w[i] * s[i]

    return obj


def obj_fun_expectation(counts, W, w):
    """Computes expectation value based on measurement results
    Args:
        counts: (dict) key as bit string, val as count
    Returns:
        avg: (float) expectation value
    """
    avg = 0
    sum_count = 0
    for bit_string, count in counts.items():
        obj = obj_fun(bit_string, W, w)
        avg += obj * count
        sum_count += count
    return avg / sum_count


### single layer analytical energy
def get_energy_p1(params, W, w):
    # G[i][j] corresponds to edge weight of edge ij
    # this weight corresponds to the interaction strength
    gamma, beta = params
    w = -np.array(w)

    N = len(w)
    JZZ = 0.0
    for i in range(N):
        for j in range(i):
            node_list = [n for n in range(N)]
            node_list.remove(i)
            node_list.remove(j)

            f1 = 1.0
            f2 = 1.0
            for k in node_list:
                f1 *= np.cos(2 * gamma * W[i][k])
                f2 *= np.cos(2 * gamma * W[j][k])
            f1 *= np.cos(2 * gamma * w[i])
            f2 *= np.cos(2 * gamma * w[j])

            first = (
                0.5
                * W[i][j]
                * np.sin(4 * beta)
                * np.sin(2 * gamma * W[i][j])
                * (f1 + f2)
            )

            g1 = 1.0
            g2 = 1.0
            for k in node_list:
                g1 *= np.cos(2 * gamma * (W[i][k] + W[j][k]))
                g2 *= np.cos(2 * gamma * (W[i][k] - W[j][k]))
            g1 *= np.cos(2 * gamma * (w[i] + w[j]))
            g2 *= np.cos(2 * gamma * (w[i] - w[j]))

            second = 0.5 * W[i][j] * np.sin(2 * beta) ** 2 * (g1 - g2)

            JZZ += first - second

    hZ = 0.0
    for i in range(N):
        node_list = [n for n in range(N)]
        node_list.remove(i)

        h1 = 1.0
        for k in node_list:
            h1 *= np.cos(2 * gamma * W[i][k])
        h1 *= w[i] * np.sin(2 * beta) * np.sin(2 * gamma * w[i])

        hZ += h1

    return JZZ + hZ
