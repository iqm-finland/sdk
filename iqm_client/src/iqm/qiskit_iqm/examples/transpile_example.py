# Copyright 2023 Qiskit on IQM developers
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
"""This file is an example of using Qiskit on IQM to run a simple but non-trivial quantum circuit on an IQM quantum
computer. See the Qiskit on IQM user guide for instructions:
https://docs.meetiqm.com/iqm-client/user_guide_qiskit.html
"""

import argparse

from iqm.qiskit_iqm.iqm_provider import IQMProvider
from qiskit import QuantumCircuit, transpile


def transpile_example(server_url: str) -> tuple[QuantumCircuit, dict[str, int]]:
    """Run a GHZ circuit transpiled using the Qiskit transpile function.

    Args:
        server_url: URL of the IQM server used for execution

    Returns:
        transpiled circuit, a mapping of bitstrings representing qubit measurement results to counts for each result

    """
    backend = IQMProvider(server_url).get_backend()

    num_qubits = min(backend.num_qubits, 5)  # use at most 5 qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    circuit.measure_all()

    transpiled_circuit = transpile(circuit, backend)
    counts = backend.run(transpiled_circuit, shots=1000).result().get_counts()

    return transpiled_circuit, counts


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--url",
        help="URL of the IQM service",
        default="https://cocos.resonance.meetiqm.com/deneb",
    )
    circuit_transpiled, results = transpile_example(argparser.parse_args().url)
    print(circuit_transpiled)
    print(results)
