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


def bell_measure(server_url: str) -> dict[str, int]:
    """Run a circuit that prepares and measures a Bell state.

    Args:
        server_url: URL of the IQM server used for execution

    Returns:
        a mapping of bitstrings representing qubit measurement results to counts for each result

    """
    backend = IQMProvider(server_url).get_backend()
    if backend.num_qubits < 2:
        raise ValueError("We need two qubits for the Bell state.")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    new_circuit = transpile(circuit, backend)
    return backend.run(new_circuit, shots=1000).result().get_counts()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--url",
        help="IQM server URL",
        # For example https://cocos.resonance.meetiqm.com/garnet
        default="https://<IQM SERVER>",
    )
    print(bell_measure(argparser.parse_args().url))
