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
"""An example on using Qiskit on IQM to run a simple quantum circuit on an IQM quantum computer.

See the Qiskit on IQM user guide for instructions, found in the documentation at
https://docs.meetiqm.com/iqm-client/
"""

import argparse

from iqm.qiskit_iqm.iqm_provider import IQMProvider
from qiskit import QuantumCircuit, transpile


def bell_measure(server_url: str, token: str | None = None, shots: int = 1000) -> dict[str, int]:
    """Execute a quantum circuit that prepares and measures a generalized Bell (aka GHZ) state.

    Args:
        server_url: URL of the IQM Server used for execution
        token: API token for authentication. If not given, uses :env:`IQM_TOKEN`.
        shots: Requested number of shots.

    Returns:
        Mapping of bitstrings representing qubit measurement results to counts for each result.

    """
    print(f"Executing a circuit on {server_url}")
    # Initialize a backend without metrics as IQMClient._get_calibration_quality_metrics is not supported by resonance
    backend = IQMProvider(server_url, token=token).get_backend()
    if backend.num_qubits < 2:
        raise ValueError("We need two qubits for the Bell state.")

    # Just to make sure that "get_static_quantum_architecture" method works
    static_quantum_architecture = backend.client.get_static_quantum_architecture()
    print(f"static_quantum_architecture={static_quantum_architecture}")

    # Define a quantum circuit for a GHZ state
    n_qubits = min(backend.num_qubits, 5)  # use at most 5 qubits
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for qb in range(1, n_qubits):
        qc.cx(0, qb)
    qc.barrier()
    qc.measure_all()

    # Transpile the circuit
    qc_transpiled = transpile(qc, backend)
    print(qc_transpiled.draw(output="text"))

    # Run the circuit
    job = backend.run(qc_transpiled, shots=shots)
    return job.result().get_counts()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--url", required=True, help='IQM Server URL, for example "https://cocos.resonance.meetiqm.com/garnet"'
    )
    argparser.add_argument(
        "--token",
        help="API token for authentication",
        # Provide the API token explicitly or set it as an environment variable
        # following the Qiskit user guide at https://docs.meetiqm.com/iqm-client/
    )

    args = argparser.parse_args()
    counts = bell_measure(args.url, args.token)
    print(counts)
