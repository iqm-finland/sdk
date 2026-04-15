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


def bell_measure(
    server_url: str, *, quantum_computer: str | None = None, token: str | None = None, shots: int = 1000
) -> dict[str, int]:
    """Execute a quantum circuit that prepares and measures a generalized Bell (aka GHZ) state.

    Args:
        server_url: URL of the IQM Server used for execution (e.g. "https://resonance.meetiqm.com/").
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one (e.g. "garnet"). ``None`` means connect to the
            default one.
        token: API token for authentication. If not given, uses :envvar:`IQM_TOKEN`.
        shots: Requested number of shots.

    Returns:
        Mapping of bitstrings representing qubit measurement results to counts for each result.

    """
    if quantum_computer is None:
        print(f"Executing a circuit on {server_url}.")
    else:
        print(f"Executing circuit on {server_url} quantum computer '{quantum_computer}'.")
    # Initialize a backend without metrics as IQMClient._get_calibration_quality_metrics is not supported by resonance
    backend = IQMProvider(server_url, quantum_computer=quantum_computer, token=token).get_backend()

    dqa = backend.client.get_dynamic_quantum_architecture()
    print(f"DQA={dqa}")

    # Check how many qubits we can use. Pick a qubit with most neighbors.
    coupling = backend.target.build_coupling_map()
    n_qubits = max(coupling.connected_components(), key=lambda cmap: cmap.size()).size()
    n_qubits = min(n_qubits, 5)  # use at most 5 qubits

    if n_qubits < 2:  # noqa: PLR2004
        raise ValueError("We need two qubits for the Bell state.")
    # Define a quantum circuit for a GHZ state.
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
    argparser.add_argument("--url", required=True, help='IQM Server URL, for example "https://resonance.meetiqm.com/"')
    argparser.add_argument(
        "--qc",
        help="ID or alias of the quantum computer to connect to, if the IQM Server instance controls more than one "
        '(e.g. "garnet")',
    )
    argparser.add_argument(
        "--token",
        help="API token for authentication",
        # Provide the API token explicitly or set it as an environment variable
        # following the Qiskit user guide at https://docs.meetiqm.com/iqm-client/
    )

    args = argparser.parse_args()
    counts = bell_measure(args.url, quantum_computer=args.qc, token=args.token)
    print(counts)
