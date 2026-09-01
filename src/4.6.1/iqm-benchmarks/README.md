# IQM Benchmarks

IQM Benchmarks is a suite of Quantum Characterization, Verification, and
Validation (QCVV) tools for quantum computing. It is designed to be a
comprehensive tool for benchmarking quantum hardware. The suite is designed
to be modular, allowing users to easily add new benchmarks and customize
existing ones. The suite is designed to be easy to use, with a simple API
that allows users to run benchmarks with a single command.

Below is a list of the benchmarks currently available in the suite:

- Gates / Layers:
  - Standard Clifford Randomized Benchmarking
    [[Phys. Rev. A 85, 042311](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.85.042311)
    (2012)]
  - Interleaved Randomized Benchmarking
    [[Phys. Rev. Lett. 109, 080505](https://doi.org/10.1103/PhysRevLett.109.080505)
    (2012)]
  - Compressive Gate Set Tomography
    [[PRX Quantum 4, 010325](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010325)
    (2023)] (Optional dependencies required)
  - Mirror Randomized Benchmarking
    [[Phys. Rev. Lett. 129, 150502](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.150502)
    (2022)]
  - Error Per Layered Gate
    [[arXiv:2311.05933 [quant-ph]](https://arxiv.org/abs/2311.05933)
    (2023)]
- Holistic:
  - Quantum Volume
    [[Phys. Rev. A 100, 032328](https://doi.org/10.1103/PhysRevA.100.032328)
    (2019)]
  - CLOPS
    [[arXiv:2110.14108 [quant-ph]](https://arxiv.org/abs/2110.14108)
    (2021)]
- Entanglement:
  - GHZ State Fidelity
    [[arXiv:0712.0921 [quant-ph]](https://arxiv.org/abs/0712.0921)
    (2007)]
  - Graph State Bipartite Entanglement
    [[Adv. Quantum Technol., 2100061](https://doi.org/10.1002/qute.202100061)
    (2021)]
- Optimization:
  - Q-Score
    [[IEEE Trans. Quantum Eng., 2](https://doi.org/10.1109/TQE.2021.3090207)
    (2021)]

The project is split into different benchmarks, all inheriting
from the `Benchmark` parent class.
Each individual benchmark takes
as an argument their own `BenchmarkConfigurationBase` class.

## Characterize Physical Hardware

The IQM Benchmarks suite is designed to be used with real quantum hardware.
To use the suite, you will need to have access to a quantum computer. The suite
is designed to work with both IQM Resonance (IQM's quantum cloud service) and
on-prem devices, but can be easily adapted to work with other quantum
computing platforms.

To use the suite with IQM Resonance, you will need to set up an account and
obtain an API token. You can then set the `IQM_TOKEN` environment variable to
your API token. The suite will automatically use this token to authenticate
with IQM Resonance.

```python
import os
os.environ["IQM_TOKEN"] = "your_token"
```

### Using a Jupyter notebook or Python script

You can easily set up one or more benchmarks by defining a configuration
for them. For example, for Randomized, Interleaved and Mirror Benchmarking,
or Quantum Volume:

```python
from iqm.benchmarks.randomized_benchmarking.interleaved_rb.interleaved_rb \
    import InterleavedRBConfiguration
from iqm.benchmarks.randomized_benchmarking.mirror_rb.mirror_rb \
    import MirrorRBConfiguration
from iqm.benchmarks.quantum_volume.quantum_volume \
    import QuantumVolumeConfiguration

EXAMPLE_IRB = InterleavedRBConfiguration(
    qubits_array=[[3,4],[8,9]],
    sequence_lengths=[2**(m+1)-1 for m in range(7)],
    num_circuit_samples=30,
    shots=2**10,
    parallel_execution=True,
    interleaved_gate = "iSwapGate",
    interleaved_gate_params = None,
    simultaneous_fit = ["amplitude", "offset"],
)

EXAMPLE_MRB = MirrorRBConfiguration(
    qubits_array=[[0,1],
                  [0,1,3,4],
                  [0,1,3,4,8,9],
                  [0,1,3,4,8,9,13,14],
                  [0,1,3,4,8,9,13,14,17,18]],
    depths_array=[[2**m for m in range(9)],
                  [2**m for m in range(8)],
                  [2**m for m in range(7)],
                  [2**m for m in range(6)],
                  [2**m for m in range(5)]],
    num_circuit_samples=10,
    num_pauli_samples=5,
    shots=2**8,
    two_qubit_gate_ensemble={"CZGate": 0.7, "iSwapGate": 0.3},
    density_2q_gates=0.25,
)

EXAMPLE_QV = QuantumVolumeConfiguration(
    num_circuits=800,
    shots=2**8,
    num_sigmas=2,
    choose_qubits_routine="custom",
    custom_qubits_array=[[0,1,2,3], [0,1,3,4]],
    max_circuits_per_batch=100,
    rem=True,
    mit_shots=1_000,
)
```

In order to execute them, you must specify a backend:

- First you need to specify the quantum computer name, i.e. "garnet"
- Then you need to specify the base URL of the quantum computer.

Also, you need to reference the benchmark configuration you want to run:

```python
from iqm.benchmarks.randomized_benchmarking.mirror_rb.mirror_rb import *
from iqm.qiskit_iqm import IQMProvider

quantum_computer = "" # the quantum computer name. i.e. "emerald", "garnet", "sirius"
iqm_server_url = "https://resonance.iqm.tech/" # base url

provider = IQMProvider(iqm_server_url, quantum_computer=quantum_computer)
backend = provider.get_backend()

EXAMPLE_EXPERIMENT = MirrorRandomizedBenchmarking(backend, EXAMPLE_MRB)
EXAMPLE_EXPERIMENT.run()
```

Full examples on how to run benchmarks and analyze the results can be found in
the `docs/examples` folder.
