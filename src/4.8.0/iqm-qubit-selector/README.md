# IQM Qubit Selector

Picks the best set of qubits on an IQM QPU to run a given circuit on.

Gate and readout fidelities vary from qubit to qubit and from coupler to
coupler, so where a circuit runs matters. IQM Qubit Selector enumerates the
placements the QPU's couplings allow and ranks them with the latest
calibration metrics, without transpiling the circuit for any candidate.

## Quick start

```python
from math import pi

from iqm.iqm_client import IQMClient
from iqm.pulse import Circuit
from iqm.pulse.circuit_operations import CircuitOperationList
from iqm.qubit_selector.core import select_layout

client = IQMClient.from_url("https://resonance.iqm.tech/", quantum_computer="garnet")

# A three-qubit GHZ circuit on placeholder qubits QB1..QB3
builder = CircuitOperationList(num_qubits=3)
builder.prx(pi / 2, 0.0, 0)
for i in range(1, 3):
    builder.prx(pi / 2, 0.0, i)
    builder.cz(i - 1, i)
circuit = Circuit(name="ghz", instructions=tuple(builder))

# The architecture and its calibration metrics are fetched from the client
for layout, error in select_layout(circuit, client=client, num_layouts=3):
    print(layout, f"{error:.2%}")
```

Set the `IQM_TOKEN` environment variable to your API token before connecting.
To place several circuits, fetch the architecture and metrics once and pass
them instead of the client, as `select_layout(circuit, arch, metrics)`. This
also works without a client, e.g. with metrics measured offline.

## Features

- **SDK-agnostic.** The core, `iqm.qubit_selector.core`, works on an
  `iqm.pulse.Circuit`, so circuits from IQM Pulse, Qiskit, Cirq and Qrisp
  are all placed with the same call.
- **Calibration-aware.** Layouts are ranked by an error budget built from
  two-qubit, single-qubit and, optionally, readout fidelities.
  `score_layouts` returns the breakdown for every candidate.
- **Crystal and Star QPUs.** On Star QPUs the resonator is handled
  automatically; no extra arguments are needed.
- **Extensible.** The coupling graph, enumerators and cost model can be used
  on their own, or replaced with your own.

Version 2.0 replaces the Qiskit-based `iqm.qubit_selector.qubit_selector` and
`iqm.qubit_selector.qiskit_utils` modules with `iqm.qubit_selector.core`. The
old modules are removed and there is no compatibility layer; see the changelog
for how to port existing code.

## Documentation

The user guide, API reference and example notebooks are at
[docs.iqm.tech/iqm-qubit-selector](https://docs.iqm.tech/iqm-qubit-selector/).
