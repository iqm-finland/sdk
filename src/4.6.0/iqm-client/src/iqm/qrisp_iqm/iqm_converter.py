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
"""Circuit converter for mapping Qrisp QuantumCircuits to IQM Circuits.

This module provides the conversion function that maps Qrisp circuits to IQM format.
For transpilation (layout, routing, gate conversion), use the PassManager from
:func:`.create_iqm_pass_manager`, or :func:`.transpile_to_iqm`.
"""

from __future__ import annotations

import numpy as np
from qrisp import ClControlledOperation, PRXGate, QuantumCircuit, U3Gate
from qrisp.misc.stim_tools import StimNoiseGate

from iqm.pulse import Circuit
from iqm.pulse import CircuitOperation as Op
from iqm.station_control.interface.models import DynamicQuantumArchitecture

from .pulse_operation import IQMPulseOperation


def get_coupling_map(dqa: DynamicQuantumArchitecture) -> list[tuple[int, int]]:
    """Extract the coupling map from DQA.

    Args:
        dqa: Describes the gates and loci available with a given calibration set.

    Returns:
        The coupling map as a list of pairs of ``dqa.qubits`` indices that have
        the CZ gate available.

    Raises:
        ValueError: If the device uses computational resonators, which are not
            yet supported by the Qrisp-IQM connector.

    """
    if (cz_info := dqa.gates.get("cz")) is None:
        return []

    qubits = dqa.qubits

    # Detect computational resonator components in CZ loci.
    # These devices use a central resonator for two-qubit gates and are not
    # yet supported by the plasma-sabre transpilation pipeline.
    unsupported = sorted({c for locus in cz_info.loci for c in locus if c not in qubits})
    if unsupported:
        raise ValueError(
            "Computational resonator devices are not yet supported by the "
            "Qrisp-IQM connector. "
            f"The CZ gate locus/loci involve component(s) {unsupported} "
            "which are computational resonators, not standard qubits. "
            "Transpilation (layout + routing) for this device class is not "
            "implemented."
        )

    return [(qubits.index(locus[0]), qubits.index(locus[1])) for locus in cz_info.loci]


def qrisp_to_iqm_converter(
    qc: QuantumCircuit,
    dqa: DynamicQuantumArchitecture,
    circuit_name: str = "Qrisp_converted",
) -> Circuit:
    """Converts a Qrisp QuantumCircuit to an IQM :class:`.Circuit`.

    The circuit should already be transpiled to native gates (CZ, PRX) and have
    a valid qubit layout.  For a one-step transpile-and-convert workflow, use
    :func:`~iqm.qrisp_iqm.passes.transpile_to_iqm` followed by this function.

    Args:
        qc: The Qrisp circuit to convert. Must already be transpiled to
            IQM-native gates (CZ, PRX).
        dqa: Determines the physical qubit names and available gate loci.
            ``dqa`` maps Qrisp's ``qubits[i]`` to the i-th physical qubit
            name (``QB1``, ``QB2``, …) in the IQM device.
        circuit_name: Name assigned to the output IQM circuit.

    Returns:
        ``qc`` converted to IQM circuit format.

    Raises:
        ValueError: If the circuit has more qubits than the DQA.
        Exception: If an unknown gate type is encountered (circuit was not properly transpiled).

    Examples:
        .. code-block:: python

            from qrisp import QuantumCircuit
            from iqm.qrisp_iqm import transpile_to_iqm, qrisp_to_iqm_converter

            # 1. Create and transpile a Qrisp circuit
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 2)
            qc.measure(range(3))

            coupling_map = [(0, 1), (1, 2), (2, 3)]
            transpiled = transpile_to_iqm(qc, coupling_map)

            # 2. Obtain the IQM device DQA
            from iqm.iqm_client import IQMClient
            client = IQMClient(
                iqm_server_url="https://resonance.iqm.tech/",
                quantum_computer="garnet",
                token="YOUR_API_TOKEN",
            )

            dqa = client.get_dynamic_quantum_architecture()

            # 3. Convert the transpiled circuit to IQM format
            iqm_circuit = qrisp_to_iqm_converter(transpiled, dqa)
            print(iqm_circuit)

            # Yields:

            # Circuit(
            #   name='Qrisp_converted',
            #   instructions=(
            #     CircuitOperation(name='prx', locus=('QB1',),
            #       args={'angle': 1.5707963267948966, 'phase': 6.283185307179586}, implementation=None),
            #     CircuitOperation(name='prx', locus=('QB2',),
            #       args={'angle': 1.5707963267948966, 'phase': 6.283185307179586}, implementation=None),
            #     CircuitOperation(name='cz', locus=('QB1', 'QB2'), args={}, implementation=None),
            #     CircuitOperation(name='measure', locus=('QB1',), args={'key': 'cb_0'}, implementation=None),
            #     CircuitOperation(name='measure', locus=('QB3',), args={'key': 'cb_0'}, implementation=None),
            #     CircuitOperation(name='prx', locus=('QB2',),
            #       args={'angle': 1.5707963267948966, 'phase': 3.141592653589793}, implementation=None),
            #     CircuitOperation(name='measure', locus=('QB2',), args={'key': 'cb_0'}, implementation=None)),
            #   metadata=None,
            # )


    The printed IQM circuit shows each gate as a ``CircuitOperation`` with
    its name, locus (target qubits), and arguments.

    """
    qubit_list = dqa.qubits

    # Validate qubit count
    if len(qc.qubits) > len(qubit_list):
        raise ValueError(f"Circuit requires {len(qc.qubits)} qubits, but DQA only has {len(qubit_list)}: {qubit_list}")

    # Map Qrisp qubits to IQM physical qubits
    qb_to_locus = dict(zip(qc.qubits, qubit_list))

    # Map classical bits for measurement keys.
    # Zero-fill the numeric suffix so that plain sorted() produces
    # the correct order (cb_000 < cb_001 < … < cb_010 < …).
    n_clbits = len(qc.clbits)
    zfill_width = len(str(max(1, n_clbits)))
    clbit_to_key = {clbit: "cb_" + str(i).zfill(zfill_width) for i, clbit in enumerate(qc.clbits)}

    iqm_ops = []

    last_measurement_dict = {}

    # Convert each instruction
    for instr in qc.data:
        op_name = instr.op.name
        iqm_op = None
        locus = tuple(qb_to_locus[qb] for qb in instr.qubits)
        meas_keys = tuple(clbit_to_key[cb] for cb in instr.clbits)

        # Skip allocation/deallocation markers
        if op_name in ["qb_alloc", "qb_dealloc", "parity"] or isinstance(instr.op, StimNoiseGate):
            continue

        # --- IQM Pulse Operation ---
        if isinstance(instr.op, IQMPulseOperation):
            param_dict = instr.op.param_dict.copy()
            if instr.op.quantum_op.name in ("measure", "measure_fidelity"):
                # measurement operations need keys that uniquely identify the result
                # each clbit can only be used for one measurement
                # hence we can construct a unique key from clbit names
                param_dict["key"] = ",".join(meas_keys)
            iqm_op = Op(
                name=instr.op.quantum_op.name,
                locus=locus,
                args=param_dict,
            )

        # --- CZ Gate ---
        elif op_name == "cz":
            iqm_op = Op(
                name="cz",
                locus=locus,
                args={},
            )

        # --- PRX Gate (Phased RX) ---
        elif isinstance(instr.op, PRXGate):
            iqm_op = Op(
                name="prx",
                locus=locus,
                args={
                    "angle": float(instr.op.alpha % (2 * np.pi)),
                    "phase": float(instr.op.beta % (2 * np.pi)),
                },
            )

        # --- U3 Gate (fallback; should be converted to PRX by the pass manager) ---
        elif isinstance(instr.op, U3Gate) and not isinstance(instr.op, PRXGate):
            iqm_op = Op(
                name="u",
                locus=locus,
                args={
                    "theta": float(instr.op.theta),
                    "phi": float(instr.op.phi),
                    "lam": float(instr.op.lam),
                },
            )

        # --- Measurement ---
        elif op_name == "measure":
            iqm_op = Op(
                name="measure",
                locus=locus,
                args={"key": meas_keys[0]},
            )

            last_measurement_dict[meas_keys[0]] = locus

        # --- Barrier ---
        elif op_name == "barrier":
            iqm_op = Op(
                name="barrier",
                locus=locus,
                args={},
            )

        # --- Reset ---
        elif op_name == "reset":
            iqm_op = Op(
                name="reset",
                locus=locus,
                args={},
            )

        # --- Classically Controlled Operations ---
        elif isinstance(instr.op, ClControlledOperation):
            base_op = instr.op.base_op

            if not isinstance(base_op, PRXGate):
                raise Exception(f"Don't know how to translate classically controlled operation {base_op.name} to IQM")

            iqm_op = Op(
                name="cc_prx",
                locus=locus,
                args={
                    "angle": float(base_op.alpha % (2 * np.pi)),
                    "phase": float(base_op.beta % (2 * np.pi)),
                    "feedback_key": meas_keys[0],
                    "feedback_qubit": last_measurement_dict[meas_keys[0]],
                },
            )

        # --- Unhandled Gates ---
        else:
            raise Exception(
                f"Don't know how to convert operation '{instr.op.name}' to IQM. "
                f"Make sure to transpile the circuit to native gates (CZ, PRX) first."
            )

        iqm_ops.append(iqm_op)

    # Construct and return the IQM Circuit
    return Circuit(circuit_name, tuple(iqm_ops))
