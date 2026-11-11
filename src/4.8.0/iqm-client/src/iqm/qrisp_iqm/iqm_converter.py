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

from collections.abc import Iterable, Sequence

import numpy as np
from qrisp import Clbit, ClControlledOperation, Instruction, PRXGate, QuantumCircuit, Qubit, U3Gate
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


def _apply_condition(
    cc_op: ClControlledOperation,
    clbits: Sequence[Clbit],
    iqm_instructions: Iterable[Op],
    clbit_to_measure: dict[Clbit, tuple[Op, str]],
) -> None:
    """Apply a classical condition to circuit instructions.

    Modifies ``iqm_instructions`` and the values of ``clbit_to_measure`` in place.

    Args:
        cc_op: Operation containing the classical condition.
        clbits: Classical bits ``cc_op`` is conditioned on.
        iqm_instructions: Instructions to apply the condition to.
        clbit_to_measure: Maps classical bits to the pair
            (latest "measure" instruction, measured qubit name)
            to store its result in that classical bit.

    """
    # check that the condition is supported
    if cc_op.num_control != 1:
        raise ValueError(f"{cc_op} is conditioned on {cc_op.num_control} classical bits, only one is supported.")
    if cc_op.ctrl_state != "1":
        raise ValueError(f"{cc_op} is conditioned on bitstring {cc_op.ctrl_state}, only 1 is supported.")
    clbit = clbits[0]

    # Set up feedback routing.
    # The latest "measure" instruction to write to that classical bit is modified, it is
    # given an explicit feedback_key equal to its measurement key.
    # The same feedback_key is given to the controlled instruction, along with the feedback qubit.
    if (pair := clbit_to_measure.get(clbit)) is None:
        raise ValueError(f"{cc_op} conditioned on {clbit}, which does not contain a measurement result yet.")
    measure_inst, feedback_qubit = pair
    feedback_key = measure_inst.args["key"]
    measure_inst.args["feedback_key"] = feedback_key  # this measure is used to provide feedback

    for inst in iqm_instructions:
        # TODO we do not check anywhere if cc_prx is available for this locus!
        if inst.name != "prx":
            raise ValueError(f"This backend does not support conditional {inst.name} gates")
        if inst.locus[0] != feedback_qubit:
            raise ValueError(
                "Currently in qrisp-iqm qubits can only take feedback from themselves, "
                f"{feedback_qubit} -> {inst.locus[0]} attempted."
            )
        inst.name = "cc_prx"
        inst.args["feedback_key"] = feedback_key
        inst.args["feedback_qubit"] = feedback_qubit


def _serialize_instructions(
    instructions: Iterable[Instruction],
    qubit_to_name: dict[Qubit, str],
    clbit_to_key: dict[Clbit, str],
    *,
    clbit_to_measure: dict[Clbit, tuple[Op, str]],
) -> list[Op]:
    """Serialize Qrisp instructions into the IQM data transfer format.

    Args:
        instructions: Qrisp circuit instructions to serialize.
        qubit_to_name: Mapping from Qrisp qubits to the corresponding physical qubit names.
        clbit_to_key: Mapping from Qrisp classical bits to the corresponding measurement keys.
        clbit_to_measure: Maps classical bits to the pair
            (latest "measure" instruction, measured qubit name)
            to store its result in that classical bit, or None if it _seria

    Returns:
        IQM instructions representing the circuit.

    Raises:
        ValueError: circuit contains an unsupported instruction or is not transpiled in general

    """
    # keep track of which qubit was last measured into each Clbit, for the Qrisp c_if construct
    if clbit_to_measure is None:
        clbit_to_measure = {}

    # convert each instruction
    iqm_ops = []
    for instr in instructions:
        op_name = instr.op.name
        locus = tuple(qubit_to_name[qb] for qb in instr.qubits)
        clbit_keys = tuple(clbit_to_key[cb] for cb in instr.clbits)

        # Skip allocation/deallocation markers
        if op_name in ["qb_alloc", "qb_dealloc", "parity"] or isinstance(instr.op, StimNoiseGate):
            continue

        if isinstance(instr.op, IQMPulseOperation):
            iqm_op = Op(
                name=instr.op.quantum_op.name,
                locus=locus,
                args=instr.op.param_dict.copy(),
            )
            if iqm_op.name in ("measure", "measure_fidelity"):
                # measurement operations need keys that uniquely identify the result
                # each clbit can only be used for one measurement
                # hence we can construct a unique key from clbit names
                meas_key = ",".join(clbit_keys)
                iqm_op.args["key"] = meas_key
                clbit_to_measure.update(
                    {clbit: (iqm_op, qubit_name) for clbit, qubit_name in zip(instr.clbits, locus, strict=True)}
                )

        elif op_name == "cz":
            iqm_op = Op(
                name="cz",
                locus=locus,
                args={},
            )

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

        elif op_name == "measure":
            # only 1-qubit loci
            iqm_op = Op(
                name="measure",
                locus=locus,
                args={"key": clbit_keys[0]},
            )
            clbit_to_measure[instr.clbits[0]] = (iqm_op, locus[0])

        elif op_name == "barrier":
            iqm_op = Op(
                name="barrier",
                locus=locus,
                args={},
            )

        elif op_name == "reset":
            iqm_op = Op(
                name="reset",
                locus=locus,
                args={},
            )

        elif isinstance(instr.op, ClControlledOperation):
            base_op = instr.op.base_op
            # a bit extensive since currently c_if only applies to a single op, but...
            conditional_ops = _serialize_instructions(
                [Instruction(base_op, instr.qubits)],
                qubit_to_name,
                clbit_to_key,
                clbit_to_measure=clbit_to_measure,
            )
            _apply_condition(instr.op, instr.clbits, conditional_ops, clbit_to_measure)
            iqm_ops.extend(conditional_ops)
            continue  # skip the rest of the loop

        else:
            raise ValueError(
                f"Don't know how to convert operation '{instr.op.name}' to IQM native instruction. "
                f"Make sure to transpile the circuit to native gates (CZ, PRX) first."
            )

        iqm_ops.append(iqm_op)
    return iqm_ops


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
            client = IQMClient.from_url(
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

    # Map Qrisp classical bits to measurement keys.
    # Zero-fill the numeric suffix so that plain sorted() produces
    # the correct order (cb_000 < cb_001 < … < cb_010 < …).
    n_clbits = len(qc.clbits)
    zfill_width = len(str(max(1, n_clbits)))
    iqm_ops = _serialize_instructions(
        qc.data,
        qubit_to_name=dict(zip(qc.qubits, qubit_list)),  # Map Qrisp qubits to IQM physical qubit names
        clbit_to_key={clbit: "cb_" + str(i).zfill(zfill_width) for i, clbit in enumerate(qc.clbits)},
        clbit_to_measure={},
    )
    # Construct and return the IQM Circuit
    return Circuit(circuit_name, tuple(iqm_ops))
