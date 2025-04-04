# Copyright 2024 IQM
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
"""Utilities for working with a QIR module."""

from __future__ import annotations

import logging
import warnings

from pyqir import (
    Call,
    ConstantAsMetadata,
    Context,
    FloatConstant,
    IntConstant,
    Metadata,
    Module,
    is_entry_point,
    is_qubit_type,
    is_result_type,
    qubit_id,
    required_num_qubits,
    required_num_results,
    result_id,
)

from iqm.cpc.compiler.compiler import Compiler
from iqm.cpc.interface.compiler import Circuit as CPC_Circuit
from iqm.pulla.pulla import Pulla
from iqm.pulse.builder import CircuitOperation

qir_logger = logging.getLogger(__name__)


def _removeprefix(self: str, prefix: str) -> str:
    if self.startswith(prefix):
        return self[len(prefix) :]
    return self[:]


def _removesuffix(self: str, suffix: str) -> str:
    if suffix and self.endswith(suffix):
        return self[: -len(suffix)]
    return self[:]


# Convert a QIS operation to a simple string representation
def _gate_inst_to_str(inst: Call) -> CircuitOperation | None:
    """Convert a QIS instruction to a Circuit instruction.

    Returns:
        CircuitOperation: The Circuit operation corresponding to the QIS instruction.
        None: If the instruction is not a quantum instruction.

    Raises:
        ValueError: If quantum instruction is not supported.
        ValueError: If quantum instruction has an invalid number of arguments.

    """
    qir_logger.debug("Processing instruction: %s", inst)
    if not inst.callee.name.startswith("__quantum__qis__"):
        qir_logger.debug("Instruction is not a quantum instruction")
        return None

    operation = _removesuffix(_removeprefix(inst.callee.name, "__quantum__qis__"), "__body")
    if operation in ["phased_rx", "prx", "r"]:
        operation = "prx"
    if operation in ["mz", "measurement", "measure"]:
        operation = "measure"
    qir_logger.debug("Processing operation: %s", operation)

    # Dictionary mapping operations to their argument handling functions
    operation_handlers = {
        "prx": lambda args: {
            "name": "prx",
            "locus": [str(qubit_id(arg)) for arg in _find_args_by_type(args, is_qubit_type)],
            "args": {
                "angle": _parse_double(_find_double_args(args)[0]),
                "phase": _parse_double(_find_double_args(args)[1]),
            },
        },
        # Mimic the Qiskit naming convention for the measurement operation
        # for compatibility with the Pulla utils
        "measure": lambda args: {
            "name": "measure",
            "locus": [str(qubit_id(arg)) for arg in _find_args_by_type(args, is_qubit_type)],
            "args": {"key": f"m_1_{result_id(_find_arg_by_type(args, is_result_type))}_0"},
        },
        "cz": lambda args: {
            "name": "cz",
            "locus": [str(qubit_id(arg)) for arg in _find_args_by_type(args, is_qubit_type)],
            "args": {},
        },
    }

    if operation not in operation_handlers:
        qir_logger.error("Unsupported operation: %s", operation)
        raise ValueError(f"Unsupported operation: {operation}")

    try:
        qir_logger.debug("Processing %s with args: %s", operation, inst.args)
        params = operation_handlers[operation](inst.args)
        return CircuitOperation(**params)
    except (IndexError, ValueError, AttributeError) as e:
        qir_logger.error("Error processing operation %s: %s", operation, e)
        raise ValueError(f"Error processing operation {operation}: {e}") from e


def _find_arg_by_type(args, type_check_func):
    for arg in args:
        if type_check_func(arg.type):
            return arg
    raise ValueError(f"Expected exactly one argument of type {type_check_func}")


def _find_args_by_type(args, type_check_func):
    matches = []
    for arg in args:
        if type_check_func(arg.type):
            matches.append(arg)
    return matches


def _find_double_args(args):
    """Return non-qubit, non-result arguments which should be doubles."""
    return [arg for arg in args if arg.type.is_double]


def _parse_double(value: str) -> float:
    """Helper function to parse double values from hex or decimal strings."""
    if isinstance(value, FloatConstant):
        return value.value
    if isinstance(value, IntConstant):
        return float(value.value)
    raise ValueError(f"Invalid double value: {value}")


def qir_to_pulla(  # noqa: PLR0915, PLR0912
    pulla: Pulla, qir: str | bytes, qubit_mapping: dict[int:str] | None = None
) -> tuple[list[CPC_Circuit], Compiler]:
    """Convert a QIR module to a CPC circuit.

    Args:
        pulla: The Pulla instance to get compiler from.
        qir: The QIR source or bitcode to convert to a circuit.
        qubit_mapping: A dictionary mapping QIR qubit indexes to physical qubit names,
                       None will assume opaque pointers match physical names.

    Returns:
        str: The QIR program name,
        tuple[CircuitOperation, ...]: The circuit operations extracted from the QIR code.

    Raises:
        ValueError: If the QIR program has more than one basic block.

    """
    if isinstance(qir, bytes):
        qir_logger.debug("Loading QIR from bitcode")
        module = Module.from_bitcode(Context(), qir)
    else:
        qir_logger.debug("Loading QIR from source")
        module = Module.from_ir(Context(), qir)

    qir_logger.debug("QIR module IR: %s", str(module))

    _required_num_qubits: int | None = 0
    _required_num_results: int | None = 0
    _qir_profiles: str | None = None
    for func in module.functions:
        qir_logger.debug("Function: %s, is entry point: %s", func, is_entry_point(func))
        if is_entry_point(func):
            for attr in func.attributes.func:
                if attr.string_kind == "qir_profiles":
                    _qir_profiles = attr.string_value
                    break
            _required_num_qubits = required_num_qubits(func)
            _required_num_results = required_num_results(func)
            qir_logger.info("Required number of qubits: %s", _required_num_qubits)
            qir_logger.info("Required number of results: %s", _required_num_results)

    if _qir_profiles not in ["base_profile", "basic_qir"]:
        warnings.warn(f"{_qir_profiles} may not be a supported QIR profile")

    if not _required_num_qubits or _required_num_qubits == 0:
        err_msg = "QIR program must specify the number of qubits required"
        qir_logger.error(err_msg)
        raise ValueError(err_msg)

    if not _required_num_results or _required_num_results == 0:
        err_msg = "QIR program must specify the number of results required"
        qir_logger.error(err_msg)
        raise ValueError(err_msg)

    qir_major_version_int = 0
    qir_major_version: Metadata | None = module.get_flag("qir_major_version")
    if isinstance(qir_major_version, ConstantAsMetadata) and isinstance(qir_major_version.value, IntConstant):
        qir_major_version_int = qir_major_version.value.value

    qir_minor_version_int = 0
    qir_minor_version: Metadata | None = module.get_flag("qir_minor_version")
    if isinstance(qir_minor_version, ConstantAsMetadata) and isinstance(qir_minor_version.value, IntConstant):
        qir_minor_version_int = qir_minor_version.value.value

    qir_logger.info("QIR version: %s.%s", qir_major_version_int, qir_minor_version_int)
    if qir_major_version_int != 1 or qir_minor_version_int != 0:
        err_msg = f"Unsupported QIR version {qir_major_version_int}.{qir_minor_version_int}"
        qir_logger.error(err_msg)
        raise ValueError(err_msg)

    entry_point = next(filter(is_entry_point, module.functions))
    calls: list[Call] = []

    if len(entry_point.basic_blocks) > 1:
        err_msg = "QIR program must have a single basic block"
        qir_logger.error(err_msg)
        raise ValueError(err_msg)

    for block in entry_point.basic_blocks:
        for inst in block.instructions:
            qir_logger.debug("Processing instruction: %s", inst)
            if isinstance(inst, Call):
                calls.append(inst)

    # Convert the calls into CircuitOperation representations
    converted = [_gate_inst_to_str(call) for call in calls]
    circuit_instructions = tuple(x for x in converted if x is not None)
    name = module.source_filename if module.source_filename else "QIR Program"
    qir_logger.debug("QIR program name: %s", name)
    circuits = [CPC_Circuit(name=name, instructions=circuit_instructions)]
    qir_logger.debug("Converted circuit: %s", circuits)

    # Create a compiler containing all the required station information
    compiler = pulla.get_standard_compiler()

    if qubit_mapping:
        # QIR programs reference to qubits as opaque pointer indexes,
        # however, for example qiskit is using logical names for qubits,
        # so we need to map these indexes to physical qubit names
        compiler.component_mapping = {f"{i}": qubit_mapping[i] for i in range(_required_num_qubits)}
    else:
        # QIR programs reference to qubits as opaque pointer indexes,
        # we expect these indexes to match physical qubit names,
        # transform QIR instructions to Pulla circuit instructions accordingly,
        # so we need mapping from 0, 1, 2, ... to QB1, QB2, QB3, ... for all the components
        compiler.component_mapping = {f"{i}": f"QB{i + 1}" for i in range(_required_num_qubits)}

    return circuits, compiler
