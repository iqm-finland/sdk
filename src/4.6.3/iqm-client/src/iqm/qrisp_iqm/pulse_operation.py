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
"""Using IQM Pulse quantum operations in Qrisp quantum circuits."""

from __future__ import annotations

from typing import Any, Protocol, Self

import numpy as np
from qrisp import Operation, append_operation
from qrisp.jasp import check_for_tracing_mode
from sympy import Expr, symbols

from iqm.pulse.quantum_ops import QuantumOp

GREEK_LETTERS = symbols(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
    "xi omicron pi rho sigma tau upsilon phi chi psi omega"
)
"""SymPy greek letters."""

JAX_COMPATIBLE_TYPES = (int, float, bool)
"""Types that JAX can trace and that are valid for the dynamic parameter pipeline."""


class IQMPulseOperation(Operation):
    """Qrisp Operation that represents an IQM Pulse quantum operation.

    This class allows users to insert specific IQM Pulse quantum operations (like barriers, delays,
    or custom gate implementations) directly into a Qrisp QuantumCircuit. These operations
    are treated as opaque boxes by the Qrisp transpiler, ensuring they survive the
    compilation process and reach the backend unchanged.

    Mechanism for transpilation survival:
    Because IQMPulseOperation inherits from the base ``Operation`` class but does not provide
    a ``definition`` (decomposition into standard gates) or built-in QASM conversion,
    Qrisp's transpilation passes generally ignore it. It passes through optimizations
    untouched, preserving its position in the circuit relative to other gates.

    Parameter handling:
    Parameters are passed as a dictionary. Qrisp stores the values in :attr:`params`
    (required for any parameter optimization or bind logic). This class additionally
    stores parameter names in :attr:`param_names`. The IQM Pulse converter uses these names
    to reconstruct the dictionary argument required by :meth:`QuantumOp.__init__`.

    Args:
        quantum_op:
            IQM Pulse quantum operation definition. This object is expected to contain the
            concrete parameter values in its ``params`` attribute and the concrete qubit
            count in its ``arity`` attribute.
        param_dict:
            Dictionary of parameter name-value pairs. Must match the QuantumOp's params.
        num_clbits:
            Number of classical bits the operation uses.

    Examples:
        Create a delay operation and embed it in a Qrisp circuit:

        .. code-block:: python

            from iqm.pulse.quantum_ops import QuantumOp
            from iqm.pulse.gates.delay import Delay
            from iqm.qrisp_iqm import IQMPulseOperation
            from qrisp import QuantumCircuit

            # Build the delay QuantumOp from the IQM SDK primitives
            delay_quantum_op = QuantumOp(
                name="delay",
                arity=1,
                params={"duration": (float,)},
                implementations={"wait": Delay},
                symmetric=True,
            )

            # Wrap it in a Qrisp-compatible operation with a 100 ns duration
            pulse_op = IQMPulseOperation(
                delay_quantum_op,
                param_dict={"duration": 100e-9},
            )

            # Insert into a Qrisp circuit — the delay survives transpilation
            qc = QuantumCircuit(2)
            qc.cz(0, 1)
            qc.append(pulse_op, [qc.qubits[0]])

    """

    def __init__(
        self,
        quantum_op: QuantumOp,
        param_dict: dict[str, Any] | None = None,
        num_clbits: int = 0,
    ) -> None:
        if param_dict is None:
            param_dict = {}

        # Verify that the keys of the supplied dictionary match the keys of the QuantumOp
        if set(param_dict) != set(quantum_op.params):
            raise ValueError(
                f"Tried to initialize IQMPulseOperation {quantum_op.name} with "
                f"incompatible arguments (required: {list(quantum_op.params)}, "
                f"given: {list(param_dict)})"
            )

        # Verify that the values have the correct type
        # (skip for SymPy symbolic placeholders used during Jasp tracing)
        for k, v in param_dict.items():
            if isinstance(v, Expr):
                continue
            if not isinstance(v, quantum_op.params[k]):
                raise TypeError(
                    f"Tried to initialize IQMPulseOperation {quantum_op.name} parameter "
                    f"{k} with type {type(v)} (allowed: {quantum_op.params[k]})"
                )

        self.param_dict = param_dict.copy()
        self.param_names = list(param_dict)
        self.quantum_op = quantum_op

        # Separate dynamic (JAX-traceable) parameters from static ones.
        # Only SymPy Expr (symbolic placeholders) or JAX-compatible concrete
        # types are passed through Operation.params so they participate in
        # bind_parameters / abstract_params resolution.  Non-JAX types
        # (e.g. str) are kept only in param_dict as static values.
        self.dynamic_param_names = [k for k, v in param_dict.items() if isinstance(v, (Expr, JAX_COMPATIBLE_TYPES))]
        self.static_param_names = [k for k in self.param_names if k not in self.dynamic_param_names]

        dynamic_values = [param_dict[k] for k in self.dynamic_param_names]

        # Determine num_qubits from the arity of the passed instance
        num_qubits = quantum_op.arity

        # Initialize the base Operation, passing only dynamic param values
        # so that symbolic parameters (SymPy greek letters) are correctly tracked
        # as abstract_params and resolved via bind_parameters during staticalization.
        Operation.__init__(
            self,
            name="iqm." + quantum_op.name,
            num_qubits=num_qubits,
            num_clbits=num_clbits,
            params=dynamic_values,
        )

        self.permeability = dict.fromkeys(range(num_qubits), "neutral")
        self.is_qfree = False

        if quantum_op.unitary is None:
            self.unitary = None
        else:
            self.unitary = quantum_op.unitary()  # FIXME what if the quantum_op has args? the unitary depends on them.

    def bind_parameters(self, subs_dict: dict[Any, Any]) -> Self:
        """Bind abstract (symbolic) parameters to concrete values.

        Extends the base Operation.bind_parameters to also update
        ``param_dict`` with the resolved values.  Only dynamic
        (JAX-compatible) parameters are substituted; static parameters
        are preserved unchanged.
        """
        res = super().bind_parameters(subs_dict)
        # res.params now contains the resolved dynamic values.
        # Reconstruct param_dict: start with static params, then overlay dynamic.
        res.param_dict = {k: self.param_dict[k] for k in res.static_param_names}
        res.param_dict.update(dict(zip(res.dynamic_param_names, res.params)))
        return res

    def get_unitary(self, decimals: int = -1) -> np.ndarray:
        """Returns the unitary matrix representation of the operation.

        Raises:
            ValueError: If the operation has no defined unitary.

        """
        if self.unitary is None:
            raise ValueError(f"Don't know unitary of IQM Pulse operation {self.quantum_op.name}")
        return self.unitary.copy()


class QrispFunction(Protocol):
    """Function for appending an IQM Pulse quantum operation to a Qrisp quantum circuit."""

    def __call__(self, *qubits: int, **params) -> None:
        """Append the quantum operation acting on the given ``qubits``, with the given ``params``."""


def quantum_op_to_qrisp_func(
    quantum_op: QuantumOp,
) -> QrispFunction:
    """Create a function for appending IQM Pulse quantum operations to a Qrisp quantum circuit.

    Returns a callable that can be used to append an IQMPulseOperation for the specified
    QuantumOp to a Qrisp circuit. The returned function accepts locus qubits as positional
    arguments and operation parameters as keyword arguments.

    In Jasp tracing mode, dynamic (JAX-traced) parameter values are supported:
    SymPy placeholder symbols are used in the operation definition, while the
    actual traced values are forwarded via ``param_tracers``.

    Args:
        quantum_op: IQM Pulse quantum operation definition.

    Returns:
        Function for adding the corresponding operation to a Qrisp circuit.

    Example:
        Define a custom ``delay`` pulse operation, convert it into a Qrisp gate
        function, and use it inside a Jasp-traced circuit that is then compiled to
        an IQM pulse circuit:

        .. code-block:: python

            from qrisp import QuantumVariable, h, cx, measure
            from iqm.pulse.quantum_ops import QuantumOp
            from iqm.qrisp_iqm import extract_iqm_pulse, quantum_op_to_qrisp_func

            delay_quantum_op = QuantumOp(name="delay", params={"duration": (float,)})
            delay = quantum_op_to_qrisp_func(delay_quantum_op)

            @extract_iqm_pulse(dqa=...)
            def my_circuit():
                qv = QuantumVariable(2)
                h(qv[0])
                delay(qv[0], duration=100e-9)
                cx(qv[0], qv[1])
                return measure(qv)

            meas_keys, iqm_circuit = my_circuit()

    """
    # Determine which params have at least one JAX-compatible accepted type.
    params_with_jax_types = frozenset(
        name for name, types in quantum_op.params.items() if any(t in JAX_COMPATIBLE_TYPES for t in types)
    )

    def return_function(*qubits, **kwargs) -> None:
        if check_for_tracing_mode():
            # In tracing mode: use symbolic placeholders for JAX-compatible params
            # and pass the actual JAX tracers separately.  Non-JAX types (str, etc.)
            # are embedded directly as concrete values in the operation.
            symbolic_dict = {}
            param_tracers = []
            sym_idx = 0
            for name, value in kwargs.items():
                if name in params_with_jax_types:
                    symbolic_dict[name] = GREEK_LETTERS[sym_idx]
                    param_tracers.append(value)
                    sym_idx += 1
                else:
                    # Static param — embed concrete value directly
                    symbolic_dict[name] = value

            pulse_op = IQMPulseOperation(quantum_op, param_dict=symbolic_dict)
            append_operation(operation=pulse_op, qubits=list(qubits), param_tracers=param_tracers)
        else:
            pulse_op = IQMPulseOperation(quantum_op, param_dict=kwargs)
            append_operation(operation=pulse_op, qubits=list(qubits))

    return return_function
