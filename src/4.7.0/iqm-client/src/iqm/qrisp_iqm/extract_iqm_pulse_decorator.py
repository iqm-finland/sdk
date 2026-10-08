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
"""Jasp decorator for converting Qrisp Jasp-traceable functions into IQM quantum circuits."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qrisp.circuit import Clbit, PassManager, convert_to_cz
from qrisp.jasp import make_jaspr

from .iqm_converter import qrisp_to_iqm_converter

if TYPE_CHECKING:
    from iqm.pulse import Circuit
    from iqm.station_control.interface.models import DynamicQuantumArchitecture


def _get_default_pass_manager() -> PassManager:
    """Create the default PassManager with convert_to_cz pass."""
    pm = PassManager()
    pm.add_pass(convert_to_cz())
    return pm


def extract_iqm_pulse(
    dqa: DynamicQuantumArchitecture,
    pass_manager: PassManager | str | None = "default",
) -> Callable[[Callable], Callable[..., Circuit | tuple[Any, ...]]]:
    """Jasp-trace a Qrisp function and compile it to :class:`.Circuit`.

    This is the primary entry point for taking a Qrisp quantum function and
    turning it into an IQM-native circuit without manual transpilation.  The
    decorator:

    1. Traces the function with :func:`~qrisp.jasp.make_jaspr` to capture
       all quantum operations (including :class:`.IQMPulseOperation`
       instances).
    2. Converts the Jasp representation into a Qrisp :class:`~qrisp.QuantumCircuit`.
    3. Optionally applies transpilation passes via a :class:`~qrisp.PassManager`
       (by default, a ``convert_to_cz`` pass is applied).
    4. Converts the transpiled circuit to an :class:`.Circuit` using
       :func:`.qrisp_to_iqm_converter`.

    Args:
        dqa: Determines the physical qubit names and available gate loci.
        pass_manager: Transpilation strategy for the circuit before IQM conversion:

            - ``"default"`` — a :class:`~qrisp.PassManager` containing
              only :func:`~iqm.qrisp_iqm.passes.convert_to_cz`.
            - ``None`` — no transpilation; the circuit is converted as-is
              (useful when it contains only :class:`IQMPulseOperation` gates).
            - A custom :class:`~qrisp.PassManager` — full user control over
              layout, routing, and gate decomposition.

    Returns:
        A decorator that, when applied to a function, returns a wrapper
        whose return type is :class:`.Circuit`,  or ``tuple[..., Circuit]``
        when the original function returns values.

        If the decorated function has **no return value** (only builds a circuit
        as a side effect), the wrapper returns the corresponding IQM Circuit directly.

        If the function **returns measurement results** (e.g., a
        :class:`~qrisp.QuantumVariable` or a list of :class:`~qrisp.circuit.Clbit`),
        the wrapper returns a tuple ``(*processed_returns, iqm_circuit)`` where
        classical bits are replaced by their IQM readout key strings (``"cb_0"``,
        ``"cb_1"``, …).

    Raises:
        ValueError: If *pass_manager* is a string other than ``"default"``.

    Examples:
        First, obtain the device DQA via the IQM Backend:

        .. code-block:: python

            from iqm.qrisp_iqm import IQMBackend

            backend = IQMBackend(
                device_instance="garnet",
                token="YOUR_API_TOKEN",
            )
            dqa = backend.iqm_client.get_dynamic_quantum_architecture()

        **Circuit without return values** (implicit circuit capture):

        .. code-block:: python

            from qrisp import QuantumVariable, h, cx, measure
            from iqm.qrisp_iqm import extract_iqm_pulse

            @extract_iqm_pulse(dqa=dqa)
            def bell_circuit():
                qv = QuantumVariable(2)
                h(qv[0])
                cx(qv[0], qv[1])
                measure(qv)

            iqm_circuit = bell_circuit()          # → iqm.pulse.Circuit
            print(iqm_circuit)

        **Circuit returning a measurement result**:

        .. code-block:: python

            from qrisp import QuantumFloat, h, measure

            @extract_iqm_pulse(dqa=dqa)
            def random_float():
                qf = QuantumFloat(3)
                h(qf)
                res = measure(qf)
                return res

            random_bits, iqm_circuit = random_float()
            print(random_bits)                    # → ("cb_0", "cb_1", "cb_2")
            print(iqm_circuit)

        **Pulse-only circuit (no transpilation needed)**:

        .. code-block:: python

            from iqm.qrisp_iqm import delay

            @extract_iqm_pulse(dqa=dqa, pass_manager=None)
            def pulse_delay():
                qv = QuantumVariable(1)
                delay(qv, duration=100e-9)

            iqm_circuit = pulse_delay()
            # iqm_circuit.instructions contains a single delay operation
            print(iqm_circuit.instructions[0].name)  # → "delay"

    """
    # Handle default pass manager
    if isinstance(pass_manager, str):
        if pass_manager == "default":
            resolved_pass_manager = _get_default_pass_manager()
        else:
            raise ValueError(f"Unknown pass_manager string: {pass_manager}")
    else:
        resolved_pass_manager = pass_manager

    def decorator(func: Callable) -> Callable[..., Circuit | tuple[Any, ...]]:
        def wrapper(*args, **kwargs) -> Circuit | tuple[Any, ...]:
            # 1. Jasp Tracing
            # Create a Jaspr from the function. Captures logic without execution.
            jaspr = make_jaspr(func)(*args, **kwargs)

            # 2. Staticalization
            # Convert Jaspr to Qrisp QuantumCircuit.
            # Returns (*original_returns, qc)
            staticalization_result = jaspr.to_qc(*args, **kwargs)

            # Check return structure
            # If jaspr.outvars has length 1, it means the function returned nothing
            # (only the implicit QuantumCircuit is in staticalization_result)
            has_return_values = len(jaspr.outvars) > 1

            if has_return_values:
                qc = staticalization_result[-1]
                return_values = staticalization_result[:-1]
            else:
                qc = staticalization_result
                return_values = []

            # 3. Apply transpilation passes if a PassManager is provided
            if resolved_pass_manager is not None:
                qc = resolved_pass_manager.run(qc)

            # 4. Conversion to IQM Circuit
            iqm_circuit = qrisp_to_iqm_converter(
                qc,
                dqa=dqa,
                circuit_name=func.__name__,
            )

            if not has_return_values:
                return iqm_circuit

            # 5. Process Return Values (Map Clbits to Readout Keys)
            # The qrisp_to_iqm_converter mapping logic assigns keys like "m{id}" based on Clbit identifier/index.
            # We need to replicate or extract this mapping.
            # In qrisp_to_iqm_converter: clbit_to_key = {cb: cb.identifier ...}

            processed_results: list[Any] = []
            for val in return_values:
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], Clbit):
                    # List of Clbits (e.g. from QuantumVariable)
                    # Convert to tuple of keys
                    new_val = tuple(clbit.identifier for clbit in val)
                elif isinstance(val, Clbit):
                    # Single Clbit
                    new_val = val.identifier
                else:
                    # Pass through classical values
                    new_val = val
                processed_results.append(new_val)

            # Append circuit to results
            processed_results.append(iqm_circuit)
            return tuple(processed_results)

        return wrapper

    return decorator
