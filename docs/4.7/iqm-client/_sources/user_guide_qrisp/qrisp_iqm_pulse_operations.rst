.. _qrisp_iqm_pulse_operations:

Pulse operations & conversion
=============================

Provides a Qrisp frontend for IQM's IQMClient pulse-level compiler. This module does
**not** perform pulse-level compilation itself — it lets you construct pulse-aware
Qrisp circuits, extract them as IQM :class:`~iqm.pulse.Circuit` objects, and then
hand them to IQMClient for compilation into a pulse-level playlist. For more info
on how to modify and build pulse level compilation schedules, please see the
:class:`.IQMClient` documentation.

A typical workflow:

.. code-block:: python

    import os
    from qrisp import QuantumVariable, h, cx, measure
    from iqm.qrisp_iqm import IQMBackend, extract_iqm_pulse, quantum_op_to_qrisp_func
    from iqm.pulse.quantum_ops import QuantumOp

    os.environ["IQM_SERVER_URL"] = "https://resonance.iqm.tech/"
    os.environ["IQM_QUANTUM_COMPUTER"] = "garnet"
    os.environ["IQM_TOKEN"] = "<YOUR_TOKEN>"

    backend = IQMBackend()
    dqa = backend.iqm_client.get_dynamic_quantum_architecture()

    # Define a custom pulse operation (e.g. a delay)
    delay = quantum_op_to_qrisp_func(QuantumOp(name="delay", params={"duration": (float,)}))

    @extract_iqm_pulse(dqa=dqa)
    def my_circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        delay(qv[0], duration=300e-9)   # survives transpilation as a native pulse op
        cx(qv[0], qv[1])
        return measure(qv)

    # Extract as IQM pulse Circuit
    meas_keys, iqm_pulse_qc = my_circuit()

    from iqm.iqm_client import IQMClient
    iqm_client = IQMClient()

    # Compile to a pulse playlist via IQMClient
    compiler = iqm_client.get_standard_compiler()
    playlist, context = compiler.compile([iqm_pulse_qc])


:func:`extract_iqm_pulse` traces a Qrisp function via Jasp and converts it to an
IQM Pulse :class:`~iqm.pulse.Circuit`.  :class:`IQMPulseOperation` lets you embed native
pulse instructions (delays, barriers, custom gates) that survive transpilation unchanged.

Circuit conversion
------------------

.. currentmodule:: iqm.qrisp_iqm.iqm_converter

.. autosummary::
   :signatures: none

   qrisp_to_iqm_converter


Pulse operations
----------------

.. currentmodule:: iqm.qrisp_iqm.pulse_operation

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
     - A Qrisp :class:`~qrisp.Operation` subclass that wraps a native IQM pulse
       :class:`~iqm.pulse.quantum_ops.QuantumOp`. Survives transpilation unchanged,
       enabling pulse-level instructions (delays, barriers, custom gates) to be
       embedded directly in Qrisp circuits.
   * - :func:`~iqm.qrisp_iqm.quantum_op_to_qrisp_func`
     - Convenience function that converts an IQM :class:`~iqm.pulse.quantum_ops.QuantumOp`
       into a Qrisp-callable gate function (usable like ``h``, ``cx``, etc.).
   * - :func:`~iqm.qrisp_iqm.extract_iqm_pulse`
     - Decorator that traces a Qrisp quantum function via Jasp, transpiles the
       resulting circuit, and converts it to an IQM :class:`~iqm.pulse.Circuit`.
       Supports custom :class:`~qrisp.PassManager` pipelines (or ``None`` to skip
       transpilation). Returns measurement keys alongside the compiled circuit.


Custom pulse operations
-----------------------

.. currentmodule:: iqm.qrisp_iqm.custom_pulse_operations


.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.custom_pulse_operations.delay`
     - A pre-built delay operation that inserts idle time into the circuit.
       This is a :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
       wrapper around IQM's native delay gate.
