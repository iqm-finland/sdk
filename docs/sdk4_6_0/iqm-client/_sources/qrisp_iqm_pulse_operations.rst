.. _qrisp_iqm_pulse_operations:

Pulse Operations & Conversion
==============================

Provides a Qrisp frontend for IQM's Pulla pulse-level compiler.  This module does
**not** perform pulse-level compilation itself — it lets you construct pulse-aware
Qrisp circuits, extract them as IQM Pulse :class:`~iqm.pulse.Circuit` objects, and then
hand them to Pulla for compilation into a pulse schedule (playlist). For more info
on how to modify and build pulse level compilation schedules, please visit the Pulla
documentation.

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

    from iqm.pulla.pulla import Pulla
    p = Pulla()

    # Compile to a pulse playlist via Pulla
    compiler = p.get_standard_compiler()
    playlist, context = compiler.compile([iqm_pulse_qc])


:func:`extract_iqm_pulse` traces a Qrisp function via Jasp and converts it to an
IQM Pulse :class:`~iqm.pulse.Circuit`.  :class:`IQMPulseOperation` lets you embed native
pulse instructions (delays, barriers, custom gates) that survive transpilation unchanged.

Circuit Conversion
------------------

.. currentmodule:: iqm.qrisp_iqm

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-short-name.rst
   :nosignatures:

   qrisp_to_iqm_converter


Pulse Operations
----------------

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-class-template.rst
   :nosignatures:

   pulse_operation.IQMPulseOperation

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-short-name.rst
   :nosignatures:

   extract_iqm_pulse
   quantum_op_to_qrisp_func

Custom Pulse Operations
-----------------------

.. currentmodule:: iqm.qrisp_iqm.custom_pulse_operations

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-short-name.rst
   :nosignatures:

   delay
