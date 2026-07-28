:orphan:

.. _qrisp_iqm_api:

Qrisp IQM API Reference
=======================

This section provides a curated overview of the ``iqm.qrisp_iqm`` package.
For the complete auto-generated API, see the


Backend Infrastructure
----------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :class:`~iqm.qrisp_iqm.backends.IQMBackend`
     - Qrisp backend for executing circuits on IQM quantum hardware. Supports both
       gate-level and pulse-level submission. Automatically detects whether a circuit
       contains :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation` instructions
       and routes to the appropriate submission path.
   * - :class:`~iqm.qrisp_iqm.backends.IQMCircuitJob`
     - Job handle for gate-level circuit submissions.
   * - :class:`~iqm.qrisp_iqm.backends.IQMPulseJob`
     - Job handle for pulse-level playlist submissions.


Circuit Conversion
------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.qrisp_to_iqm_converter`
     - Converts a Qrisp :class:`~qrisp.QuantumCircuit` into an IQM
       :class:`~iqm.pulse.Circuit`. Maps Qrisp qubit indices to physical qubit names
       (``QB1``, ``QB2``, …) based on the device's dynamic quantum architecture.


Pulse Operations
----------------

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


Transpilation Passes
--------------------

Default Pipeline
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.passes.create_iqm_pass_manager`
     - Factory that builds a fully configured :class:`~qrisp.PassManager` implementing
       the complete plasma-sabre transpilation pipeline: predicate-based decomposition,
       layout, routing, SWAP optimization, and gate conversion to CZ + PRX.
   * - :func:`~iqm.qrisp_iqm.passes.transpile_to_iqm`
     - Convenience function that calls :func:`~iqm.qrisp_iqm.passes.create_iqm_pass_manager`
       and immediately runs the resulting pipeline on a circuit.

Layout Passes
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.passes.routing.plasma_layout`
     - Stochastic initial qubit placement. Tries VF2++ subgraph isomorphism first;
       falls back to heuristic search. Parameters: ``connectivity``, ``effort``,
       ``depth_weight``.
   * - :func:`~iqm.qrisp_iqm.passes.routing.vf2pp_layout`
     - Exact subgraph isomorphism layout. Raises an error if no perfect mapping exists.
   * - ``manual_layout``
     - Explicit mapping of logical qubits to physical qubit indices.

Routing Passes
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.passes.routing.plasma_route`
     - Inserts SWAP gates to satisfy connectivity constraints. Parameters:
       ``connectivity``, ``effort``, ``depth_weight``.

Optimization Passes
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.passes.commute_phases`
     - Commutes phase gates to reduce circuit depth.
   * - :func:`~iqm.qrisp_iqm.passes.measurement_parallelization`
     - Parallelizes measurement operations for more efficient execution.


Quantum Error Correction
------------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :class:`~iqm.qrisp_iqm.qec.DetectorExperiment`
     - Decorator class for defining parameterised QEC experiments. Provides
       ``.compute_LER()``, ``.batched_compute_LER()``, ``.to_stim()``,
       ``.to_qc()``, and ``.to_iqm()`` methods for the full workflow from
       classical simulation to hardware execution.


Custom Pulse Operations
-----------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~iqm.qrisp_iqm.custom_pulse_operations.delay`
     - A pre-built delay operation that inserts idle time into the circuit.
       This is a :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
       wrapper around IQM's native delay gate.
