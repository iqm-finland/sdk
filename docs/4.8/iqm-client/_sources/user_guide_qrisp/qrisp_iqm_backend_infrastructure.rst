.. _qrisp_iqm_backend_infrastructure:

Backend infrastructure
======================

The central entry point for running Qrisp circuits on IQM hardware.
:class:`~iqm.qrisp_iqm.backends.backend.IQMBackend` manages authentication, job submission, and result retrieval.
Before submission, circuits are transpiled through a :class:`~qrisp.PassManager`
that you control: accept the pre-configured default pipeline, customize it with
your own passes, or pass a fully hand-tuned pipeline via the ``pass_manager``
argument.  The backend automatically routes circuits to either gate-level or pulse-level execution depending on
whether the circuits contain :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
instructions.

Various Job classes are used to track execution progress and retrieve results:

* :class:`.IQMCircuitJob` — gate-level circuit submissions.
* :class:`.IQMPulseJob` — pulse-level playlist submissions.

.. currentmodule:: iqm.qrisp_iqm.backends

.. autosummary::
   :signatures: none

   IQMBackend
   IQMCircuitJob
   IQMPulseJob
