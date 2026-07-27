.. _qrisp_iqm_backend_infrastructure:

Backend Infrastructure
======================

The central entry point for running Qrisp circuits on IQM hardware.
:class:`IQMBackend` manages authentication, submission, and result retrieval.
Before submission, circuits are transpiled through a :class:`~qrisp.PassManager`
that you control: accept the pre-configured default pipeline, customize it with
your own passes, or pass a fully hand-tuned pipeline via the ``pass_manager``
argument.  The backend automatically routes circuits to either gate-level
execution (via IQM Client) or pulse-level execution (via Pulla) depending on
whether the circuit contains :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
instructions.

Job handles track execution progress and retrieve results:

* :class:`IQMCircuitJob` — gate-level submissions.
* :class:`IQMPulseJob` — pulse-level playlist submissions.

.. currentmodule:: iqm.qrisp_iqm.backends

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-class-template.rst
   :nosignatures:

   IQMBackend

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-job-template.rst
   :nosignatures:

   IQMCircuitJob
   IQMPulseJob
