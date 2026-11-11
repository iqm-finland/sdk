.. _qrisp_iqm_qec:

Quantum error correction
========================

.. currentmodule:: iqm.qrisp_iqm.qec

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :class:`~iqm.qrisp_iqm.qec.DetectorExperiment`
     - Decorator class for defining parameterised QEC experiments. Provides
       ``.compute_LER()``, ``.batched_compute_LER()``, ``.to_stim()``,
       ``.to_qc()``, and ``.to_iqm()`` methods for the full workflow from
       classical simulation to hardware execution.
