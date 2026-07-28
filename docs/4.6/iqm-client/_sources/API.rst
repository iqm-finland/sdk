API Reference
=============

The IQM Client library provides adapters for several quantum computing frameworks.
Each adapter exposes a dedicated Python package (``iqm.*``) with classes and
functions for circuit construction, transpilation, execution, and result retrieval.
The table below links to the API reference for each package.

.. toctree::
   :hidden:

   api/iqm.iqm_client
   api/iqm.iqm_server_client
   api/iqm.cirq_iqm
   api/iqm.qiskit_iqm
   qrisp_iqm_api_index

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Framework
     - Description
     - Documentation
   * - **cirq_iqm**
     - Cirq adapter for IQM quantum computers.
     - :doc:`API <api/iqm.cirq_iqm>`
   * - **iqm_client**
     - Client-side library for executing quantum circuits on IQM hardware.
     - :doc:`API <api/iqm.iqm_client>`
   * - **iqm_server_client**
     - Client-side library for connecting to the IQM Server API.
     - :doc:`API <api/iqm.iqm_server_client>`
   * - **qiskit_iqm**
     - Qiskit adapter for IQM quantum computers.
     - :doc:`API <api/iqm.qiskit_iqm>`
   * - **qrisp_iqm**
     - Qrisp adapter for IQM — curated reference covering backend infrastructure,
       transpilation (Plasma-SABRE), pulse operations & conversion, and quantum error
       correction.
     - :doc:`Backend Infrastructure <qrisp_iqm_backend_infrastructure>`,
       :doc:`Transpilation <qrisp_iqm_transpilation>`,
       :doc:`Pulse Operations <qrisp_iqm_pulse_operations>`,
       :doc:`QEC <qrisp_iqm_qec>`
