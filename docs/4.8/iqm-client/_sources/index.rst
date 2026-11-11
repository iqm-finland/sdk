IQM client
##########

:Release: |release|
:Date: |today|

Client-side library for connecting to an `IQM <https://iqm.tech/>`_ quantum computer.
Provides both circuit-level and pulse-level access.

Includes adapters for the `Qiskit <https://qiskit.org/>`_, `Cirq <https://quantumai.google/cirq>`_ and
`Qrisp <https://www.qrisp.eu/>`_ quantum computing frameworks, each with its own optional dependencies.
See user guides for :ref:`Qiskit <User guide Qiskit>`, :ref:`Cirq <User guide Cirq>` and
:ref:`Qrisp <User guide Qrisp>` for introductions on how to install and use the framework adapters.
Each adapter exposes a dedicated Python import package with classes and
functions for circuit construction, transpilation, execution, and result retrieval.


Contents
========

.. toctree::
   :maxdepth: 2

   readme
   user_guide_qrisp/index
   user_guide_qiskit
   user_guide_cirq
   integration_guide
   API

.. toctree::
   :maxdepth: 1

   common_errors
   references
   changelog
   license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
