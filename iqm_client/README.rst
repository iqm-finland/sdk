IQM Client
###########

Client-side Python library for connecting to an `IQM <https://meetiqm.com/>`_ quantum computer.

Includes as an optional feature `Qiskit <https://qiskit.org/>`_ and `Cirq <https://quantumai.google/cirq>`_
adapters for `IQM's <https://www.meetiqm.com>`_ quantum computers, which allow you to:

* Transpile arbitrary quantum circuits for IQM quantum architectures
* Simulate execution on IQM quantum architectures with IQM-specific noise models
  (currently only the Qiskit adapter contains IQM noise models)
* Run quantum circuits on an IQM quantum computer

Also includes a CLI utility for managing user authentication when using IQM quantum computers.

Installation
============

For executing code on an IQM quantum computer, you can use for example the
Qiskit on IQM or Cirq on IQM user guides found in the documentation.
Qiskit on IQM and Cirq on IQM are optional features that can be installed alongside the base IQM Client library.
An example showing how to install from the public Python Package Index (PyPI):

.. code-block:: bash

    $ uv pip install iqm-client[qiskit,cirq]

.. note::

    If you have previously installed the (now deprecated) ``qiskit-iqm`` or ``cirq-iqm`` packages in your
    Python environment, you should first uninstall them with ``$ pip uninstall qiskit-iqm cirq-iqm``.
    In this case, you should also include the ``--force-reinstall`` option in the ``iqm-client`` installation command.

The CLI utility for managing user authentication can also be installed as an optional feature:

.. code-block:: bash

    $ uv pip install iqm-client[cli]

IQM Client by itself is not intended to be used directly by human users. If you want just the base IQM Client library,
though, you can install it with

.. code-block:: bash

    $ uv pip install iqm-client

.. note::

    `uv <https://docs.astral.sh/uv/>`_ is highly recommended for practical Python environment and package management.

Documentation
=============

Documentation for the latest version is `available online <https://docs.meetiqm.com/iqm-client/>`_.
You can build documentation for any older version locally by downloading the corresponding package from PyPI,
and running the docs builder. For versions greater than equal to 20.12 but less than 33.0.0 this is done by
running ``./docbuild`` in the ``iqm-client`` root directory, and for earlier versions by running ``tox run -e docs``.

``./docbuild`` or ``tox run -e docs`` will build the documentation at ``./build/sphinx/html``.

Versions greater than or equal to 33.0.0 use the command:
``sphinx-build -q -d build/.doctrees/iqm-client iqm-client/docs build/docs/iqm-client``
(``build/docs/`` directory has to be created first).

These commands require installing the ``sphinx`` and ``sphinx-book-theme`` Python packages and
`graphviz <https://graphviz.org/>`_.

Copyright
=========

IQM Client is free software, released under the Apache License, version 2.0.

Copyright 2021-2025 IQM Client developers.
