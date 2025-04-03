IQM Client
###########

Client-side Python library for connecting to an `IQM <https://meetiqm.com/>`_ quantum computer.

Includes as an optional feature `Qiskit <https://qiskit.org/>`_ and `Cirq <https://quantumai.google/cirq>`_
adapters for `IQM's <https://www.meetiqm.com>`_ quantum computers, which allow you to:

* Transpile arbitrary quantum circuits for IQM quantum architectures
* Simulate execution on IQM quantum architectures with IQM-specific noise models
  (currently only the Qiskit adapter contains IQM noise models)
* Run quantum circuits on an IQM quantum computer

Installation
============

For executing code on an IQM quantum computer, you can use for example
`Qiskit on IQM <https://docs.meetiqm.com/iqm-client/user_guide_qiskit.html>`_ or
`Cirq on IQM <https://docs.meetiqm.com/iqm-client/user_guide_cirq.html>`_, which can be installed as optional
features of IQM Client from the Python Package Index (PyPI), e.g.:

.. code-block:: bash

    $ uv pip install iqm-client[qiskit,cirq]

.. note::

    If you have previously installed the (now deprecated) ``qiskit-iqm`` or ``cirq-iqm`` packages in your
    Python environment, you should first uninstall them with ``$ pip uninstall qiskit-iqm cirq-iqm``.
    In this case, you should also include the ``--force-reinstall`` option in the ``iqm-client`` installation command.

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
and running the docs builder. For versions 20.12 and later this is done by running ``./docbuild`` in the
``iqm-client`` root directory, and for earlier versions by running ``tox run -e docs``.

``./docbuild`` or ``tox run -e docs`` will build the documentation at ``./build/sphinx/html``.
These commands require installing the ``sphinx`` and ``sphinx-book-theme`` Python packages and
`graphviz <https://graphviz.org/>`_.

Copyright
=========

IQM Client is free software, released under the Apache License, version 2.0.

Copyright 2021-2025 IQM Client developers.
