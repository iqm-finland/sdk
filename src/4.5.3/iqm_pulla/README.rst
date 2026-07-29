IQM Pulla
#########

Pulla (pulse-level access) is a client-side Python library which enables the generation and
execution of pulse-level jobs on an `IQM <https://iqm.tech/>`_  quantum computer.
Within the existing IQM QCCSW stack, Pulla is somewhere between
circuit-level execution and EXA experiments.

An interactive user guide is available as a Jupyter notebook in the ``docs`` folder.


Installation
============

Create and activate a virtual environment, and install Pulla with some extras:

.. code-block:: bash

    $ uv pip install "iqm-pulla[notebook, qiskit, qir]"

The ``[notebook]`` option is to be able to run the example Jupyter notebooks:

.. code-block:: bash

    $ jupyter-notebook

The ``[qiskit]`` option is to enable Qiskit-related features and utilities, like converting Qiskit circuits
to Pulla circuits, or constructing a Qiskit-compatible compiler instance.

The ``[qir]`` option is to enable QIR support, e.g. the ``qir_to_pulla`` function.

.. note::

    `uv <https://docs.astral.sh/uv/>`_ is highly recommended for practical Python environment and package management.


Documentation
=============

Documentation for the latest version is `available online <https://docs.iqm.tech/iqm-pulla/>`_.


Testing
=======

If you want to run a particular notebook and see the output cells printed in the terminal, you can use ``nbconvert`` with ``jq``
(https://jqlang.github.io/jq/download/) like so:

.. code-block:: bash

    jupyter nbconvert --to notebook --execute  docs/Quick\ Start.ipynb --stdout | jq -r '.cells[] | select(.outputs) | .outputs[] | select(.output_type == "stream") | .text[]'


Copyright
=========

Copyright 2025-2026 IQM

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
