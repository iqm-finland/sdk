IQM Pulla
#########

Pulla (pulse-level access) is a client-side software which allows the user to control the generation and
execution of pulse schedules on a quantum computer. Within the existing IQM QCCSW stack, PulLA is somewhere between
circuit-level execution and EXA-experiment.

An interactive user guide is available as a Jupyter notebook in the `docs` folder.

Use
===

Create a virtual environment and install dependencies:

.. code-block:: bash

    conda create -y -n pulla python=3.11 pip=23.0
    conda activate pulla
    pip install iqm-pulla[notebook, qiskit, qir]

The ``[qiskit]`` option is to enable Qiskit-related features and utilities, like converting Qiskit circuits to Pulla circuits, constructing a compatible compiler instance, or constructing a ``PullaBackend`` for running Qiskit jobs.

The ``[qir]`` option is to enable QIR support, e.g. the ``qir_to_pulla`` function.

The ``[notebook]`` option is to be able to run the example notebooks, using
and run it in Jupyter Notebook:

.. code-block:: bash

    jupyter-notebook

Development
===========

Install development and testing dependencies:

.. code-block:: bash

    pip install -e ".[dev,notebook,qiskit,qir,testing,docs]"

e2e testing is execution of all user guides (Jupyter notebooks). User guides cover the majority of user-level features,
so we achieve two things: end-to-end-test Pulla as a client library, and make sure the user guides are correct.
(Server-side use of Pulla is e2e-tested as part of CoCoS.)

You have to provide CoCoS and Station Control URLs as environment variables:

.. code-block:: bash

    COCOS_URL=<COCOS_URL> STATION_CONTROL_URL=<SC_URL> tox -e e2e

Notebooks are executed using `jupyter execute` command. It does not print any output if there are no errors. If you want
to run a particular notebook and see the output cells printed in the terminal, you can use ``nbconvert`` with ``jq``
(https://jqlang.github.io/jq/download/) like so:

.. code-block:: bash

    jupyter nbconvert --to notebook --execute  docs/Quick\ Start.ipynb --stdout | jq -r '.cells[] | select(.outputs) | .outputs[] | select(.output_type == "stream") | .text[]'

Run unit tests, build docs, build package:

.. code-block:: bash

    tox
    tox -e docs
    tox -e build


Copyright
=========

Copyright 2024 IQM

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
