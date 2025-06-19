IQM QAOA
########

Installation
============

Usually it makes sense to use a new Python environment, to isolate your setup from the global Python installation. That way, you can play around without messing the rest of your system.

- Using uv in terminal:

  .. code-block:: bash

    uv venv --python 3.11
    source .venv/bin/activate

- Using Conda in terminal:

  .. code-block:: bash

    conda create -n qaoa-library python=3.11
    conda activate qaoa-library

- In Visual Studio Code:

  #. Open the list of commands ``Ctrl`` + ``Shift`` + ``p``.

  #. Select `Python: Create Environment`.

  #. Select `Venv`.

  #. Select the correct Python version.


Then run


.. code-block:: bash

    pip install iqm-qaoa


If you have already installed the ``QAOA`` library and want to get the latest release you can add the ``--upgrade`` flag


.. code-block:: bash

    pip install iqm-qaoa --upgrade

Documentation
=============

Documentation for the latest version is `available online <https://docs.meetiqm.com/iqm-qaoa/>`_.
