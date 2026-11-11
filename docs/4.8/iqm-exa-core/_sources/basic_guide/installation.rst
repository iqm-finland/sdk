Installation
============

We strongly recommend to install into a virtual environment.
For creating one, refer to the documentation of the tool of your choice.
In this example, we use
`uv <https://docs.astral.sh/uv/>`_.

Install EXA with

.. code-block:: bash

    uv pip install iqm-exa-core

Optionally, install Jupyter notebooks:

.. code-block:: bash

   uv pip install iqm-exa-core notebook

To get started, continue reading :ref:`run_and_use_experiments`.
Or if you prefer a more hands-on approach, see the example notebook:
:doc:`../notebooks/experiment_basics`.

Optional files
--------------

The quantum computer server provides a default configuration which enables most functionality.
For extra customization, you can have personal configuration files on your system which override the defaults.
The optional files are: The EXA configuration file, a ``.env``, and a logger configuration file.

The purpose and format of the EXA configuration file is explained in :doc:`Configuration File <configuration_file>`.
You can also have a separate configuration file for controlling the behavior of log messages.

The paths to the configuration files and the URL to the server can be specified at runtime,
or alternatively through environment variables.
To use the environment variables, create a file named ``.env`` in your working directory,
for example, where your Jupyter notebook files are located.

Examples for the EXA configuration, ``.env``, and logger configuration:

* :download:`Example experiment configuration file <experiment.yml>`
* :download:`Example .env file <.env.example>`
* :download:`Example logger configuration <loggers.yml.example>`
