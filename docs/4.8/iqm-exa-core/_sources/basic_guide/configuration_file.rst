EXA Configuration File
======================

EXA uses an optional YAML file for storing a configuration
that overrides the default configuration from the server.
This page explains the sections of the configuration file,
its intended structure and the functionality that it enables.

:download:`Example experiment configuration file <experiment.yml>`

.. note::

    Configuration files are written for a specific target system.
    The example file is compatible with a (5-qubit) IQM Spark system.
    To use a configuration file for some other system, download a file template from the server.

.. _configuration_initialisation:

Loading the configurations
--------------------------

EXA configuration is loaded using :func:`~exa.core.provider.provider.init_experiment_configuration`.
By default, the recommended configuration is loaded from the server.

.. code:: python

    init_experiment_configuration(username="John")

Alternatively, you can give a relative path to a configuration file.
You can copy the recommended configuration from
`{host}/api/v1/quantum-computers/{quantum-computer}/artifacts/exa-configuration``,
make changes to it,
and save it locally. Then

.. code:: python

    init_experiment_configuration(path="path/to/experiment_configuration.yml", username="John")

If you only wish to browse old data and don't want to declare a specific experiment file,
you can skip loading the configuration.

.. _experiment_section:

EXA configuration
-----------------

EXA configuration file is structured as shown here:

.. code-block:: yaml

    experiment:
        user: "Jane Doe"  # Can be used to search experiment data.
        controller_mapping: "qpu_convention"
        components:
            ...
        component_groups:
            ...
        gate_definitions:
            ...
        gates_used:
            ...
        gates:
            ...
        settings:
            ...
        <experiment_name>:
            ...

The different sections are explained below.

Section: ``gate_definitions``
*****************************

This section allows defining new gates, their properties and implementations.

Each key in this section is the name of a gate/quantum operation, and the content corresponds
to a :class:`iqm.pulse.quantum_ops.QuantumOp`. When creating instruction schedules,
the gate name can be used to get an implementation from the schedule builder.

Some gate names such as "measure", "prx" and "cz" are reserved for the canonical native operations
defined in :mod:`iqm.pulse.gates.default_gates`. They may appear in the ``gate_definitions`` section, but you can
only change their :attr:`implementations` and :attr:`defaults_for_locus` attributes.

Example of the ``gate_definitions`` section:

.. code-block:: yaml

    gate_definitions:
        prx:  # standard gate
            implementations:
                # if no specific implementation is requested, the first in the dict is used
                drag_gaussian: PRX_DRAGGaussian
        cnot: # not a standard gate
            arity: 2
            params: []
            symmetric: false
            implementations:
                my_cnot_impl: MyCNOTImplementation  # Reference to a class defined and registered in runtime.
                my_cnot_impl2: path/to/my/custom/implementation.py::MyCNOTImplementation2  # Class in a file

Section: ``gates_used``
***********************

This section is a simple list of gate names, which must be canonical native operations from :mod:`iqm.pulse`,
or defined in the ``gate_definitions`` section. Only the gates listed here will appear in the EXA
settings tree and be available to the experiments.

If the section is missing, all the canonical gates and the gates defined in ``gate_definitions`` will be
available. However, this may be undesirable since it can easily clutter up the settings tree with unnecessary
entries.


Section: ``gates``
******************

This section defines the default parameter values for gate implementations.
It is used e.g. to define the starting point of the gate calibration procedure.
Once the station has been calibrated, many of these values are typically
overridden by calibration observations.

The structure for each parameter is ``<gate name>: <implementation name>: <locus>: <parameter name>: <value>``.


.. code-block:: yaml

    gates:
        prx:
            drag_gaussian:
                []: # Empty locus means these values serve as the defaults to all loci, including unlisted ones.
                    # Individual default values are overridden by locus-specific sections below.
                    duration: 40e-9
                    amplitude_i: 1.0
                    amplitude_q: 0.0
                    full_width: 50e-9
                [QB2]:
                    duration: 60e-9
                    full_width: 70e-9
                [QB5]:
                    duration: 60e-9
                    full_width: 70e-9


.. _components_section:

Section: ``components``
***********************

This section lists chip components (qubits, couplers, etc) of the QPU that we are interested in.
Each component is associated with a set of operations that can be performed on it,
and each operation is a link to a controller exposed by Station Control.

In standard setups, the components and their operations follow a naming convention,
see :meth:`~iqm.cpc.core.config.ExperimentConfiguration.add_conventional_controller_mapping`,
and we do not need to define them explicitly.
Instead, we can define a top-level key and value ``controller_mapping: "qpu_convention"``.

When the convention is used, the ``components`` and ``component_groups`` sections
can be omitted completely. If they do exist, they override the convention component by component.
Overriding the convention only makes sense in advanced use cases, where the user has control over the
station configuration.

The standard configuration supports 3 operations on qubits: drive, readout, and flux.
The components section should also include couplers (for CZ gates, only flux operation is used), and probelines
(at least readout operation is used).

The components that exist on a QPU can be found in the :attr:`~exa.core.experiment.experiment.Experiment.chip_topology`
attribute of ``Experiment``.
The component names (keys in the file) should also match the ones defined in the chip topology.

Here are two equivalent examples, with and without ``controller_mapping``:

.. code-block:: yaml

    controller_mapping: "qpu_convention"
    components:
        QB1:
            drive_2: QB1__drive_2 # non-standard operation

.. code-block:: yaml

    components:
        QB1:
            drive: QB1__drive
            drive_2: QB1__drive_2 # non-standard operation
            flux: QB1__flux
        QB2:
            drive: QB2__drive
            flux: QB2__flux
        TC-1-2:  # tunable coupler
            flux: TC-1-2__flux
        PL-A:  # Readout probeline
            readout: PL-A__readout

.. _component_groups_section:

Section: ``component_groups``
*****************************

The components of the previous section can be grouped for different purposes.
When constructing settings for experiments involving one of the group members, EXA experiments automatically add
all operations of the group into the settings.
The members of the group are declared with the keyword ``member_names``.
If keyword ``member_names`` is not declared, all qubits from the chip are used as default value.

Example of the ``component_groups`` section:

.. code-block:: yaml

    component_groups:
        my_group:
            member_names: ["QB1", "QB2", "QB3", "QB4", "QB5"]
            crosstalk: crosstalk_controller

.. _settings_section:

Section: ``settings``
*********************

In this section, default controller settings can be specified. The nested structure should match precisely the structure
of the settings returned by ``experiment.get_settings``.

There are several places where you may assign settings and they
are merged based on this order (1 - lowest priority, 4 - highest):

1. ``common`` section under ``settings``. These settings will be common to all experiments and components.
2. Named component under ``settings``. E.g. ``QB1`` in the example below.
3. ``common`` section under :ref:`experiment_setting_section`. These settings will be common to all components
   when using the specific experiment.
4. specific component under :ref:`experiment_setting_section`.

Example of settings section:

.. code-block:: yaml

    settings:
        common:  # Settings common to all experiments and components.
            readout:
                center_frequency: 5e9

            drive:
                frequency: 4e9
                awg:
                    trigger_delay: 100e-9

        QB1:  # Settings for QB1 component in all experiments.
            drive:
                frequency: 4.44e9

.. _experiment_setting_section:

Section: ``experiments``
************************

All other sections in the EXA configuration are considered experiment-specific sections, matched by the name of the
experiment.

Each experiment can define settings (priorities 3 and 4) that override the common settings defined above (priorities
1 and 2).
The structure is the same as in the common section.

In addition, the default sweep ranges can be specified per experiment:

Example of experiment settings section:

.. code-block:: yaml

    spectroscopy:
        settings: # These settings apply only when running "spectroscopy".

            QB1:  # Settings for QB1 in a spectroscopy experiment.
                drive:
                    awg:
                        status: False

        sweeps:  # Default sweep options.
            - name: "frequency"
              type: "linear"
              parameter: "readout_1.frequency"
              start: 6.375e9
              stop: 6.425e9
              count: 101
