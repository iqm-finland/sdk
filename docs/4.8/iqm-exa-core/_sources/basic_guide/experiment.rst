.. _run_and_use_experiments:

*******************
Running Experiments
*******************

Here we describe the basic workflow when working with Experiments.
This section is **not** about writing new experiments, or about running quantum circuits.

Basic workflow
==============

.. _initial_setup:

0. Initial setup
----------------

First, we need to connect to a selected Quantum computer and load an Exa configuration to use.

.. code-block:: python

    from exa.core.provider import provider

    provider.init_iqm_server_runtime(
        iqm_server_url="https://example.qc.com/",
        quantum_computer="<quantum-computer-name>",
        username="<username>",
        token="<api-token>",  # or set the IQM_TOKEN environment variable instead
        # experiment_configuration_path="experiment.yml",  # optional
    )

    # Optional: set up logging
    # provider.init_loggers("loggers.yml")


Here, ``iqm_server_url`` is the URL of the IQM Server, and ``quantum_computer`` is the name
of the quantum computer to use, since one server can host several quantum computers.
The API token can be obtained from the IQM Server frontend using a web browser.

The optional command uses the ``"loggers.yml"`` file to set up logging behavior.

Instead of the above, you can define the same information through environment variables.
To do that, set

.. code-block:: bash

    STATION_CONTROL_SERVICE_URL="https://example.qc.com"
    IQM_TOKEN="..."
    EXA_CONFIGURATION_PATH="experiment.yml"
    EXA_LOGGERS_CONF_PATH="loggers.yml.example"

in the ``.env`` file and call

.. code-block:: python

    provider.init_from_env()

When ``env_path`` is not given, ``init_from_env()`` reads the current process
environment and loads values from an auto-discovered ``.env`` file.

If you want to load a specific env file instead of the auto-discovered ``.env``, pass it explicitly:

.. code-block:: python

    provider.init_from_env(env_path="path/to/.env")

These two approaches can be mixed.

1. Choose calibration
---------------------

To get meaningful results, EXA must know which calibration values to use.
Calibration is described using Observations which are stored in the server's database.

In short, you can select the observations using tags or calibration sets.
The various rules related to Observation loading are explained in more detail in :ref:`observations`.

To get started without diving into the details, you can load the default calibration set, which is the same
that is used by circuit execution through other client libraries:

.. code-block:: python

    import exa.core as ec
    ec.set_global_observation_stash([ec.GetCalibrationSet("default")])


2. Create experiment instance(s)
--------------------------------

To conduct an experiment, we create an instance of the Experiment class, for example

.. code-block:: python

    from exa.core import Experiment
    from exa.public_experiments import Rabi
    exp = Experiment()
    rabi = Rabi()

An experiment represents a recipe to conduct a certain task on the quantum computer.
The ``Experiment`` instances generate the execution instructions,
keep experimental data gathered during the session in memory,
and allow analyzing and plotting the data.
Each class has one or multiple default Sweeps
they are designed to perform, and default parameters they are designed to measure.

:class:`.Experiment` is the base class that works also as a standalone prototyping tool.
All other experiments, like :class:`.Rabi` in the example, are derived from the base class.
``Rabi`` contains the standard recipe for executing a textbook Rabi experiment.

Notable methods of experiments are
:meth:`~exa.core.experiment.experiment.Experiment.get_settings`,
:meth:`~exa.core.experiment.experiment.Experiment.run`,
and :meth:`~exa.core.experiment.experiment.Experiment.analyze`, explained below.

3. Modify options (optional)
----------------------------

Basic behaviour of experiments is configured via the ``options`` attribute of the experiment instance.
It is of the type :class:`~exa.core.experiment_options.ExperimentOptions`.

Some options are shared by all experiments, while some are experiment-specific.
The common attributes are explained below:

:attr:`~.ExperimentOptions.repetitions`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of times each circuit is executed to collect data.
It depends on the used ``readout_mode`` how the averaging is done.

:attr:`~.ExperimentOptions.readout_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``readout_mode`` has 5 possible values. Their meanings and effects are as follows:

- ``"complex"`` (the default): normal complex voltage readout for each channel. The contrast and the excited state probability
  are computed in post processing. The averaging is done in hardware.

- ``"threshold"``: thresholded readout for each readout channel, representing the excited state probability.
  The averaging is done in hardware.

- ``"single_shot"``: complex single shot readout for each channel. The averaged complex readout (equivalent to the above
  ``"complex"`` mode) is computed in post processing, in addition to contrast and excited state probability.
  The averaging is done in software, in the aforementioned post processing step.

- ``"single_shot_threshold"``: thresholded single shot readout for each readout channel. The averaged complex readout
  (equivalent to the above ``"complex"`` mode) is computed in post processing. The averaging is done in software,
  in the aforementioned post processing step.

- ``"counter"``: the correlated multi-qubit readout, counting the occurrences of bitstrings.
  The data for each readout group is extracted from the full data in addition to the single-qubit excited state
  probability for each qubit. The counting and averaging is done on the server.

:attr:`~.ExperimentOptions.active_reset_cycles`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of active ``reset`` operations inserted at the beginning of each segment for each active qubit. Uses the default
implementation of ``reset``. ``None`` means active reset is not used but instead reset is done by waiting (relaxation),
using the default implementation of the ``reset_wait`` operation. Integer values smaller than 1 result in neither active
nor reset by waiting (relaxation) is used, in which case any reset operations must be explicitly added in the circuit generation
function.

:attr:`~.ExperimentOptions.readout_programming_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``readout_programming_mode`` controls how readout operations are added to the circuit(s).

- ``"classic"`` (the default): Each qubit is measured at the end of the circuit.

- ``"heralding"``: Each qubit is measured before and after each circuit execution. The measurement before is the
  so-called "heralding" measurement that can be used to check that the qubit was initially in the ground state.
  The heralding data appears in the dataset as a separate key, but data is not post-selected automatically.

- ``"reset_at_end"``: Similarly to ``"classic"``, a terminal measurement is added. However, the terminal measurement is
  done with a feedback key, and a conditional prx is applied after. NOTE: this mode
  overrides :attr:`~exa.core.experiment_options.ExperimentOptions.active_reset_cycles`
  (forcing one conditional reset at the end of the segment)
  and requires a station which supports fast feedback.

- ``"custom"``: Measurements are not added automatically. Any measurement must be defined explicitly by the
  experiment source code.

:attr:`~.ExperimentOptions.used_measure_operations`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Names of the measurement QuantumOps the various readout programming modes use when building
the circuits.

The purpose of having more than one measure QuantumOps is to allow calibrating and optimizing different
implementations for different use cases. For example, for a terminal measurement of a qubit, it is not so important
that the measurement is projective, as the qubits state is just discarded after that.

The value of this attribute is a dict mapping different measure-use cases to the measure operation names. By default,
it is given as:

.. code-block:: python

    used_measure_operations: dict[str, str] = {
        "terminal": "measure_fidelity",
        "heralding": "measure",
        "reset_at_end": "measure",
    }

* ``"terminal"`` is used for terminal measurement when ``readout_programming_mode`` is
  either ``"classic"`` or ``heralding``.
* ``"heralding"`` is used for the heralding measurement when ``readout_programming_mode`` is ``"heralding"``.
* ``"reset_at_end"`` is used when ``readout_programming_mode`` is ``"reset_at_end"``.

Here, the terminal measurement is by default done with the ``measure_fidelity`` gate, which is meant to optimize
measurement fidelity but might not be optimally projective. The heralding measurement and the conditional feedback
providing measurements however must be projective, so they are done with the ``measure`` gate that is optimized to
be a compromise between fidelity and QNDness (this is the standard measurement best suitable for
mid-circuit-measurements)

Users may change these default values in their experiment run, but note that this (e.g. making ``"heralding"`` use
``"measure_fidelity"``) may result in sub-optimal behavior. Specific experiments may also hard-code different default
behaviour based on their needs.

Note: this attribute applies only to the measure operations added automatically by Experiment. If the experiment adds measure
operations explicitly in its circuit, you must write the logic for choosing the right operation to the circuit
generation function itself. Typically, experiments handle this also via options, in which case there is an additional
options attribute, usually named as ``measure_gate_name``.

:attr:`~.ExperimentOptions.subscribed_components`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``subscribed_components`` define which components are read out in the experiment. The subscribed components are given as
a list of component names. These can be different from the active components given as run arguments. The run settings
will however limit the used ``subscribed_components`` (as one cannot access the readout channel of a component that is
not in the settings). By default, the ``subscribed_components`` is set to None. This will result in the behaviour that
all active run components are read out.

If the ``"counter"`` readout mode is used, the ``subscribed_components`` can also be given as a dict
of lists of component names (the so called "grouped readout"). The counter then returns correlation data
for each group, but not between the groups (the groups could be e.g. all the qubit-pairs in the QPU).
The grouped readout is only available in the ``"counter"`` readout mode, otherwise an error will be thrown.

.. code-block:: python

    # not defining the subscribed_components will read out the
    # run components: "QB1", "QB2", "QB3", "QB4"
    rabi.run(components=["QB1", "QB2", "QB3", "QB4"])

    # now we read only "QB1" and "QB2", but run with all four
    rabi.options.subscribed_components = ["QB1", "QB2"]
    rabi.run(components=["QB1", "QB2", "QB3", "QB4"])

    # use the counter's grouped readout to read all adjacent pairs
    rabi.options.readout_mode = "counter"
    rabi.options.subscribed_components = {
        "pair_1_2": ["QB1", "QB2"],
        "pair_2_3": ["QB2", "QB3"],
        "pair_3_4": ["QB3", "QB4"]}
    rabi.run(components=["QB1", "QB2", "QB3", "QB4"])


4. Modify settings (optional)
-----------------------------

The options determine the general behavior of the Experiment.
They define how to generate the Settings and pulse playlists, which are the final truth about what will be executed
on the quantum computer.

Before starting an experiment run, we might want to deviate from the baseline defined by the latest calibration.
After choosing the options and the target components for a run, the settings can be generated as, e.g.
``rabi.get_settings()``.
The usage of settings is described in detail in :doc:`Workflow with Experiment settings <settings_workflow>`.

5. Run the Experiment
---------------------
By default, the experiment is executed on all qubits / qubit pairs on the QPU in parallel,
or sequentially if the pairs overlap,
whichever is applicable to the experiment.

.. code-block:: python

    rabi.run()

The experiment knows what settings to set, sweep and measure.
We only need to select the qubit on which we want to conduct the experiment on.

To run the Rabi experiment on a selected subset of qubits we can do

.. code-block:: python

    rabi.run(["QB2", "QB3"])

where ``"QB2"`` and ``"QB3"`` are named qubits of the QPU.
The names are defined in the `components` section in the
:attr:`~exa.core.experiment.experiment.Experiment.chip_topology`.

The :meth:`.Experiment.run` method has optional arguments, explained below.
These arguments together provide a lot of flexibility and a way to improvise when conducting experiments.

``components``
~~~~~~~~~~~~~~

The locus or target of the experiment. In the example above, the experiment is performed on both qubits in parallel.

The components given to the run method as an argument define the active components that will be part of the generated circuits.
The run components are propagated to the circuit generation function.
Note: the run components must be present in the settings.
By default, the settings are generated for all components even if they are not active.

The run components can be given in four different forms:

- Flat list of component names

- List of tuples of component names, i.e. component groups

- Lists of lists of tuples of component names, i.e. colour groups. The lists of tuples then define partitions
  of the QPU based on the application's needs.

- (default) ``None``. By default, this results in running the experiment as if
  the user had provided a flat list of all qubits found in the QPU.

.. code-block:: python

    exp = Experiment(circuit_generation_function=some_circuit_function)

    # flat list, i.e. run the experiment on these qubits. if this
    # is a pair-experiment, this could result in automatically
    # finding all connected pairs of these qubits and
    # then running the experiment on them
    exp.run(components=["QB1", "QB2", "QB3", "QB4"])

    # no components given, will run with all qubits in the QPU
    exp.run()

    # groups, e.g. run the experiment on these qubit pairs exactly
    exp.run(components=[("QB1", "QB2"), ("QB3", "QB4")])

    # colour groups, i.e. partition the QPU into logically
    # separate partitions of components. these could for example
    # denote pairs we want to run in parallel.
    # The first circuit performs the operations on QB1-QB2 and QB3-QB4,
    # and second on QB2-QB4.
    exp.run(components=[[("QB1", "QB2"), ("QB3", "QB4")], [("QB2", "QB4")]])


``settings``
~~~~~~~~~~~~~~~~~~~~

.. margin:: Example: Passing custom settings

    .. code-block:: python

        settings = experiment.get_settings()
        settings.controllers.QB2.drive.frequency = 5.5e9
        experiment.run(
            settings=settings
        )

These are the ones generated in step 4.

``default_sweep_ranges``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depending on the experiment, this allows specifying a sweep range of the default sweep(s).
For the FineSpectroscopy for example, this would change the minimum, maximum and
number of points in the frequency scan of the qubit readout signal.
If not specified, the range(s) are read from the configuration file, or hardcoded defaults of the experiment are used.

Similar to the settings, the most convenient way to make modifications is to ask for the defaults using
:meth:`~exa.core.experiment.experiment.Experiment.get_default_sweeps_as_dict`,
modify the values, and feed them to ``run``.

.. margin:: Example: Passing custom settings

    .. code-block:: python

        import numpy as np
        sweeps = experiment.get_default_sweeps_as_dict()
        sweeps["my dimension"] = np.linspace(0, 1, 57)
        experiment.run(
            default_sweep_ranges=sweeps
        )

``sweeps``
~~~~~~~~~~~~~~~~~~~~

.. margin:: Example: Add sweep over frequency

    .. code-block:: python

        settings = experiment.get_settings()
        freq = settings.controllers.QB1.drive.frequency
        ranges = np.linspace(-50e6, 50e6, 50) + 5.5e9
        experiment.run(
            sweeps=[freq.sweep(ranges)]
        )

Additional sweeps to perform over the default sweeps.
These can be thought as additional ``for`` loops on top of the default sweeps of the experiment.
In general, most Settings in EXA can be swept over.
For the Rabi for example, we could add a sweep over the pulse duration, or the drive frequency.

Sweeps have a 3-level hierarchy. The simplest entity is a ``Sweep`` object which
defines a single parameter and the associated values to loop over.
A tuple Sweep objects defines a **parallel sweep**, which means several parameters are changed in tandem,
similar to the `zip` function in Python. The tuple can be omitted.
Finally, a list of N parallel sweeps defines an N-dimensional **cartesian sweep** over all the parameters.
This can be thought of as N nested for loops.


.. admonition:: Note on sweep types
    :class: dropdown

    Singular sweeps come in 2 varieties:

    **Software sweep**: Software sweep, or soft sweep, means the looping logic is done "in software", that is,
    there is a for loop in the Station Control which goes through the range of values one by one,
    passing the control to the devices after setting the current iteration of values.

    **Hardware sweep**: In contrast to software sweeps, hard sweeps perform the looping logic in a dedicated instrument.
    This means that the range of values to iterate over is first gathered to a list, which is then passed to the
    instrument.
    The instrument then returns a list of results.
    The most common use case is sweeping pulse or circuit parameters.
    The list of pulse variations is generated in advance and sent to the station to be executed as a batch.

    The hardware sweeps are always the "innermost" loop of the N-dimensional sweep, meaning the values in hardware
    sweeps change most frequently and the values in software sweeps are the slowest to change.
    To the user, they appear the same.


6. Inspect results
------------------

When calling :meth:`.Experiment.run`, a sweep request is generated and sent to the server.
The request contains the *settings* to
instruments, instructions to perform an N-dimensional sweep over the specified *sweep parameters* and to measure
data of the specified *return parameters*.
Station Control saves the data to a database, and the experiment instance reads it into the runtime memory.

After the run, the resulting dataset can be accessed by :attr:`.Experiment.last_run`
or :attr:`.Experiment.run_results`, which provide a :class:`~iqm.cpc.core.run_result.RunResult` object.

.. code-block:: python

    rabi.run_results[-1]  # Returns the latest run in the session, containing dataset and metadata.
    rabi.last_run  # Returns dataset from latest run, same as rabi.run_results[-1].

The metadata includes the run ID, experiment options, the sweep request that was sent to the station, etc.
The measurement data is collected to a `xarray datasets <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_.
It includes the sweep parameters as *coordinates* and measured return parameters as *data variables*.
You are encouraged to get familiar with the `xarray documentation <http://xarray.pydata.org/en/stable/>`_.

7. Analyze results
------------------

After a run, the dataset contains mostly raw data. To make sense of the data, we analyze it with
:meth:`.Experiment.analyze`:

.. code-block:: python

    result1 = rabi.analyze()  # analyze latest run by default

Here, we have analyzed the latest run of the session.
Most analysis functions do some sort of fitting to the raw data according to the mathematical model defined in the
experiment, and store the resulting fitting parameters.

The ``analyze`` method returns a :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult` object,
which contains plots of data, observations made, and any data arrays that were additionally calculated during analysis.

8. Plot results
---------------

Finally, we can visualize the results with
:meth:`AnalyzeResult.plot <exa.core.analyze_result.analyze_result.AnalyzeResult.plot>`.

.. code-block:: python

    result1.plot()  # show the default plot
    result1.plot("QB3", figure_options=...)  # show selected plot and customize it
    result1.plot_all()  # show all defined plots

See the ``plotting_guide`` notebook for an in-depth guide on the plotting functionality.

We can also summarize the results in a report, which contains the plots and a list of observations:

.. code-block:: python

    result1.render_report()
