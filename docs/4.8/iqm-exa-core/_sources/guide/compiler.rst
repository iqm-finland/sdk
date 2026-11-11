.. _guide_compiler:

Configuring compiler passes
===========================

Under the hood, the :class:`~exa.core.experiment.experiment.Experiment` uses the Circuit-to-Pulse-Compiler
(CPC, :class:`~iqm.cpc.compiler.compiler.Compiler`) to produce the final :class:`~iqm.models.playlist.playlist.Playlist`
and :class:`~iqm.station_control.client.RunDefinition` which are sent to the station for execution. You might even view
the Experiment as a thin layer on top of the
Compiler, or a convenience class which simply lets you define the circuits and analysis, but all the "heavy lifting"
is done elsewhere.

In this chapter we will learn more about how the Compiler works and how to modify it to best serve the specific
experiment we are writing. It is heavily recommended to also consult the :doc:`compiler documentation <iqm.cpc:index>`,
specifically the :doc:`compilation stages <iqm.cpc:Compilation Stages>`.

What is a compilation pass?
###########################

In CPC, the compilation pass is simply a function that takes some ``data``, which could be an abstract circuit of some
kind, a list of :class:`~iqm.pulse.timebox.TimeBox`\ es, or even :class:`~iqm.pulse.playlist.schedule.Schedule`,
and also some context arguments, and produces a different kind of data, potentially modifying the context.

It could be visualized as a generic "process", in the style of Bob Coecke and Aleks Kissinger, with time flowing left to
right:

.. figure:: figures/pass.png
    :alt: A compiler pass taking "circuit" and "context" and outputting "timebox" and "context".
    :align: center

    The pass consumes the circuit, produces the timebox and modifies the context.

In fact, we have actually seen a compilation pass already, although not a typical one! The circuit generation
function is functionally equivalent to a pass that only takes context as input:

.. figure:: figures/pass_timebox_gen.png
    :alt: A compiler pass taking "context" and outputting "timebox" and "context".
    :align: center

    The pass produces the timebox based on the context and modifies the context.

Ok, let's be explicit. What is this context?

Context is simply a dictionary of objects and variables which can be helpful in compilation. The default context contains:

1. The full settings tree under key "settings".
2. The calibrated :class:`~iqm.pulse.builder.ScheduleBuilder` under the key "builder".
3. The :class:`~exa.common.qcm_data.chip_topology.ChipTopology` under the key "chip_topology".
4. All the components given to the run and properly processed, under the key "components".

And also some other things. If a pass function has an argument matching the name of one of the **default** context variables
or the context itself,
during the compilation that context variable is simply given to the pass. All the other arguments become **compiler settings**,
which are fetched from ``settings.stages.stage_name.pass_name.argument_name``. You may recognize that this is how
the circuit generation arguments work, as described in the :doc:`circuit generation functions <circuit>` chapter.

This makes writing them extremely convenient, as long as you know what is in the context. Since the compiler passes are
pure functions and thus there is no way to do ``self`` references within them, the context is the designed method for adding
arbitrary stateful effects to your compiler pass functions. As such there's no limitation to what you can write to context,
so any stateful effects should be achievable.

The first argument to the pass function should always be the circuit payload - circuits, timeboxes, schedules,
etc -- the actual thing being compiled. Any further arguments other than ``context`` or the aforementioned default variables
need to be typed and conform to the supported :class:`~exa.common.data.parameter.Setting` data types.

The passes are arranged into :class:`~iqm.cpc.compiler.compilation_stage.CompilationStage` s, although this arrangement is
mostly for convenience and to arrange the passes according to abstraction layers they operate on -- in the end, all the
passes are executed in order.

The entire experiment execution can be visualized like so:

.. figure:: figures/compiler_exa.png
    :alt: All the compilation stages in exa and beyond, in order from left to right.
    :align: center

    The default compiler used in EXA. The presence of the "circuit stage" is dependent on the
    :class:`~exa.core.experiment.experiment.InputType`. Also included are the execution (which is not a compiler stage,
    but also a process in the Coecke-Kissinger sense) and post processing.

Common compilation options
##########################

We have talked about :class:`~exa.core.experiment_options.ExperimentOptions`, and how it is a general rule that the
options are allowed to affect the settings, but in case of conflict, the settings are the final source of truth. This
is also true for options which are designed to influence the compilation. Those are included by default, because they
are often changed.

Readout programming mode
************************

The option :attr:`~exa.core.experiment_options.ExperimentOptions.readout_programming_mode` takes values from
:class:`~exa.core.experiment_definitions.ReadoutProgrammingMode`. This option affects settings of the passes:

1. Heralding pass :func:`~iqm.cpc.compiler.standard_stages.prepend_heralding` - if set to ``'heralding'``, adds a
   readout **before** the circuit execution. You can use it in analysis to post-select the data, keeping only those where the
   heralding readouts found the qubits in the ground state.
2. Terminal measurement pass :func:`~exa.core.experiment.stages.add_terminal_measurement` - if set to ``'classic'``,
   adds a terminal measurement to all qubits. If set to ``'reset_at_end'``, uses the bit from that measurement to
   conditionally reset the qubits with X gates. Classic is the default option for experiments, and it should be kept
   that way for experiments that do not explicitly need to benchmark or calibrate the readout, and are just using
   one readout. This is especially true if using the parallel execution framework, which does not work very well with
   measurements added explicitly.

If the readout programming mode is set to ``'custom'``, no readout-related passes will do anything, no additional
measurements will be added on top of the one explicitly defined in the circuit. You should use this option if writing
experiments that use repeated or otherwise non-trivial readout operations.

Active reset cycles
*******************

The option :attr:`~exa.core.experiment_options.ExperimentOptions.active_reset_cycles` specifies how many active resets
are performed on each qubit before each shot. The active reset refers to a measurement followed by a conditional bit
flip using the X gate. If this option is not ``None``, the passive wait of the "reset_wait" between two subsequent realizations of the
circuit is removed, so setting it to 0 will result in a "restless" experiment. This option affects the
:func:`~iqm.cpc.compiler.standard_stages.prepend_reset` pass.

Used measure operations
***********************

The option :attr:`~exa.core.experiment_options.ExperimentOptions.used_measure_operations` is a mapping from the keys "terminal", "heralding" and "reset_at_end" onto names of measure operations
used for each one. Unless you decide to use this option for something else, it only affects the measurements added by
the passes already mentioned above and no other measurements.

Subscribed components
*********************

The option :attr:`~exa.core.experiment_options.ExperimentOptions.subscribed_components` is a list of components which
have readout, and will receive some measurement operations, whose measurement results will end up in the final dataset.
Note that if you include components that have no readout capability here, they will be ignored.

.. note::
    **Q: So not all data from the measured qubits is in the dataset?**

    A: No. The final measurement is calibrated to be performed in parallel on multiple qubits. The compilation stage
    adds the readout pulses to **all** components, not just the ones the circuit is run on. Most of the time, you are
    only interested in the qubits you are doing something to, so to not clutter the dataset, only the results
    from those qubits are included. You can, however, change that behaviour by including some additional qubits in
    this option.

By default, the option value is ``None``, which results in subscribing the components included in the ``components``
variable, which is what you give when running the experiment. It does not affect any pre-processing stage, but the
actual execution, the setting is still in the ``stages`` settings subtree: ``settings.stages.additionally_subscribed_components``.

Job options
***********

Finally, the options :attr:`~exa.core.experiment_options.ExperimentOptions.data_size_safety_switch` and
:attr:`~exa.core.experiment_options.ExperimentOptions.force_ragged_data` affect the
:func:`~iqm.cpc.compiler.standard_stages.create_run_definition` pass, which is by default the last pass executed by
the :class:`~iqm.cpc.compiler.compiler.Compiler`, building the final :class:`~iqm.station_control.client.models.RunDefinition`
sent to the station for execution. They affect the size and shape of the dataset.

The data size safety switch should generally be never False by default in your experiment, but the users can switch
it to run a longer version of experiment. You should force ragged data if the number of readouts on qubits may change
as a result of an extra sweep (most likely of some circuit argument), but you are not doing that sweep by default, as
then writing analysis which is robust to that sweep is easier.

Input type and default passes
#############################

The default input type for :class:`~exa.core.experiment.experiment.Experiment` is TimeBox, and unless you change it,
none of the circuit passes will be added. Currently, the only other available :class:`~exa.core.experiment.experiment.InputType`
is Circuit. If you wish to write an experiment, where the :doc:`circuit generation function <circuit>` returns an
abstract circuit, you should change the :attr:`~exa.core.experiment.experiment.Experiment.input_type` attribute on your
class.

The input type affects the default passes added to the compiler when initializing your experiment class. With
circuit input type, the compiler will also receive some default circuit level passes. Those passes assume the input is
an iterable of :class:`~iqm.pulse.circuit_operations.Circuit`, which contain :class:`~iqm.pulse.builder.CircuitOperation` s
and some metadata. Unless that is what you want your circuit generation function to output, you will need to create your
own passes that convert the abstract circuit you create into either an iterable of :class:`~iqm.pulse.circuit_operations.Circuit`
or an iterable of :class:`~iqm.pulse.timebox.TimeBox` (where you would delete the default circuit passes).

This behaviour is to ensure that when the circuit generation function is added as the **first pass**, and it will always
be first because it does not take any data input, the output produced by it will actually match the input of the first
default pass.

The default stages and passes are detailed in the :doc:`CPC documentation <iqm.cpc:Compilation Stages>`. The
Experiment does one additional pass on the pulse level, the aforementioned :func:`~exa.core.experiment.stages.add_terminal_measurement`
pass, which is not in the default compiler used for circuit execution.

Post-processing passes
#############################

At the moment, the post-processing passes work a bit differently in :class:`~exa.core.experiment.experiment.Experiment`,
compared to the standalone use of the :class:`~iqm.cpc.compiler.compiler.Compiler`. When running experiments, the
raw data is saved in the database as a binary, and only then sent to the user. It is also possible to extract the data
from the database long after the experiment has been run, using the experiment ``run_id`` and the class method
:meth:`~exa.core.experiment.experiment.Experiment.get_by_run_id`. In that situation, the originally used context is lost.

The first pass, :func:`iqm.cpc.compiler.post_process.construct_run_result` produces a :class:`iqm.cpc.core.run_result.RunResult` object,
which contains not just data, but also all the settings used when preparing the run, and many additional run properties,
which you can also affect by adding passes to the ``job_stage`` which stash relevant objects into the ``RunDefinition``,
to be accessible later. The original ``context`` used to compile the run definition is lost, and the post-processing
passes start with an empty ``context`` dictionary. They may freely add objects to the context and then later reference
them.

You should not modify the first post-processing stage, which just makes the :class:`iqm.cpc.core.run_result.RunResult`.
The second post-processing stage converts all the raw data into convenient derived variables. Which variables are
calculated depends on the :attr:`~exa.core.experiment_options.ExperimentOptions.readout_mode` option. We will discuss it
more in the :doc:`analysis <analysis>` chapter, as this information is needed there to understand what variables are
available for you to use.

The second post-processing stage could in principle be completely replaced for your experiment, however doing it is
definitely quite advanced and only recommended if necessary. Remember that the passes attach to the experiment object,
but recovering the data from the DB via :func:`~exa.core.experiment.experiment.get_experiment` **creates a new object**,
so the modified post-processing passes need to be defaults set in the experiment class. In contrast, the pre-processing
passes could be simply modified by the user when running and prototyping the experiment.

Adding and deleting passes
##########################

Addition of new passes is also detailed in the :doc:`CPC documentation <iqm.cpc:Compilation Stages>`, but let's take
a second on how to do it when writing an :class:`~exa.core.experiment.experiment.Experiment` subclass.

The :class:`~iqm.cpc.compiler.compiler.Compiler` is attached as an attribute :attr:`~exa.core.experiment.experiment.Experiment.compiler`,
and created at initialization. The Experiment also has attributes :attr:`~exa.core.experiment.experiment.Experiment.circuit_stages`,
:attr:`~exa.core.experiment.experiment.Experiment.pulse_stages`, and :attr:`~exa.core.experiment.experiment.Experiment.final_stages`,
which are deepcopied into the Compiler at initialization. You can override those attributes, so that during the
initialization your custom passes are automatically added to the compiler.

Alternatively, you can also modify the passes and stages inside the compiler after the initialization. Simply
override the :meth:`~exa.core.experiment.experiment.Experiment.setup_additional_init_properties`:

.. code-block:: python

    def setup_additional_init_properties(self):

        del self.compiler.circuit_stages.circuit_stage.passes[1]
        self.compiler.circuit_stages.circuit_stage.add_passes(convert_circuit, index=0)
        self.compiler.final_stages.job_stage.add_passes(modify_run_definition)


Compiling abstract circuits
###########################

The :class:`~iqm.cpc.compiler.compiler.Compiler` abstraction allows you to write experiments which take your favourite
quantum language and compile it into IQM data formats. As mentioned above, all you need to do is define additional
circuit-level passes, which will finally result in either a list of :class:`~iqm.pulse.circuit_operations.Circuit` s or
directly a list of :class:`~iqm.pulse.timebox.TimeBox` es.

.. note::
    **Q: Why would I want to use TimeBoxes?**

    A: The IQM :class:`~iqm.pulse.circuit_operations.Circuit` class is simply a flat list of circuit operations,
    assuming the circuit abstraction, i.e. only the order of the gates matters. If your abstract language has a concept
    of otherwise commuting operations happening simultaneously or non-simultaneously, this concept will be entirely
    dropped when converting to :class:`~iqm.pulse.circuit_operations.Circuit`. Such languages are for example
    Cirq or Stim. To maintain this abstraction, you need to not only compile to Circuit differently, but skip the
    class altogether.

To illustrate, consider this abstract draft of a pass converting Cirq Circuit to TimeBox while maintaining the Moment
structure:

.. code-block:: python

    def cirq_to_timebox_with_moments(
        circuits: list[cirq.Circuit],
        components: ComponentGrouping,
        builder: ScheduleBuilder,
    ) -> list[TimeBox]:

        timeboxes = []
        for circuit in circuits:
            boxes = [_map_moment_to_timebox(moment, components, builder) for moment in circuit]
            timeboxes.append(TimeBox.composite(boxes))

        return timeboxes

    def _map_moment_to_timebox(
        moment: cirq.Moment,
        components: ComponentGrouping,
        builder: ScheduleBuilder
        ) -> TimeBox:

        ...

Because the function (not shown here) ``_map_moment_to_timebox`` returns :class:`~iqm.pulse.timebox.TimeBox`, the
later resolution and scheduling passes will respect that each of **those** boxes is a hard-boundary time-space region,
and the gates included in that moment are indeed happening simultaneously, while gates included in different moments are
forbidden to overlap in time.

Then, we would define a compilation stage:

.. code-block:: python

    cirq_stage = CompilationStage(
        name="cirq_stage",
        info="Perform cirq.Circuit-level transformations.",
    )

    cirq_stage.add_passes(cirq_to_timebox_with_moments)

And include this stage in our experiment:

.. code-block:: python

    self.circuit_stages = [cirq_stage]

Of course, this one pass is not enough. In earlier passes, you could for example **transpile** the circuit to IQM
native gates (which you can reference via the builder), or make other logical passes. They will work as long as they
conform to the general structure of a pass function.

Conclusions
###########

In this chapter, we learnt how to affect the default compilation behaviour, control what kind of experiment we are
performing (circuit or pulse-level) and even add our own arbitrary passes. Adding passes at the beginning of
the compilation is equivalent to just making the circuit generation function do more heavy lifting, but it is preferred
if you are designing a whole suite of experiments and want to reuse code.
