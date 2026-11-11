.. _sequence:


=========
Sequences
=========


Sequences are a standardized way to execute arbitrary sequences of Experiments.
The main use case is to standardize commonly repeated scripts that are too complicated for a single Experiment,
and to generate reports automatically.

The :mod:`exa.core.sequence` module defines the framework.

Features
--------

The main principles of sequences are:

- A sequence allows arbitrary control over execution nodes (experiments or subsequences) with loops and conditionals.
- The executions and analyses of the nodes are tracked such that the results
  can be retrieved at later stage based on only the unique ID of the execution event.
- The end results of a sequence (plots, observations, report) are deferred and can be generated at analysis stage.
- The inputs to a sequence must be simple such that a sequence can be launched from e.g. command line.
  While a sequence can be written to accept many options such as sweep ranges and settings,
  the recommended way is to rely on the configuration file to pass the information.

The data model of sequences is roughly the following:

1. A subclass of :class:`~exa.core.sequence.sequence.Sequence` defines a certain control flow of sequence nodes, which
   are either Experiments or other (sub)sequences.
   Sequences can be nested and mixed with Experiments arbitrarily.
   The sequence in which the nodes are executed is defined as arbitrary Python code in the
   :meth:`~exa.core.sequence.sequence.Sequence.execute` method.

2. One execution event is recorded using a context manager from
   :meth:`Sequence.context <exa.core.sequence.sequence.Sequence.context>`.
   The event is assigned a unique identifier.
   The context manager records the execution of all nodes, their inputs and outputs.
   The execution of an Experiment node means running the Experiment on hardware, analyzing the result, and saving the
   resulting observations.
   The results of the previous nodes are available in the context, and can be used in the control flow logic.

3. The output of the :meth:`~exa.core.sequence.sequence.Sequence.execute` is a
   :class:`~exa.core.sequence.result.SequenceExecuteResult`, which contains the results of
   the nodes and the context information. It is equivalent to
   :class:`~iqm.cpc.core.run_result.RunResult` of Experiments.

4. Sequence also defines a function to analyze the result of the whole sequence.
   :meth:`SequenceExecuteResult.analyze <exa.core.sequence.result.SequenceExecuteResult.analyze>` calls this function
   to produce a :class:`~exa.core.sequence.result.SequenceAnalyzeResult`,
   which is the end result of the process. The SequenceAnalyzeResult can be used to plot the results,
   generate a report, and save sequence-level observations, similar to
   :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult` of Experiments.

5. Sequence execution results are stored in a structure of Pydantic models in :mod:`exa.core.sequence.interface` which
   are serialized and stored to persistence.
   Any exceptions happened during sequence execution will also be stored, and will halt further sequence execution.
   Analysis results and exceptions during analysis are stored in
   :class:`~exa.core.sequence.interface.SequenceAnalyzeResultData` which will not be serialized nor saved to
   persistence. Analysis results are obtained during normal sequence execution, or when loading back from persistence.

.. note::

    Sequence is not yet supported when connected through IQM Server.


Usage
-----

Defining a Sequence
~~~~~~~~~~~~~~~~~~~~~~~~

An example sequence might look like this

.. code-block:: python

    class MySequence(Sequence):
        name = "my_sequence"
        analysis = staticmethod(my_analysis_function)
        default_options = {"qubit": "QB1"}
        ignore_analyze_errors = True

        def execute(self) -> SequenceExecuteResult:
            qubit = self.options["qubit"]
            with self.context() as context:
                result = context.execute_sequence(SomeCalibrationSequence(qubit=qubit))
                if is_acceptable(result):
                    context.execute_experiment(T1, run_args={"qubit": qubit})
            return context.to_result()

The ``default_options`` defines that the keyword ``qubit`` must be defined when using ``MySequence``.
When executed, it runs ``SomeCalibrationSequence`` as a subsequence and, depending on the result,
runs the ``T1`` experiment.
The lines ``with self.context() as context:`` and ``return context.to_result()`` are the mandatory ways to begin and
end the session. The attribute :attr:`~exa.core.sequence.sequence.Sequence.ignore_analyze_errors`, set ``True`` here,
allows one to continue sequence execution regardless of any errors happening in the analyze functions of the sequence
itself or in any of its nodes. Unless overridden like this, its value is by default ``False``.

``my_analysis_function`` could look like this:

.. code-block:: python

    def my_analysis_function(result: SequenceExecuteResult, some_arg: float = 1.0) -> SequenceAnalyzeResult:
        ...
        # Read result and compute dataset, observations...
        ...
        return SequenceAnalyzeResult(execution_result=result, dataset=..., observations=...)


One can also provide custom analysis functions for an experiment that will then be called instead of the experiment's
native analyze method. The custom analyze function should take an experiment instance as its first argument
(the rest of the arguments are for analyze kwargs), and return an :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult`
object. The custom analyze could for example first call the experiment's native analyze and then do additional manipulations:

.. code-block:: python

    def my_first_custom_analysis(experiment: Experiment, **kwargs) -> AnalyzeResult:
        result = experiment.analyze(**kwargs)
        result.extra["cool_stuff"] = "Mom, look at me doing custom analyses like a pro!"
        return result

    class MySequence(Sequence):
        name = "my_sequence"

        def execute(self) -> SequenceExecuteResult:
            with self.context() as context:
                result = context.execute_experiment(MyExperiment(), custom_analysis=my_first_custom_analysis)
            return context.to_result()


Running a sequence
~~~~~~~~~~~~~~~~~~~~~~

``MySequence`` would be constructed to operate on ``QB12`` and executed like this:

.. code-block:: python

    my_sequence = MySequence(qubit="QB12")
    result = my_sequence.execute()

When the result is analyzed, the ``my_analysis_function`` will be called with given arguments:

.. code-block:: python

    analyzed = result.analyze(some_arg=2.5)  # my_analysis_function will be called with (some_arg=2.5)
    analyzed.render_report()
    analyzed.observations.save()

Sequences can be run on an :class:`~exa.core.observation.observation_stash.ObservationStash` object instead
of accessing the DB when loading/saving observations. This happens simply by providing the (already initiated) stash in the argument
``observation_stash`` in the sequence's constructor:

.. code-block:: python

    MySequence(observation_stash=my_stash)

What then happens internally in sequence is that all experiment execution nodes will load and save their observations
using the given stash. If the sequence contains any subsequences, they will be by default run on a copy
of the parent's stash, and at the end of the subsequence execution, the copied stash's new observations will be merged
to the parent (or if the parent does not have a stash, the new observations will be saved to the DB). This merging of
stashes can be controlled by the :attr:`~exa.core.sequence.sequence.Sequence.save_stash`.

User can also provide a custom stash for the experiments and subsequences in the Sequence's definition:

.. code-block:: python

    def execute(self) -> SequenceExecuteResult:

        with self.context() as context:
            experiment_stash = some_stash_creation_function()  # whatever stash init and manipulation code here
            context.execute_experiment(MyExperiment(), observation_stash=experiment_stash)

            subsequence_stash = other_stash_creation_function()
            subsequence = MySubsequence(observation_stash=subsequence_stash)
            context.execute_sequence(subsequence)

        return context.to_result()


Sequence options
~~~~~~~~~~~~~~~~~~

All kwargs given as the Sequences init (apart from ``observation_stash``) will end up in the Sequence's attribute
``options`` (Note: these must be basic python types for serialisation reasons). It is also possible to set
the options of any execution nodes from the parent sequence's options by specifying a corresponding option name with
a prefix that consists of the subsequence and/or experiment class names. For example, consider a sequence that contains
a subsequence (class) ``MySubSequence`` which further executes experiment of class ``MyExperiment``. Then by providing
the option ``"MySubSequence_MyExperiment_some_option"`` in the parent sequence's init, one can set the value of
``some_option`` in the execution of ``MyExperiment`` inside the subsequence.


Loading sequence data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function :func:`~exa.core.sequence.data_access.load_sequence_result` can be
used to load back results by sequence UUID. When loading, all nodes within the main sequence are loaded recursively
and analyzed with their respective analysis arguments.

Sequences are saved to persistence incrementally after each experiment or subsequence execution. Execution and analysis
along with their respective results are separate data by design. Only the data necessary to reconstruct the sequence
execution results are saved. The data structure is represented by Pydantic models in :mod:`exa.core.sequence.interface`
and versioned by :data:`exa.core.sequence.interface.SERIALIZER_VERSION`. Thus, the loading code has to have matching
serializer version with the code that saved the results.


Tracker classes
-------------------

:class:`~exa.core.sequence.experiment_tracker.ExperimentTracker` classes are an advanced feature for situations where
more customization is needed than
:meth:`~exa.core.sequence.context.SequenceContext.execute_experiment` allows.
A tracker wraps any Experiment such that it can be used like a normal Experiment instance while also tracking the
inputs and outputs.

In the above example, the conditional execution could be written and extended as:

.. code-block:: python

     if is_acceptable(result):
        exp = context.track(T1)
        exp.options.iterations = 1000  # modify instance as needed
        settings = exp.get_settings(...)
        ...
        exp.run(qubit, settings)
        exp.analyze()
        result = context.end_tracking(exp)
        result.observations.validate()
        result.observations.tag(...).save()

In fact, the ExperimentTracker is always used under the hood.

Finally, users are able to change the ExperimentTracker class a Sequence will use. By changing the class with their own
implementation, it is possible to further modify the behavior of the wrapped experiments.
For example, the :class:`~exa.core.sequence.trackers.retry_on_run_failure.RetryOnRunFailure` tracker
retries to run any Experiment in case of a spurious error.

