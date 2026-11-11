.. _guide_concept:

*****************************************
Conceptualization: What is my experiment?
*****************************************

Idea
####

Any creative process starts from an idea. The idea for a new experiment may come from a paper, conference talk,
a corridor discussion or it might strike from nowhere. This section will help you categorize the idea and understand
how it fits into the framework, so you can use appropriate tools when implementing it. The section is structured into
a series of questions -- it is recommended to answer each and simply keep the answers in mind (or write them down)
for now.


Why? -- Purpose
###############

In this guide, we consider an **experiment** any protocol executed on a quantum chip to extract information. As such,
a resonator spectroscopy is an experiment, but so is a collection of QAOA quantum circuits
equipped with automatic post-processing.

The extracted information might be anything, and what it is defines a **purpose** of the experiment. Most experiments
will fall into one (or more) of the following categories:

1. Characterization experiment - extracts information about the quantum chip or the entire system. Examples include
   relaxation experiments (:math:`T_1`), experiments measuring the coupling between qubits and couplers, measurements
   of the microwave transfer function etc. The numbers measured can later be used to inform the calibration, but are not
   used to modify the calibration or any instrument settings directly. Typically they measure some intrinsic quality
   of the quantum device.

2. Calibration experiment - used to find best parameters for instrument settings or quantum gates, which improve the
   quality of the quantum computer. Examples include :class:`~exa.public_experiments.rabi.Rabi` experiment, Ramsey
   experiment, experiments setting the appropriate idling voltage for the coupler etc.

3. Benchmarking experiment - used to measure the quality of the calibration. Examples include Randomized Benchmarking,
   Floquet benchmarking, Gate Set Tomography, Cycle Benchmarking, Quantum Volume, etc. The information extracted
   depends mostly on the calibration.

It is good to keep in mind the purpose of the experiment, but it does not translate directly into specific practices,
other than the kind of **observations** the experiment produces. Think about why you need this experiment to exist.

Finally, it is possible that your idea is not a suitable candidate for an experiment at all. If you require multiple
sequential steps of measurements to be performed, with data processing and logic in between, it is recommended to use the
:class:`~exa.core.sequence.sequence.Sequence` instead. The experiment classes are building blocks which you will need
to create a Sequence, so continue reading this guide even if that is what you want to do.


What? -- Circuits and sweeps
############################

In exa, the experiments are defined by **circuits** and **sweeps**. The **circuit** is not necessarily the abstract
quantum circuit, only containing gates and measurements -- it is a broader term including low-level pulse schedules.
The sweeps define which experimental parameters need to be varied by default and in what way. Those may be just circuit
parameters, angles of rotation gates, or number of certain "blocks" included in the circuit. Alternatively, they might
be specific instrument parameters, such as DC flux voltage applied to a certain component, or the time AWG waits after
it is triggered. For example, in the Rabi experiment, the circuit consists of a single X gate followed by a measurement,
and there is one default sweep, varying the amplitude of the pulse implementing the X gate.

Think about the main building blocks of your experiment. What does the circuit or schedule look like? Do you need
multiple different circuits, or is one, static circuit, enough as long as it is parametrized? How would you generate
those multiple circuits (this logic will need to be implemented as a sweep!)?

What are the main parameters you want to vary in the experiment? How do you want to vary them: independently, or
in a complicated, dependent fashion, e.g. following a space-filling curve? Conversely, what are the parameters that
should not be varied in your experiment? Exa allows adding arbitrary sweeps to every experiment, but not every sweep
is a good idea!


How? -- Abstraction level
#########################

The experiments might be defined to operate directly on the level of different instrument settings, continuous wave
microwaves, and voltages; or they may be defined by abstract quantum circuits. Generally speaking, characterization
experiments tend to be more low-level, less abstract, and benchmarking experiments tend to be more abstract, but it is
not a strict rule.

1. Pulse-level experiment uses **pulses** and continuous waves directly, has direct access to all the instrument settings and
   makes use of that. Typically you would think of an experiment **schedule** or **playlist** describing the used pulses,
   instead of a **quantum circuit**. Measurements of the TWPA electrical delay or flux pulse distortions are pulse-level
   experiments.

2. Semi-abstract experiment primarily uses **quantum gates** as building blocks, but it might involve some pulses
   acting on the level of the QPU physics as well as some waiting in the "circuit", left there for the free Hamiltonian
   evolution. Those experiments do not generally measure classical system parameters, and can be modelled using cQEC
   Hamiltonians. Measurement of the ZZ-coupling or Floquet benchmarking are semi-abstract.

3. Circuit-level experiment only concerns itself with quantum gates arranged in circuits. The experiment is modelled
   using abstract unitary operators in computational frame and maybe some physical error models. A good criterion
   to distinguish a circuit-level from a semi-abstract experiment is the role of waiting in the schedule - in a
   circuit-level experiment, the waiting is expected to act as identity. Gate Set Tomography is circuit-level.

:class:`~exa.core.experiment.experiment.Experiment` can handle all the abstraction levels, but it is currently most optimized for the semi-abstract ones.
The abstraction mostly matters when designing the **circuit generation function**, which is the main engine generating
subsequent schedules.

.. note::
    **Q: I only know about circuits! I thought all quantum protocols use them. Will I be able to effectively use**
    **exa, which is a pulse-level access tool first? I am not an experimentalist.**

    A: Right now, most of the really pulse-accessy things were abstracted away, and you need to use them only if you
    actually need to use them. This means, you can write your experiment fully thinking in an abstract way, and
    it would work. You may need to dig a bit deeper into IQM packages to find automatic circuit transpilation, if
    that's what you want to do, but for example sampling of random circuits and their inversion based on their unitary
    operators was implemented for randomized benchmarking.


How specific? -- Focus
######################

Some experiment classes are written to do one specific task, and perform it automatically and quickly. Others are
primarily tools used to explore physics. The first need to be more mature as pieces of code, but humans do not really
need to look at the generated data. The latter might be less mature, because their primary function is flexibility.

An example of the former could be an error amplification experiment used to calibrate the virtual Z rotations in the CZ
gate. An example of the latter could be Floquet benchmarking of the coupler leakage during the CZ gate.

Think about how you want your experiment to be used. This will inform how you write your code. An experiment designed
to do one specific thing does not need to be flexible and implement all the possible tricks, but it needs to break
as rarely as possible.
