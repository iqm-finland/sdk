Creating your own EXA experiments
#################################

EXA is a software created by experimentalists for experimentalists --- but you can also use it effectively if you're
a theorist! In this guide we will learn how to create your own automated experiment, beginning at conceptualization
and ending with a fully integrated procedure. Familiarity with using EXA to run experiments will be useful, but
it is only assumed that you know how operating and calibrating superconducting quantum computers works, and ideally
have some experimental experience, not necessarily with this software.

.. note::
    This guide is written in a conversational, casual style, as this is what works best in communication.
    EXA is a flexible and casual but powerful tool, and this guide should reflect that!

    I will include questions and answers from the reader to the author in the following way:

    **Q: This sounds a lot like pulse level access (Pulla) in IQM Client.
    Why are there two similar frameworks and which one should I use?**

    A: You can think of EXA as being built *on top* of the circuit-to-pulse compiler,
    to which IQM Client is just another access point -- you need to
    define everything yourself every time you want to run some quantum circuits. An EXA Experiment is a way
    to combine multiple notebook cells into a single object, which later can just be run, and produce all the
    required output.

We will begin with :doc:`conceptualizing <concept>` the experiment, or answering the question *What does my
experiment do?* You should always start any development by asking yourself this question -- it helps ground the work
you're doing. An experiment on two connected qubits, executable in parallel, using the full power of pulse level access
may need to be written differently than an experiment acting on the whole QPU and defined in terms of abstract quantum
circuits.

Next, we will explore :doc:`prototyping <prototype>` the experiment. The
:class:`~exa.core.experiment.experiment.Experiment` class is
designed to make prototyping easy and fun using a Jupyter notebook or your favourite Python IDE. The code you produce
while prototyping can be easily reused when creating a mature experiment. You should always prototype with a real
system if possible - this is quantum physics we are talking about, and you may discover something unexpected!

After you're happy with your prototype, it's time to write an Experiment :doc:`subclass <subclass>`. Here we will
put the code together into something that does not require setup - just initialize, run and analyze! This section is
self-contained, and you may decide to stop reading the guide there -- but all the building blocks are described later
in more detail.

The first topic to discuss in depth is the :doc:`circuit generation functions <circuit>`, which are the
main engine of the experiment, producing pulse schedules or abstract circuits which together define it.
We will talk about different kinds of "circuits" that can be generated, from the most abstract to the bare pulse
schedules, discuss tips and tricks and even dive into :py:mod:`iqm.pulse`.

Next, we will discuss adding default :doc:`sweeps <sweep>` to the experiment, and how to make them work
in different scenarios, especially with experiments meant to run in parallel. We will look into different types of
sweeps and how to think about them - those modifying or selecting a circuit, those affecting the gates in the circuit
and those affecting global calibration settings.

We will briefly talk about :doc:`compiler passes <compiler>` next. EXA is built on top of the circuit-to-pulse compiler
(CPC), with fully adjustable compilation passes from abstract circuit definitions all the way to the low level
description of the quantum job. We will learn how to adjust, add and remove the passes as necessary for the experiment
you're writing.

After the experiment is run, it is time to :doc:`analyze <analysis>` the collected data. We will learn about
proper post-processing, data handling and using :py:mod:`xarray` effectively -- especially when it comes to supporting extra
sweeps and maintaining the flexibility of the Experiment framework.

Even though the :doc:`plots <plotting>` and :doc:`observations <observations>` are generated in the
analysis, they deserve their own sections describing how to use them and discussing best practices. Both plots and
observations are communication tools -- plots are used to communicate with humans, other experimentalists looking at
your data, and observations are used to communicate between parts of the series of experiments.

After the experiment subclass is ready, it is time to :doc:`integrate <integration>` it in a larger context.
In this section, we discuss what features the experiment needs to be properly integrable, and explore
sequences :class:`~exa.core.sequence.sequence.Sequence`, leaving the documentation of the Graph-Based Calibration to the
appropriate place.

Finally, we will create a checklist of how :doc:`mature <maturity>` the experiment is, and provide rules
of thumb to decide if making the experiment more mature is worth it.

Welcome to EXA and remember to have fun!

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Conceptualization <concept>
   Prototyping <prototype>
   Making a subclass of Experiment <subclass>
   Circuit generation functions <circuit>
   Sweep generation functions <sweep>
   Compiler passes <compiler>
   Post-processing and analysis <analysis>
   Plotting <plotting>
   Observations <observations>
   Integrating your experiment in a sequence or graph <integration>
   Is my experiment mature? <maturity>
