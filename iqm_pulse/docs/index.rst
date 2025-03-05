*********
IQM Pulse
*********

:Version: |release|
:Date: |today|

IQM Pulse provides an abstraction that transforms high-level quantum circuit operations
to a unified set of lower-level instructions for `IQM <https://meetiqm.com/>`_ quantum computers.

A quantum circuit is an abstract mathematical construct which conveniently hides all implementation
details such as the timing of microwave pulses, waveform shapes, sampling rates, signal capture, and so on.
But in order to execute a circuit you need to convert it into a schedule of hardware instructions which involve
all of the above.

IQM Pulse provides

* a framework for defining abstract quantum gates/operations, as well as their concrete implementations in terms of hardware instructions
* machinery to easily construct circuit-level gate sequences, and compile them into instruction schedules.
* a set of ready-made gates with implementations.

IQM Pulse is not a standalone tool, but is used in IQM's client libraries, IQM Pulla and Exa.
To use them effectively, you are encouraged to familiarize yourself with IQM Pulse, especially the most common
concepts.

Contents
========

.. toctree::
   :maxdepth: 2

   concepts
   using_builder
   custom_gates
   pulse_timing
   API

.. toctree::
   :maxdepth: 1

   Changelog <changelog>
   references
   License <license>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
