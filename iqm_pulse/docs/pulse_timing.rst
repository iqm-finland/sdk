Pulse timing
############

Measure and ReadoutTrigger
--------------------------

The :class:`~iqm.pulse.playlist.instructions.ReadoutTrigger` Instruction responsible of qubit readout has several
timing-related attributes.
The ``measure.constant`` gate implementation produces a ReadoutTrigger instruction
from a simplified set of parameters.
The figure below shows how these parameters relate to the more flexible attributes of the instruction.

.. image:: /_static/images/readout_timing.svg

Fast feedback timing
--------------------

With conditional instructions, we can specify how the output from readout operations should affect
other instructions in the same Segment.
Usually, the goal is use the information as soon as possible, but it takes a finite time to propagate from the
acquisition unit to the AWG that execute the Instructions conditionally.

.. note::

    On all hardware supported by IQM QCCSW, :class:`.ConditionalInstruction` reads the signal bit at the time of
    excution, regardless of when the signal bit was last updated.
    This means that if the Conditionalinstruction is executed too early, the condition will be executed based on the
    previous state of the bit.

To facilitate efficient timing of the feedback signals, IQM Pulse uses virtual channels between probeline channels
(the source of the signals) and drive channels (the destinations).
Block instructions on the virtual channel represent the travel time of the signals.

:class:`.CCPRX_Composite` is a GateImplementation of the ``cc_prx`` (classically controlled PRX) gate that outputs two
TimeBoxes:
the first one to represent the travel time, and the second one with the actual :class:`.ConditionalInstruction`.
In typical use, both should be scheduled in the same order, to ensure the Conditionalinstrucion starts when the
signal bit is available.

The following image illustrates how the TimeBoxes are used for qubits ``QB2`` and ``QB3``.
For QB2, this is also how :class:`.Reset_Conditional` implements the ``reset`` operation.

The equaivalent code would be

.. code-block:: python

    measure = builder.measure(["QB2", "QB3", "QB4"])(feedaback_key="A")
    reset_qb2 = builder.cc_prx(["QB2"])(feedback_qubit="QB2", feedaback_key="A")  # 2 timeboxes, use both
    set_qb3_to_1 = builder.cc_prx(["QB3"])(feedback_qubit="QB3", feedaback_key="A") # 2 timeboxes, use both
    cc_prx_qb4 = builder.cc_prx(["QB4"])(feedback_qubit="QB4", feedaback_key="A") # 2 timeboxes, use 2nd only

    prx_qb3 = builder.prx(["QB3"]).rx(np.pi)
    prx_qb4 = builder.prx(["QB4"]).rx(np.pi)
    wait = builder.wait(["QB4"], 80e-9)

    circuit = measure + reset_qb2
    circuit += prx_qb3 + set_qb3_to_1
    circuit += prx_qb4 + prx_qb4 + prx_qb4 + wait + prx_qb4 + cc_prx_qb4[1]


.. image:: /_static/images/feedback_timing.svg


Instructions are spaced out in time only for visual clarity. When scheduled ASAP, they would be left-aligned
such that the ConditionalInstructions start right after the associated ``control_delay`` has passed.

The bottom of the image illustrates an alternative use of ``CCPRX_Composite`` to have more freedom in the timing.
There, the optional delay TimeBox is not used for scheduling the Instructions on QB4.
Instead, the user has ensured that the other instructions take enough time for the signal to arrive.
This could be used to act on the *previous* feedback signal (not shown).


.. note::

    This section is not about IQM Pulse itself, but might help in understanding the details of the execution.

The image below shows a typical timing of a Playlist segment with 2 AWG devices for driving, and a readout instrument.
Here, all statements that apply to an AWG apply to readout instruments as well.
The AWGs can output an arbitrary sequence of pulses, and the readout instrument can additionally read out
the response to the pulses.

With readout, the raw signal response from the readout pulse will be integrated to produce a single number, such as a
complex number or a bit, corresponding to a particular qubit in a particular segment.

In the figure, one of the AWGs has been selected as the trigger master, which means it sends trigger pulses to
start the execution on the slave devices.
As shown in the picture, different delays caused by the travel time of signals can be compensated for by
adjusting the ``trigger_delay`` setting of each device.


.. image:: /_static/images/pulse_timing.svg


Settings in the figure that can be adjusted by user in the higher level libraries:

.. list-table::
   :class: full-width
   :widths: 20, 40
   :header-rows: 1

   * - Setting
     - Explanation
   * - <awg>.trigger_delay
     - Wait time between the end of the trigger signal of the AWG master and the beginning of the pulse sequence.
   * - <awg>.trigger_delay (slave)
     - Wait time between receiving the trigger signal at the AWG slave and the beginning of the pulse sequence.
   * - options.end_delay
     - Wait time between the end of the pulse segment and the next trigger.
   * - <gate>.<implementation>.<locus>.duration
     - The duration of the hardware instruction for a gate, possibly rounded to satisfy granularity constraints.
       For the ``ReadoutTrigger`` instruction, the meaning is different, see below.

Other notes:

* The AWG spcecified by ``options.trigger_master`` is the only channel that does not wait for a trigger
  at the start of a segment.
* Slave AWGs may also emit a trigger pulse to allow daisy chaining trigger signals.
* Systems with IQM Control System are triggered centrally and the channels run independently, and the
  ``options.trigger_master`` has no effect.
* Pipeline delays are delays between the execution of a command and the pulse actually getting outputted
  from a device. This delay is caused by the hardware and cannot be changed.
  In practice, it can be thought as being part of the cable delays, and thus can be compensated with
  the ``trigger_delay`` setting.
