Common errors
#############

This document describes some common errors which you may encounter when using Pulla, and ways to fix them.

Granularity mismatch
--------------------

If you have modified any timing aspect of the schedule, you may encounter an error like this::

    Instruction length of 36 doesn't match the QB1__drive.awg granularity of the device

The granularity is an instrument limitation. Specifically, the part (e.g. HDAWG) which plays the drive pulses.
Some instrument may use e.g. a sample rate of 2.4 GHz, and each instruction duration must be an integer multiple of 16 
samples. Instructions therefore must be a multiple of 6.666ns (16 / 2.4 GHz).
Durations that are within 0.005 samples of an allowed number of samples are rounded to that 
(the software assumes it is what the user meant), otherwise an error is raised.

Different stations have different instruments.
IQM instruments use a fixed sample rate of 2 GHz, and have a granularity of 8 samples.

You can view the granularity information in channel properties, for example
``Pulla.get_channel_properties()[0]['QB1__drive.awg']``::

    ChannelProperties(sample_rate=2400000000.0, instruction_duration_granularity=16, instruction_duration_min=16, compatible_instructions=(), is_iq=True, is_virtual=False)

If you were to turn off the error and use an instruction with a duration that does not fit the granularity, it would
just silently be extended to the next longest allowed granularity, and would not do what you expect.

