=========
Changelog
=========

Version 9.21.0 (2025-07-10)
===========================

Bug fixes
---------

- Fix instructions with same field names being treated as equal in building the playlist

Version 9.20.0 (2025-07-09)
===========================

Features
--------

- Enable mypy type checking in CI and add temporary type ignores to the source code. :issue:`SW-1615`

Version 9.19.0 (2025-07-08)
===========================

Features
--------

- Fix bug in playlist visualisation where ndarray samples were not converted for the visualizer in one particular case.

Version 9.18.0 (2025-07-02)
===========================

Features
--------

- Add new MOVE implementation :class:`MOVE_SLEPIAN_CRF`.

Version 9.17.0 (2025-07-01)
===========================

Features
--------

- Faster playlist creation in ScheduleBuilder

Version 9.16.0 (2025-07-01)
===========================

Bug fixes
---------

- Fix type errors raised by mypy.

Version 9.15.0 (2025-06-17)
===========================

Bug fixes
---------

- Convert ``numpy`` types to Python's built-in types so that playlist inspector HTML is rendered correctly. :mr:`1086`

Version 9.14.0 (2025-06-13)
===========================

Features
--------

- Bump iqm-data-definitions

Version 9.13.0 (2025-06-11)
===========================

Features
--------

- Add a base class for shelved readout :class:`Shelved_Measure_CustomWaveforms` and an implementation of shelved readout
  :class:`Shelved_Measure_Constant`. This implementations consist of ``prx_12`` + ``measure`` + ``prx_12`` gates.

Version 9.12.0 (2025-05-30)
===========================

Features
--------

- Bump NumPy to 1.26.4.

Version 9.11.0 (2025-05-28)
===========================

Features
--------

Add information about raised exceptions to more docstrings.

Version 9.10.0 (2025-05-21)
===========================

Features
--------

- Fix cocos path in ruff isort to run isort for cocos correctly.

Version 9.9.0 (2025-05-19)
==========================

Bug fixes
---------

- Add rounding for reset wait gate

Version 9.8.0 (2025-05-12)
==========================

Features
--------

- Update dependency on exa-common

Version 9.7.1 (2025-05-12)
==========================

- Test patch versioning, no functional changes. :issue:`SW-1429`

Version 9.7.0 (2025-04-30)
==========================

Features
--------

- Change deprecated :class:`exa.common.data.parameter.DataType.NUMBER` usage from to ``FLOAT`` or ``INT``.

Version 9.6.0 (2025-04-28)
==========================

Features
--------

- Added a new probe waveform :class:`ProbePulse_CustomWaveforms_noIntegration` which doesn't integrate

Version 9.5.0 (2025-04-25)
==========================

Features
--------

- Add the CompositeGate :class:`RZ_PRX_Composite`, which is a physical Z rotation gate implemented as a sequence of
  PRX gates: RZ(theta) = RY(pi/2) - RX(theta) - RY(-pi/2).

Version 9.4.0 (2025-04-22)
==========================

Features
--------

- Update dependency on exa-common

Version 9.3.0 (2025-04-17)
==========================

Bug fixes
---------

- Fix broken inspect_playlist function. It was missing to add a closing IFRAME tag to the generated HTML code

Version 9.2.0 (2025-04-11)
==========================

Bug fixes
---------

- Update license

Version 9.1.0 (2025-04-10)
==========================

Features
--------

- Fix vulnerability issue with YAML loading, use safe_load to avoid potential harmful remote code execution.
  :issue:`SW-1378`

Version 9.0.0 (2025-04-09)
==========================

Breaking changes
----------------

- Add ``prx_12`` gate in the initial ``QuantumOpTable``, with one implementation ``modulated_drag_crf``.
- Add Baseclass :class:`PRX_ModulatedCustomWaveForms` for arbitrary IQ waveform modulated PRX gate.
- Add gate implementation :class:`PRX_ModulatedDRAGCosineRiseFall` for cosine rise fall modulated PRX gate.

Version 8.13.0 (2025-04-07)
===========================

Features
--------

- Fix package version in published docs footers, :issue:`SW-1392`. 

Version 8.12.0 (2025-04-03)
===========================

Feature
*******

- Format code and enable PEP 604 in linting rules, :issue:`SW-1230`.

Version 8.11.0 (2025-04-02)
===========================

Features
--------

- Added waveforms for I- and Q-envelopes of FAST DRAG and higher-derivative (HD) DRAG: ``HdDragI``, ``HdDragQ``, ``FastDragI``, ``FastDragQ``
- Added PRX implementations using FAST DRAG and HD DRAG: ``PRX_HdDragSX``, ``PRX_HdDrag``, ``PRX_FastDragSX``, ``PRX_FastDrag``

Version 8.10.0 (2025-04-02)
===========================

Features
********

- Update the documentation footer to display the package version.

Version 8.9.0 (2025-03-28)
==========================

Features
--------

- Reworked the way default gates (operations) are defined so they are decoupled from their implementations. This separation allows for the deletion of default implementations without losing information about its designated name. 
- The majority of the original functionality stays the same.
- The ``register_implementation`` function has been split into several different functions to improve readability and testing, as seen below::

    ``register_implementation``
            |
            v
    ``register_gate`` --> ``validate_operation`` --> ``compare_operations`` --> ``add_implementation``
                                                                                    |
                                                                                    v
                                                                          ``validate_implementation`` --> ``set_default``

- The ``build_quantum_ops`` function in builder.py has been split into several functions as well. 
- Trying to modify the implementation class of an existing or default gate implementation yields an error. 



Version 8.8.0 (2025-03-28)
==========================

Features
--------

- Fixing the rounding issue for rise and fall pulses in 'Constant_PRX_with_smooth_rise_fall'

Version 8.7.0 (2025-03-27)
==========================

Features
--------

- :class:`.CouplerFluxPulseQubitACStarkPulseGate` also now supports off-locus RZ corrections.

Version 8.6.0 (2025-03-26)
==========================

Features
--------

- A "gate implementation" ``FluxMultiplexer_SampleLinear`` which can be used to multiplex several flux pulse gate
  TimeBoxes together to cancel flux crosstalk.
- Handle out of locus long-distance ``VirtualRZ`` corrections in CZ gates better
  (scheduling fuses the ``VirtualRZ`` corrections to the right ``IQPulse``).

Version 8.5.0 (2025-03-26)
==========================

Bug fix
-------

- Fix that injecting a new calibration of an array-valued pulse parameter didn't work.

Version 8.4.0 (2025-03-21)
==========================

Features
--------

* Rename QPU chip types, based on either "crystal" or "star" architecture and number of qubits. For example,
  "crystal_5" or "star_6". For "mini" chips, like "mini_crystal_20", the number is not based on the actual number
  of qubits but to the chip it's trying to "minimize" instead, like "crystal_20". :issue:`SW-1059`

Version 8.3.0 (2025-03-19)
==========================

Bug fixes
---------

- Update dependency on exa-common

Version 8.2.0 (2025-03-13)
==========================

Features
--------

- added ``Constant_PRX_with_smooth_rise_fall`` and ``RZ_ACStarkShift_smoothConstant`` pulses, which create a 3-pulse
  schedule, consisting of rise, constant, and fall pulses. These pulses can have arbitrarily long duration, not limited
  by the electronics memory.

Version 8.1.0 (2025-02-28)
==========================


Bug fix
-------
- Bump exa-common

Version 8.0.0 (2025-02-27)
==========================

Features
--------
- Settings refactoring major version
- Updates to documentation.
- Replace deprecated usages of ``DataType.NUMBER`` with either new ``DataType.FLOAT`` or ``DataType.INT``.

Version 7.24.0 (2025-02-20)
===========================

Bug fixes
---------
- When registering an already existing gate with iqm-pulse's register_gate_implementation, the unitary does not need
  to be equal with the previous unitary (it is impossible to check for this, as they are functions...). If no unitary is
  provided, the previous unitary is retained.

Version 7.23.0 (2025-02-19)
===========================

Features
--------

- Bump ``python-rapidjson`` to version 1.20

Version 7.22.0 (2025-02-10)
===========================

Features
--------

- Adds the delay operation :class:`~iqm.pulse.gates.delay.Delay`. :issue:`SW-685`

Version 7.21.0 (2025-02-07)
===========================

Features
--------

- Adds a new special case to :class:`PRX_CustomWaveformsSX` for a PRX rotation angle of zero (i.e., identity), in
  which case a single zero-amplitude pulse with no phase increment is now played. Previously, this special case was not
  separately considered and two X90 pulses with phase increments were used even though this is unnecessary.

Version 7.20.0 (2025-02-04)
===========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 7.19.0 (2025-02-04)
===========================

Features
--------

- Speed up the calculation of unitaries in :class:`.CircuitOperationList` by caching the reshaping function.
- The method :meth:`map_loci` gets an optional argument ``make_circuit``, by default True, which if False will cause the
  output to be a list of :class:`.CircuitOperation` instead of :class:`.CircuitOperationList`. This is faster and often
  the circuit with mapped locus is immediately appended to another circuit or converted to :class:`.TimeBox`, neither
  of which requires the class structure.
- The method :meth:`.ScheduleBuilder.circuit_to_timebox` has an optional argument ``locus_mapping``, defaulting to an
  empty dict. If any of the components in the locus of any :class:`.CircuitOperation` is a key in that dict, it is
  replaced with the value at that key. This speeds up scheduling of identical circuits which only differ by locus.

Bug Fixes
---------
- The :meth:`__add__`, :meth:`__mul__`, and :meth:`__getitem__` of :class:`CircuitOperationList` correctly create the
  new object by assinging the :attr:`qubits` and :attr:`table` attributes at init, and not after, leading to significant
  speedup.

Version 7.18.0 (2025-02-04)
===========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 7.17.0 (2025-02-03)
===========================

Features
--------
- ``CompositeGates`` now cache their TimeBoxes when possible (cache is flushed whenever
  ``ScheduleBuilder.inject_calibration`` is called)
- Add GateImplementation documentation.

Version 7.16.0 (2025-02-03)
===========================

Features
--------

- Add two implementations of the classically conditioned prx gate, both subclasses of the existing :class:`.CCPRX_Composite`,
  which fix the ``prx`` implementation. :class:`.CCPRX_Composite_DRAGCosineRiseFall` fixes it to ``drag_crf`` and
  :class:`.CCPRX_Composite_DRAGGaussian` fixes it to ``drag_gaussian``.

Bug fixes
---------

- The calibration validation only compares the calibration data relating to the gate itself, and not any of its
  registered_gates - those are validated separately anyway whenever they are built. This enables
  :class:`.CompositeGate` s,  which both have their own calibration and registered gates to pass validation.



Version 7.15.0 (2025-01-28)
===========================

Bug fix
-------

- Schedule probe pulses in seconds also when the channel granularity is diffrenet for probe vs. drive/flux.

Version 7.14.0 (2025-01-28)
===========================

Features
--------

- Support broader range of Numpy versions and verify compatibily with Ruff, see the
  `Numpy 2.0 migration guide <https://numpy.org/doc/stable/numpy_2_0_migration_guide.html>`_.

Version 7.13.0 (2025-01-28)
===========================

Features
--------

- For unitary operations, :attr:`QuantumOp.unitary` is a function which takes the operation's params and
  returns the unitary matrix the operation should implement.
- Add a convenience :class:`CircuitOperationList` which is an extension of a builtin list, containing
  ``CircuitOperation`` objects. It can be used to easily construct IQM-compatible circuits through shortcuts of syntax
  similar to qiskit's QuantumCircuit, only defining qubits and other locus elements once, and using a consistent table
  of QuantumOps. It can map the locus to some other locus, and be directly used by the :class:`ScheduleBuilder`
  to create a schedule.
- Add a function :func:`validate_quantum_circuit` extracting the method :meth:`ScheduleBuilder.validate_quantum_circuit`
  so it can be used without the full builder, just using a :class:`QuantumOpTable`.
- Add CZ implementations ``crf_acstarkcrf`` and ``slepian_acstarkcrf`` to the default :class:`QuantumOpTable`.
- Bump exa-common.

Version 7.12.0 (2025-01-27)
===========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-1042`

Version 7.11.0 (2025-01-24)
===========================

Features
--------

* Disable horizontal scroll from playlist visualisation by default.
* Add a toggle to enable/disable horizontal scroll.

Version 7.10.0 (2025-01-17)
===========================

Features
--------

- Added a user guide covering basic concepts and Timebox usage. :issue:`SW-531`

Version 7.9.0 (2025-01-08)
==========================

Features
--------

- Remove gitlab links from public pages. :issue:`SW-776`

Version 7.8.0 (2025-01-02)
==========================

Features
--------

- Fix that using the measure gate on a system without drive lines didn't work. :mr:`SW-514`

Version 7.7.0 (2024-12-30)
==========================

Features
--------

- Bump Station Control Client dependency. :issue:`SW-776`

Version 7.6.0 (2024-12-30)
==========================

Features
--------

- Change license info to Apache 2.0. :issue:`SW-776`

Version 7.5.0 (2024-12-12)
==========================

Features
--------

- Bump exa-experiments

Version 7.4.0 (2024-12-10)
==========================

Bug fix
-------

- Improve documentation structure.

Version 7.3.0 (2024-12-09)
==========================

Features
--------

Fix extlinks to MRs and issues in sphinx docs config :issue:`SW-916`

Version 7.2.0 (2024-12-05)
==========================

Features
--------

- Fix intersphinx reference paths in docs :issue:`SW-916`

Version 7.1.0 (2024-12-04)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-665`

Version 7.0.0 (2024-11-29)
==========================

Breaking changes
----------------
- The function :func:`.apply_move_gate_phase_corrections` no longer uses a calibration set, because the value of the detuning
  needed for phase tracking of MOVE sandwiches is now a part of the MOVE gate calibration data proper.

Features
--------
- Add a parameter ``detuning`` to parent class of all MOVE implementations, :class:`.MOVE_CustomWaveforms`. This
  parameter only affects the frame tracking, and must be set to the difference of the qubit and resonator frequencies.
- Add this parameter to the :class:`.MoveMarker` instruction.

Bug fixes
---------
- Fix the behaviour of U gates: normalization of angle and phases, and pass the correct variable to the schedule
  in case the RY is realized with two or more phased SX pulses instead of one.

Version 6.14.0 (2024-11-27)
===========================

Features
--------

- Add :class:`CouplerFluxPulseQubitACStarkPulseGate` Pulse, which is a base class for AC Stark pulsed CZ gates.

Version 6.13.0 (2024-11-20)
===========================

Features
--------

- ``measure`` and ``reset_wait`` operations now use explicit :class:`.Block` instructions instead of
  :class:`.Wait` s inserted by the scheduling algorithm to idle the qubits, since the former is more
  correct semantically, and will not be disturbed by dynamical decoupling.

Version 6.12.0 (2024-11-19)
===========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-774`

Version 6.11.0 (2024-11-18)
===========================

Bug fixes
---------

- Hard box scheduling no longer uses floats anywhere, and does not leak them into the finished schedule.


Version 6.10.0 (2024-11-15)
===========================

Bug fixes
---------

- Remove iqm-internal web links in customer docs artifacts.

Version 6.9.0 (2024-11-13)
==========================

Bug fixes
---------

- Fix ResetWait gate for computational resonators

Version 6.8.0 (2024-11-12)
==========================

Features
--------
- New quantum operation ``reset_wait`` and its implementation :class:`.Reset_Wait` which is used for resetting qubits
  by waiting a time comparable to the relaxation time.
- All reset implementations now block the common couplers of their locus components.
- Changes / fixes to playlist visualisation:
    - Wait duration common to all channels at the beginning and end of a segment is truncated and shown as its own
      block ("Wait at start/end"), making playlists long waits in the beginning / end more comfortable to view.
    - The timeline axis is no longer shown, as it was broken when instructions are truncated (this
      will be fixed and redisplayed in an upcoming release).

Version 6.7.0 (2024-11-12)
==========================

Bug fixes
---------

- Prefer system fonts in schedule viewer. :mr:`358`
- Fix typos in API docs. :mr:`358`

Version 6.6.0 (2024-11-08)
==========================

Features
--------

- New changelog workflow, no functional changes. :issue:`SW-774`

Version 6.5 (2024-10-31)
========================

Bug fixes
---------
- Fix scheduling neighborhoods in :meth:``.MultiplexedProbeTimeBox.__add__`` (affected only Pulla scheduling)
- Fix probe lines not belonging to settings potentially crashing return parameter discovery


Version 6.4 (2024-10-30)
========================

- Update ``iqm-exa-common`` to version 25.14.


Version 6.3 (2024-10-30)
========================

Bug fixes
---------
- Measure_Constant now throws an error if the integration weights vector lengths do not match the integration_length.
  Previously it would propagate the wrong-length vectors to SC normally, where they would lead to nonsensical errors
  in MCMs & fast feedback
- cc_prx TimeBoxes now work correctly with Pulla's measurement multiplexing step (locus components & neighborhoods
  are adjusted)
- merge_dicts util did not work correctly with empty lists as the default values (e.g. with integration weights)


Version 6.2 (2024-10-28)
========================

Features
--------
- Add implementation :class:`.MOVE_CRF_CRF` for implementing a move operation using cosine rise fall waveform for coupler
  and qubit.
- Add ``crf_crf`` implementation to the default operations both for cz and move.


Version 6.1 (2024-10-28)
========================

- Update ``iqm-exa-common`` to version 25.13 and bump NumPy to version 1.25.2.


Version 6.0 (2024-10-25)
========================

Breaking changes
----------------
- Make fast feedback interface more consistent: feedback_label argument in measure and conditional prx renamed to
  feedback_key (similarly as the measurement key), and conditional_prx has now another argument feedback_qubit, which
  together form the feedback_label "<feedback_qubit>__<feedback_key>"

Features
--------
- In the feedback labels sent to the SC, the feedback key is replaced with a default ``FEEDBACK_KEY``, since the drive
  AWGs do not yet support multiple different feedback labels. Otherwise using fast feedback and/or resets would be
  severely limited in circuits. This will be the HW is improved (hopefully soon).
- Users are no longer able to override default ``QuantumOp`` attributes in ymls, other than the implementations and
  default implementation info



Version 5.9 (2024-10-24)
========================

- Update ``exa-common`` to 25.12


Version 5.8 (2024-10-21)
========================

Features
--------
- Some cleanup of fast-feedback internals in e.g. :class:`.ScheduleBuilder`, including a fix for the scheduling of
  the edge-case of many qubits listening to a single feedback bit in parallel.


Version 5.7 (2024-10-16)
========================

- Add a general quantum operation for reset and a gate implementation :class:`.ConditionalReset`
  for feedback-based reset using on a mid-circuit measure gate followed by a classically-controlled PRX gate.

Bug fixes
---------
- More fixing of fast feedback in many-to-many target-source cases


Version 5.6 (2024-10-16)
========================

- Add a ``measure`` gate implementation named "constant_qnd" for mid-circuit measurement operation.
  This enables optimizing calibration for QNDness and will improve experiments which use many measure gates.


Version 5.5 (2024-10-15)
========================

Features
--------
Rename `phase_increment_before` parameter into PRX(SX) gate into `rz_before`.

Bug fixes
---------
- U gate phase transformation has a wrong sign.



Version 5.4 (2024-10-15)
========================

Bug fixes
---------
- The virtual channels used in fast feedback scheduling no longer block the entire component, allowing more optimal
  schedules.
- The conditional gate :class:`.CCPRX_Composite` now schedules correctly when listening to fast feedback from another
  component to what the conditional flip acts on-


Version 5.3 (2024-10-11)
========================

Features
--------
- Update ``exa-common`` to version 25.11.


Version 5.2 (2024-10-11)
========================

- Add docs for the :func:`.phase_transformation` function.

Bug fixes
---------
- Fixes wrong sign in phase increment calculation.


Version 5.1 (2024-10-11)
========================

- Bump ``scipy`` to 1.11.4.
- Bump ``iqm-data-definitions`` to 2.3 to include documentation of all waveforms.


Version 5.0 (2024-10-08)
========================

Breaking changes
----------------
- A new gate implementation base class :class:`.SinglePulseGate` added, and :class:`.PRX_CustomWaveforms` now inherits
  from this class, which means the ``_single_iq_pulse`` method is renamed to ``_get_pulse`` (this must be done in all
  :class:`.PRX_CustomWaveforms` classes). Otherwise the functionality of the method is the same.
- :meth:`.GateImplementation.construct` removed, and the :meth::meth:`.GateImplementation.__init__` now fulfills the
  same purpose ``construct`` had before (all inits must have the same signature).

Features
--------
- :meth:`.GateImplementation.__call__` now handles ``TimeBox`` caching and the users do not have to
  implement it in every gate implementation separately. Instead you can now just override
  :meth:`.GateImplementation._call` if you are satisfied with caching based on the call arguments.


Version 4.0 (2024-10-02)
========================

Breaking changes
----------------

- :class:`.ConditionalPRX` renamed to :class:`.CCPRX_Composite` which now requires calibration for signal delays.

Features
--------

- a GateImplementation can now return an list of timeboxes, to be used in cases where the relative timing of
  instructions is less strict.
- :class:`.Measure_Constant` now accepts an empty array for the integration weights, signifying constant weigths.
  Use empty array instead of None.
- Canonical waveforms are no longer defined via inheritable class property, but instead by a static whitelist.
  Now you can inherit from a canonical waveform without issues. :issue:`EXA-2112`


Version 3.5 (2024-09-25)
========================

Features
--------
- Add a physical rz operation implemented as a AC Stark pulse.


Version 3.4 (2024-09-23)
========================

Features
--------
- Update ``exa-common`` to version 25.9.


Version 3.3 (2024-09-19)
========================

Features
--------
- :class:`Slepian` waveform now supports squid asymmetry to be taken account of.



Version 3.2 (2024-09-11)
========================

Features
--------
- Added :meth:`GateImplementation.get_custom_locus_mapping` which allows the gate implementations to define their
  own locus mappings, making it possible to write the entire logic of an implementation inside its class definition, :issue:`EXA-1831`
- :class:`GateImplementation` now has a ``bool`` attribute ``special_implementation``, which can be set as ``True``
  if the implementation is a special purpose implementation that should never get called in
  :meth:``ScheduleBuilder.get_implementation`` unless explicitly requested via the ``impl_name`` argument.
  - Special implementations cannot be set as default implementations in :class:`QuantumOp`.


Version 3.1 (2024-09-11)
========================
Features
--------
- Update exa-common.


Version 3.0 (2024-09-06)
========================
Features
--------
- New gate implementations in `PRX` using :math:`\pi/2` pulse: :class:`PRX_DRAGGaussianSX` and
  :class:`PRX_DRAGCosineRiseFallSX`, which inherits from :class:`PRX_CustomWaveformsSX`. The default implementation
  names are `drag_crf_sx` and `drag_gaussian_sx`.
- New U gate :class:`UGate` using composition :math:`Z(\phi)Y(\theta)Z(\lambda)`, with `PRX` gate
  :math:`Y(\theta)`. This gate will use the same implementation as `PRX` gate. The default gate name is `u`, and
  implementation name `prx_u`
- New sqrt(X) gate :class:`SXGate` using fixed pulse :math:`X(\pi/2)`, from `PRX` gate. The default gate name is `sx`,
  and implementation name `prx_sx`.

Breaking changes
----------------
- PRX gate attribute ``x_pi`` and classmethod :meth:`_x_pi_pulse` renamed to ``iq_pulse`` and :meth:`_single_iq_pulse`.


Version 2.13 (2024-09-04)
=========================

Features
--------
- Add `register_implementation` from `exa-core`.
- Make CompositeGate calibration logic more consistent. :issue:`SW-547`


Version 2.12 (2024-08-27)
=========================
Features
--------
- Add the waveforms :class:`.Chirp` and :class:`.ChirpImag`.


Version 2.11 (2024-08-26)
=========================

Features
--------
- Add option to generate a measurement probe pulse without acquisitions.


Version 2.10 (2024-08-26)
=========================

- Update ``exa-common`` to 25.7.


Version 2.9 (2024-08-23)
========================

Features
--------
- The :class:`ScheduleBuilder`'s priority calibration feature is supported now also for factorizable :class:`QuantumOp`s
  (such as the ``measure`` operation), when the locus contains more than one components.


Version 2.8 (2024-08-22)
========================

Features
--------
- :meth:`QuantumOp.get_default_implementation_for_locus` returns the locus-specific default for a permutation
  of a symmetric gate's locus, making the behaviour more consistent with other functionalities of ``ScheduleBuilder``.


Version 2.7 (2024-08-16)
========================

Features
--------
- Add ``FluxPulseGate_CRF_CRF`` fast flux CZ implementation.


Version 2.6 (2024-08-16)
========================

Features
--------
- Update exa-common to 25.6.


Version 2.5 (2024-08-15)
========================

Features
--------

- Bump exa-common to 25.5


Version 2.4 (2024-08-09)
========================

Features
--------
- Added :attr:`QuantumOp.defaults_for_locus` which can be used to set per-locus default implementations of a quantum
  operation.
- If assigned, :meth:`ScheduleBuilder.get_implementation` prioritises the locus-specific defaults over any globally
  defined priorities, :issue:`EXA-1929`


Version 2.3 (2024-08-05)
========================

Features
--------
- :meth:`.ScheduleBuilder.validate_quantum_circuit` now accepts mid-circuit measurements

Bug fixes
---------
- :class:`.ProbePulse_CustomWaveforms` call produces valid integration result labels that have ``"__"`` in them.


Version 2.2 (2024-07-29)
========================

- Automatic disabling of MOVE gate validation for sandwiches with different qubits when phase detuning correction is disabled (COMP-1468).


Version 2.1 (2024-07-23)
========================

Features
--------

- `feedback_signal_label` can be set when getting a TimeBox for a ``measure`` gate.
- Add composite GateImplementation :class:`.CCPRX_Composite`, usable with programmable readout. :issue:`EXA-1925`



Version 2.0 (2024-07-15)
========================

Features
--------

- Add :meth:`.TimeBox.print` as a crude way of visualizing TimeBox contents.
- Add :meth:`.ScheduleBuilder.resolve_timebox`.

Breaking changes
----------------

- Remove placeholder implementation `Measure_NOP`.


Version 1.6 (2024-07-12)
========================

Features
--------
- Bump exa-common to 25.4


Version 1.5 (2024-07-05)
========================

Features
--------
- Bump exa-common to 25.3 


Version 1.4 (2024-07-04)
========================

- Small fix to `validate_move_instructions` function.


Version 1.3 (2024-07-04)
========================

- Bump exa-common to 25.2. :issue:`EXA-2056`


Version 1.2 (2024-07-03)
========================

- Trigger clean pipeline run, no functional changes.


Version 1.1 (2024-07-02)
========================

- Enabled the option to turn off PRX validation for MOVE gate sandwiches (COMP-1468).
- Enabled the option to turn off frame tracking from MOVE gates (COMP-1468).


Version 1.0 (2024-07-01)
========================

Features
--------

- Package `iqm-exa-pulse` is renamed to `iqm-pulse`. No functional changes to `iqm-exa-pulse` version 21.7.
