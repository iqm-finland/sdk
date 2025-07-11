=========
Changelog
=========

Version 8.3.0 (2025-07-09)
==========================

Features
--------

- Enable mypy type checking in CI and add temporary type ignores to the source code. :issue:`SW-1615`

Version 8.2.0 (2025-07-02)
==========================

Bug fixes
---------

- Fix type errors raised by mypy.

Version 8.1.0 (2025-06-18)
==========================

Bug fixes
---------

- Fix get calibration set

Version 8.0.0 (2025-06-13)
==========================

Features
--------

- Move :func:`get_calibration_set_values` logic from Station Control to Pulla.
- Move :func:`create_dynamic_quantum_architecture` logic from Cocos to Pulla.

Version 7.23.0 (2025-06-12)
===========================

Features
--------

- Update dependency on station-control-client, taking updated JobExecutorStatus into use. :issue:`SW-1513`

Version 7.22.0 (2025-06-11)
===========================

Features
--------

- Update dependency on iqm-pulse

Version 7.21.0 (2025-06-02)
===========================

Features
--------

- Update dependency to station-control-client in order to revert changes from :issue:`SW-1513`.

Version 7.20.0 (2025-05-30)
===========================

Features
--------

- Update dependency on station-control-client

Version 7.19.0 (2025-05-28)
===========================

Features
--------

- Raise ``ClientError`` instances instead of standard Python errors from more places where appropriate.

Version 7.18.0 (2025-05-21)
===========================

Features
--------

- Fix cocos path in ruff isort to run isort for cocos correctly.

Version 7.17.0 (2025-05-19)
===========================

Bug fixes
---------

- Update dependency on iqm-pulse

Version 7.16.0 (2025-05-16)
===========================

Features
--------
- task management now uses `JobStatus` instead of `SweepStatus`. :issue:`SW-948`.

Version 7.15.0 (2025-05-12)
===========================

Features
--------

- Update dependency on station-control-client

Version 7.14.1 (2025-05-12)
===========================

- Test patch versioning, no functional changes. :issue:`SW-1429`

Version 7.14.0 (2025-04-22)
===========================

Features
--------

- Update dependency on station-control-client

Version 7.13.0 (2025-04-17)
===========================

Bug fixes
---------

- Fix broken search input window and shortcut

Version 7.12.0 (2025-04-17)
===========================

Features
--------

- Bump `iqm-client`, no functional changes.

Version 7.11.0 (2025-04-11)
===========================

Bug fixes
---------

- Update dependency on iqm-client

Version 7.10.0 (2025-04-11)
===========================

Bug fixes
---------

- Replace static station url to dynamic in line with other example notebooks.

Version 7.9.0 (2025-04-10)
==========================

Bug fixes
---------

- Update dependency on station-control-client

Version 7.8.0 (2025-04-10)
==========================

Features
--------

- Update dependency on iqm-client

Version 7.7.0 (2025-04-09)
==========================

Bug fixes
---------

- Update dependency on iqm-client

Version 7.6.0 (2025-04-09)
==========================

Features
--------

- Bump exa-common and iqm-pulse.

Version 7.5.0 (2025-04-09)
==========================

Bug fixes
---------

- Update dependency on iqm-client

Version 7.4.0 (2025-04-09)
==========================

Features
--------

- Update Cortex CLI to IQM Client CLI in documentation.

Version 7.3.0 (2025-04-07)
==========================

Bug fixes
---------

- Fix docs links to ``iqm.qiskit_iqm``.

Version 7.2.0 (2025-04-07)
==========================

Features
--------

- Enable Pulla usage with IQM Server backends

Version 7.1.0 (2025-04-07)
==========================

Features
--------

- Fix package version in published docs footers, :issue:`SW-1392`. 

Version 7.0.0 (2025-04-04)
==========================

Features
--------

- Replace the old quantum architecture with :class:`DynamicQuantumArchitecture` in :class:`IQMPullaBackend.__init__`.

Version 6.19.0 (2025-04-03)
===========================

Feature
*******

- Enable PEP 604 in linting rules, :issue:`SW-1230`.

Version 6.18.0 (2025-04-03)
===========================

Bug fixes
---------

- Pulla QIR example now correctly remaps qubits

Version 6.17.0 (2025-04-02)
===========================

Features
********

- Update the documentation footer to display the package version.

Version 6.16.0 (2025-04-02)
===========================

Features
--------

- Fix links to client library docs in docstrings.

Version 6.15.0 (2025-03-28)
===========================

Changes
-------

- Bump iqm-pulse

Version 6.14.0 (2025-03-27)
===========================

Features
--------

- Update dependency on iqm-pulse

Version 6.13.0 (2025-03-26)
===========================

Features
--------

- Update dependency on iqm-pulse

Version 6.12.0 (2025-03-25)
===========================

Features
--------

- Update links to ``qiskit_iqm`` documentation.

Version 6.11.0 (2025-03-24)
===========================

Features
--------

* Update dependencies, no functional changes.

Version 6.10.0 (2025-03-21)
===========================

Features
--------

- Bump dependencies.

Version 6.9.0 (2025-03-19)
==========================

Bug fixes
---------

- Update dependency on station-control-client

Version 6.8.0 (2025-03-12)
==========================

Bugfix
--------

- Small issue where extra characters were left in the notebook from SW-1005. :issue:`SW-1005`

Version 6.7.0 (2025-03-11)
==========================

Bugfix
--------

- Only raise a warning when a custom QIR-profile is submitted, such that qiskit circuits
can be converted to QIR and submitted to our devices. :issue:`SW-1005`

Version 6.6.0 (2025-03-10)
==========================

Bump dependencies.

Version 6.5.0 (2025-03-05)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-1015`

Version 6.4.0 (2025-03-05)
==========================

Features
--------

- Remove general RequestError and use new specific error classes instead.
- Use HTTPStatus code names instead of numbers for better clarity. No functional changes.

Version 6.3.0 (2025-03-04)
==========================

Bug fix
-------
- Pulla compiler's station settings are now generated with correct paths.

Version 6.2.0 (2025-03-03)
==========================

- Bump exa-common

Version 6.1.0 (2025-02-28)
==========================


Bug fix
-------
- Bump exa-common

Version 6.0.0 (2025-02-27)
==========================

Features
--------

- Adapt to setting tree reorganization
- Replace deprecated usages of ``DataType.NUMBER`` with either new ``DataType.FLOAT`` or ``DataType.INT``.

Version 5.28.0 (2025-02-25)
===========================

Features
--------

- Fix broken Configuration and Usage guide.
- Bump Qiskit dependencies.

Version 5.27.0 (2025-02-24)
===========================

Bug fix
-------

- Do not attempt to apply dynamical decoupling (DD) sequences on components with virtual drive channels (e.g. computational resonators). This enables DD to be used on stations with star architectures.

Version 5.26.0 (2025-02-24)
===========================

Features
--------

- Remove unintentional section from Quick start guide.

Version 5.25.0 (2025-02-19)
===========================

Features
--------

- Require ``qiskit-iqm >= 17.0`` in the optional ``qiskit`` dependencies.

Version 5.24.0 (2025-02-12)
===========================

Features
--------

- Add :func:`iqm.pulla.utils.calset_from_observations` for converting list of observations into a Pulla calibration set.
  :issue:`SW-905`
- Add example notebook for using locally created calibration set with Pulla. :issue:`SW-905`.

Version 5.23.0 (2025-02-11)
===========================

Features
--------

- Add missing QIR example user guide to HTML rendered docs.

Version 5.22.0 (2025-02-10)
===========================

Features
--------

- Bump ``iqm-pulse``.

Version 5.21.0 (2025-02-04)
===========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 5.20.0 (2025-02-04)
===========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 5.19.0 (2025-02-03)
===========================

* Bump iqm-pulse

Version 5.18.0 (2025-01-30)
===========================

Features
--------

- Display progress bar(s) when user sends circuit for execution using :meth:`.Pulla.execute`. :issue:`SW-881`

Version 5.17.0 (2025-01-30)
===========================

Features
********

- Updated the T1 example notebook to promote a more sensible apporach.

Bug fix
*******

- Fixed that modifying the calibration also modified the original calibration stored on the client-side cache.

Version 5.16.0 (2025-01-28)
===========================

Bump iqm-pulse.

Version 5.15.0 (2025-01-28)
===========================

Features
********
- Bump `iqm-data-definitions`, no functional changes.

Version 5.14.0 (2025-01-28)
===========================

Features
--------

- Bump exa-common and iqm-pulse.

Version 5.13.0 (2025-01-27)
===========================

Features
--------

- Adjust dependencies.

Version 5.12.0 (2025-01-21)
===========================

Bugfix
--------

- Make QIR program example compatible with more devices. :issue:`SW-1056`

Version 5.11.0 (2025-01-15)
===========================

Bugfix
--------

- Replace user-given measurement keys to be safe and compatible with exa-db. :issue:`SW-897`

Version 5.10.0 (2025-01-09)
===========================

Features
--------

- :meth:`.DummyJob.job_id` of the job returned by :meth:`.IQMPullaBackend.run` now returns ``sweep_id`` of the executed
  sweep. :issue:`SW-901`
- :class:`.StationControlResult` returned by :meth:`.Pulla.execute` now contains the ``start_time`` and ``end_time``
  even if the task failed.

Version 5.9.0 (2025-01-08)
==========================

Features
--------

- Remove gitlab links from public pages. :issue:`SW-776`

Version 5.8.0 (2025-01-07)
==========================

Features
--------

- Revoke Station Control task when user aborts Pulla execution. :issue:`SW-899`

Version 5.7.0 (2024-12-30)
==========================

Features
--------

- Update licensing and bump Station Control Client and IQM Pulse dependencies. :issue:`SW-776`

Version 5.6.0 (2024-12-12)
==========================

Features
--------

- Bump exa-experiments

Version 5.5.0 (2024-12-11)
==========================

Features
--------

- Improvements in the example notebooks.

Version 5.4.0 (2024-12-10)
==========================

Bug fix
-------

- Improve documentation structure.

Version 5.3.0 (2024-12-09)
==========================

Features
--------

Fix extlinks to MRs and issues in sphinx docs config :issue:`SW-916`

Version 5.2.0 (2024-12-05)
==========================

Features
--------

- Fix intersphinx reference paths in docs :issue:`SW-916`

Version 5.1.0 (2024-12-05)
==========================

Features
********

- Pulla now support base QIR profile as the circuit definition. :issue:`SW-865`

Version 5.0.0 (2024-12-05)
==========================

Features
--------

- Added :attr:`.StationControlResult.sweep_id` and :attr:`.StationControlResult.task_id`. :issue:`SW-807`

Version 4.8.0 (2024-12-04)
==========================

Features
--------

- By default, :class:`iqm.pulla.cpc.Compiler` can now be initialized with calibration data failing validation.
  :issue:`SW-867`

Version 4.7.0 (2024-12-04)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-665`

Version 4.6.0 (2024-11-29)
==========================

Features
--------

- Adjust the conftest calibration set for NDonis to include the parameter ``detuning`` of all MOVE gate
  nodes, containing the difference of the qubit and resonator frequency.

Version 4.5.0 (2024-11-27)
==========================

Features
--------

- Added :class:`iqm.pulla.utils_qiskit.IQMPullaBackend` allowing to use Pulla as a backend in Qiskit. :issue:`SW-821`

Version 4.4.0 (2024-11-27)
==========================

Features
--------

- Implement Dynamical Decoupling as a standard compilation stage. :issue:`HCS-432`

Version 4.3.0 (2024-11-22)
==========================

Features
********

- Update to the latest station-control-client. :issue:`SW-865`

Version 4.2.0 (2024-11-21)
==========================

Bug fix
-------

- Fix a CircuitExecutionError when submitting a batch of circuits measuring different qubits, with heralding enabled.
  :issue:`SW-880`

Version 4.1.0 (2024-11-19)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-774`

Version 4.0 (2024-11-14)
========================

* ``prepend_reset`` (TimeBox-level) standard compiler stage added (implements both reset by wait and active reset)
* added :attr:``.CircuitExecutionOptions.active_reset_cycles`` that is used to control the reset functionality between.
* :meth`.Pulla.get_standard_compiler` now has an optional argument for overriding default
 circuit execution options


Version 3.0 (2024-11-01)
========================

* Replaced the function :func:`iqm.pulla.utils_qiskit.qiskit_to_cpc` with
  :func:`iqm.pulla.utils_qiskit.qiskit_circuits_to_pulla`, changing the signature.
* Added the function :func:`iqm.pulla.utils_qiskit.qiskit_to_pulla`.
* Updated the user guide.
* Cleaned up the execution results handling.
* Bugfix: ``MeasurementMode.ALL`` now works properly with mid-circuit measurements.
* Require ``iqm-pulse >= 6.5``, ``qiskit-iqm >= 15.0``.


Version 2.1 (2024-10-25)
========================

* ``iqm-pulse`` 6.0 compatibility.


Version 2.0 (2024-10-24)
========================

* See ``docs/migration_guide.rst`` for a detailed migration guide from version 1.x to 2.0.
* Consolidate compiler code under ``iqm.cpc.compiler.compiler`` module.
* Remove ``iqm.cpc.compiler.compiler2``.
* Do not construct qubit mapping and do not connect to CoCoS.
* Remove ``register_fast_feedback`` method. Conditional ``cc_prx`` is now natively supported in CoCoS and Qiskit-on-IQM.


Version 1.8 (2024-10-18)
========================

* Convert ``cc_prx`` args like ``prx``, convert ``measure`` "feedback_key" to "feedback_label" for now.


Version 1.7 (2024-10-09)
========================

* Update ``iqm-pulse`` to 5.0.


Version 1.6 (2024-10-07)
========================

* Add trigger delays, ``twpa.voltage_1`` and ``twpa.voltage_2`` to calset whitelist.


Version 1.5 (2024-10-03)
========================

* ``register_fast_feedback`` now takes feedback signal delays from calibration data.


Version 1.4 (2024-10-02)
========================

* Qiskit is now an optional dependency.
* Qiskit-related utils are moved to ``iqm.pulla.utils_qiskit``. Old import paths are deprecated.


Version 1.3 (2024-09-30)
========================

* User guides updated for Qiskit 1.x.
* Nicer error messages on authentication problems.
* Add Custom gates user guide to the HTML documentation.


Version 1.2 (2024-09-25)
========================

* Compilation passes of the standard stages are now by default idempotent.
* User guide updated with more detailed information on authentication.
* Allow custom initial compiler context dictionary to be passed to :meth:`Compiler.compile`.


Version 1.1 (2024-09-23)
========================

* The dynamical implementations created by ``register_fast_feedback`` are now set as special implementations (protects
  against infinite recursion).


Version 1.0 (2024-09-20)
========================

* See ``docs/migration_guide.rst`` for a detailed migration guide from version 0.x to 1.0.
* Compiler and Pulla are now separated for simplicity.
* Compiler is now always refreshed automatically when needed without user's explicit action.
* Pulla no longer needs to access `/cocos/configuration` endpoint.
* Prevent user from accidentally modifying standard stages.
* Compiler stages are all multipass now.
* User guide split into multiple files.


Version 0.20 (2024-09-18)
=========================

* Support ragged acquisition (acquisition labels no longer need to present in every circuit of a batch).
* Circuits in a batch are no longer padded to the same length.
* Heralding is now done on the :class:`TimeBox` level.
* Change the logic for :class:`MeasurementMode`, controlling the final measurement in a circuit:
  * MeasurementMode.CIRCUIT now measures just the qubits that have ``measure`` gates on them in
    each circuit (previously it measured all the qubits *used* in *any circuit* in the batch).
  * Heralding in MeasurementMode.ALL now performs the heralding measurement (and results filtering)
    only on the qubits used in each circuit (if they *can* be measured, that is). Previously it
    heralded all the qubits used in any circuit in the batch.
* Always send settings to all the probe lines (and TWPAs) on the station, regardless of which
  components are measured in the batch circuits, in order to simplify the settings generation
  logic. This should cause no harm, and typically would happen anyway.


Version 0.19 (2024-09-09)
=========================

* Update to ``iqm-pulse`` 3.0.
* Add fast feedback example notebook.


Version 0.18 (2024-09-03)
=========================

* Fix and rework :class:`CompositeGate` support. Add ``Custom Gates`` example notebook.
* :func:`qiskit_to_cpc` no longer takes backend as argument.
* :func:`qiskit_to_cpc` now accepts a list of circuits.
* Adjust logging to not output debug logs by default.
* Change signature of :meth:`Pulla.add_implementation` to allow any kind of gate.
* Add :meth:`Pulla.register_fast_feedback` as a temporary helper to utilize fast feedback.


Version 0.17 (2024-08-29)
=========================

* Fix front padding of schedules in case instruments have different sampling rates.


Version 0.16 (2024-08-20)
=========================

* Fix result handling of mid-circuit measurements.


Version 0.15 (2024-08-20)
=========================

* Fix failure on null timestamp values


Version 0.14 (2024-08-14)
=========================

* (internal) Rely on chip design record instead of CHADs from station.


Version 0.13 (2024-08-12)
=========================

* Mid-circuit measurement support in the compiler.


Version 0.12 (2024-08-05)
=========================

* Optional ``MOVE`` validation in the compiler.
* Update ``iqm-pulse``.
* Drop support for Python 3.10.
* Drop requirement for ``StrEnum`` package.


Version 0.11 (2024-07-15)
=========================

* Start using programmable readout (functionally identical to CoCoS 29.0).
* Standard compilation stages adapted to programmable readout.
* User guide and Examples updated.
* Add a decorator ``@compiler_pass`` that converts a function to a pass with less boilerplate.
* Remove ``CompilationStage.add_pass`` in favor of ``.add_passes``.



Version 0.10 (2024-07-03)
=========================

* Change dependency of ``iqm-exa-pulse`` to ``iqm-pulse``.
* Change dependency of ``iqm-exa-backend-client`` to ``iqm-station-control-client``.


Version 0.9 (2024-06-28)
========================

* Utility function ``qiskit_to_cpc`` can now handle Qiskit circuits containing custom composite gates.
* GraphQL support is dropped. Calibration data is now fetched only from Station Control.
* Extended logging support. The user can now set the log level.


Version 0.8 (2024-06-20)
========================

* New utility function: ``station_control_result_to_qiskit`` to convert an execution result into a Qiskit result.
* Updated documentation with examples of constructing Qiskit results.
* New optional argument ``complex_readout`` to ``build_settings()`` to set result type to complex.
* Breaking change: ``circuit_operations_to_iqm`` renamed to ``circuit_operations_to_cpc``.
* Breaking change: ``qiskit_to_iqm`` renamed to ``qiskit_to_cpc``.


Version 0.7 (2024-06-18)
========================

* Add an example on how to create T1 Experiment with Pulla.
* Add an example of defining a circuit using IQM Pulse ``CircuitOperation``s directly and compiling it normally.
* New utility function: ``circuit_operations_to_iqm`` to convert a tuple of ``CircuitOperation``s into a
  compiler-compatible circuit.
* New utility function: ``map_qubit_indices`` to replace qubit names in a circuits with their indices from component
  mapping; can be used as a circuit-level pass.


Version 0.6 (2024-06-17)
========================

* Authentication support for connecting to CoCoS and to Station Control Service.
* Mypy type checking in tests and CI.


Version 0.5 (2024-06-17)
========================

* Method ``Pulla.execute()`` returns ``StationControlResult``.
* Method ``Pulla.execute()`` accepts an optional argument ``verbose`` (default: ``False``) to print the measurement
  results.
* Method ``Pulla.execute()`` prints links to the task id and sweep id pages of the Station Control web interface.
* When execution fails, the error from Station Control is propagated and displayed to the user.


Version 0.4 (2024-06-10)
========================

* Station Control is now the default provider of calibration data. GraphQL URL is optional.


Version 0.3 (2024-06-10)
========================

* New utility function: ``locate_instructions`` to find the channel and index of given playlist instructions with
  optional minimum duration.
* New utility function: ``replace_instruction_in_place`` to replace an instruction at a given channel+index with one or
  more other instructions, given that the total durations match.
* New utility functions: ``print_channel`` and ``print_schedule`` to help visualize the playlist instructions per
  channel.
* New notebook ``Examples`` added with an example of using the new helper functions to replace ``Wait``s with arbitrary
  sequences of pulses&waits for dynamical decoupling.


Version 0.2 (2024-06-10)
========================

* GraphQL url is now configurable when loading the configuration from url.
* Automatic fetching of latest calibration set on initialization can be disabled.
* Info about schedule visualization added to the user guide.


Version 0.1 (2024-05-21)
========================

* Initial version.
* Abstract multipass compiler interface and ``STANDARD_STAGES``.
* Basic Qiskit integration.
* Circuit compilation and execution.
* Calibration data provider.
