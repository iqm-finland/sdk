# Changelog

## Version 13.0.2 (2026-04-23)

### Features

- Prepare package for `4.5.1` release. No functional changes.

## Version 13.0.1 (2026-04-09)

### Breaking changes

- The signature of `create_dynamic_quantum_architecture` has slightly changed.
  It now follows the default implementation observations in the calset.
  If they are missing, falls back to the old behavior.
- CPC now uses the default implementation observations in the calset to pick
  implementations when they are not defined by the user. The old
  implementation priority order does not exist anymore.
- Remove old Pulla and CPC and replace them with the new "Pulla MQE" classes & tools.

### Bug fixes

- CPC's `CircuitExecutionOptions` are now validated such that active reset
  cannot be used with the option
  `CircuitExecutionOptions.convert_terminal_measurements = True` as
  `measure_fidelity` is not optimized for leakage.
  In that case we set
  `CircuitExecutionOptions.convert_terminal_measurements = False` and throw
  a warning in the compilation

- Fixed wrong-type argument in Setting instantiation
    in `SettingNode.set_from_dict`
- `add_stage_setting_options` now only assigns a source to Settings that
  have a non-None value.
- `CompositeGate.customizable_gates` now properly determines which member gates
  should appear in the settings tree under CompositeGate calibration data.
- Fix UI issues of `Compiler.show_stages` for some browsers.
- Fix float return data (encountered when doing HW averaging) being cast
  to zeros.

### Features

- The component frequency setting stored under
  ``settings.characterization.model[component]`` is not only added for
  computational resonators, but also for qubits in case they do not have a drive
  controller.
- Refactor DD utility.
- Add dependencies needed by Pulla+MQE to `iqm.cpc.core` from `exa-core`.
- Fixing more Pulla notebooks
- Some minor `Compiler` UI improvements for making it more comfortable to
  run individual stages.
- Pulla Compiler can now also sweep default_implementation in
    `settings.gate_definitions` nodes
- The settings tree can now be updated with the values of the
    sweep spot, if enabled
  - New compiler option `"update_settings"` controls this
      behavior (by default `False`)
- Finish updating rest of Pulla notebooks.
- Docs, typing, linting fixes.
- Added `.Pulla._debug_info`, which collects useful information to include
  in bug reports and customer support requests.
- Pulla now uses `iqm-client` instead of `iqm-station-control-client` to connect
  to the server.
- Add TRIGGER_INDEX parameter
- Fixing Pulla notebooks
- Put back the utility function `sweep_job_to_qiskit` into Pulla
  Qiskit utils. This replaces the obsolete `station_control_result_to_qiskit`.
- Rewrite `apply_over_coordinate` to be more efficient using `apply_ufunc`.
  - This should bring a performance improvement,
    ranging from 1.5x to 20+x depending on the size of the data,
    the number of coordinates, and how hard the function to apply is.
- Add observation parameter for F0G1 pulse stark shift polynomial coefficients
- Add default gate characterization property for `lru` gate

## Version 12.0.0 (2025-11-19)

### Breaking changes

- Update `Pulla` to use IQMServer in REST communication instead of Station Control service.
- Many models have been removed from Pulla and similar/same models are used from station control interface models.
  For example, `TaskStatus`, `DDMode`, `DDStrategy`, `HeraldingMode`, `MoveGateFrameTrackingMode`,
  and `MoveGateValidationMode`
- `CalibrationSet` model is renamed to `CalibrationSetValues`.
- `calibration_set` parameter is removed in many classes/methods/functions, use `calibration_set_values` instead.
  For example, for `find_observation`, `get_standard_compiler`, `Compiler`, etc.
- `get_calibration` method in `Compiler` is removed, use `get_calibration_set_values` instead.
- `set_calibration` method in `Compiler` is removed, use `set_calibration_set_values` instead.
- `get_calibration` method in `CalibrationDataProvider` is removed, use `get_calibration_set_values` instead.
- `station_control_url` init parameter for `Pulla` is removed, use `iqm_server_url` instead.
- `get_token_callback` init parameter for `Pulla` is removed, use `token` instead.
- `fetch_latest_calibration_set` method in `Pulla` is removed, use `fetch_default_calibration_set` instead.
- `fetch_calibration_set_by_id` method in `Pulla` is removed, use `fetch_calibration_set_values_by_id` instead.
- `execute` method in `Pulla` is removed, use `submit_playlist` instead.
- `station_control` init parameter for `CalibrationDataProvider` is removed, use `iqm_server_client` instead.
- `get_latest_calibration_set` method in `CalibrationDataProvider` is removed,
  use `get_default_calibration_set` instead.
- `station_control_result_to_qiskit` function in `utils_qiskit` is removed, use `sweep_job_to_qiskit` instead.

### Features

- Minimal changes to support Qiskit v2.0 and v2.1.
- Updated QIR User Guide example to include a note about `iqm-qiskit-qir` not supporting Qiskit v2.x.
- Removed `iqm-qiskit-qir` from the QIR requirements as the package does not support Qiskit 2.X
- Allow Python 3.12 in project configuration.
- Verify unit testswith Python 3.12

### Bug fixes

- CPC calibration set whitelist is updated to include various missing controller settings
  that already were in the calibration set.

## Version 11.16.0 (2025-10-22)

### Bug fixes

- Complex return data is processed correctly

## Version 11.15.0 (2025-10-15)

### Bug fixes

- Correctly skips the addition of DD pulses if leading and trailing waits are flagged as True

## Version 11.14.0 (2025-10-10)

### Features

- More Pulla backwards compatibility when the measure_fidelity gate is missing

## Version 11.13.0 (2025-10-09)

### Features

- Pulla backwards compatibility when the measure_fidelity gate is missing.

## Version 11.12.0 (2025-10-09)

### Features

- Update dependency on iqm-client

## Version 11.11.0 (2025-10-09)

### Bug fixes

- Update dependency on iqm-pulse

## Version 11.10.0 (2025-10-08)

### Features

- Added more efficient result converters to `iqm.pulla.utils`.
- Deprecated `convert_sweep_spot` and `convert_sweep_spot_with_heralding_mode_zero` in
  favor of the more efficient ones.

## Version 11.9.0 (2025-10-06)

### Features

- Add `dut_label` to `StaticQuantumArchitecture`.
- Update dependencies.

## Version 11.8.0 (2025-10-03)

### Bug fixes

- Skip following mypy imports to iqm-data-definitions until errors are fixed.

## Version 11.7.0 (2025-09-30)

### Features

- Update dependency on station-control-client

## Version 11.6.0 (2025-09-19)

### Bug fixes

- Bump iqm client dependency

## Version 11.5.0 (2025-09-17)

### Features

- Update dependency on iqm-pulse

## Version 11.4.0 (2025-09-12)

### Features

- Update dependency on exa-common

## Version 11.3.0 (2025-09-12)

### Features

- Update dependency on station-control-client

## Version 11.2.0 (2025-09-12)

### Features

- Allow executing jobs in async fashion

## Version 11.1.0 (2025-09-12)

### Features

- Postprocessing stages.

## Version 11.0.0 (2025-09-11)

### Breaking changes

- Replace `iqm.cpc.interface.compiler.Circuit` with `iqm.pulse.Circuit`.

- Remove `Instruction` and `Circuit` models from `iqm.pulla.interface`.

## Version 10.1.0 (2025-09-08)

### Features

- bump iqm-pulse version

## Version 10.0.0 (2025-09-08)

### Features

- Pulla compiler converts the terminal measurements into "measure_fidelity"
  - new circuit execution option `"convert_terminal_measurements"` controls this behavior (by default `True`)

## Version 9.9.0 (2025-09-03)

### Features

- Enable ruff rule for missing annotations and mark exemptions.

## Version 9.8.0 (2025-08-21)

### Bug fixes

- FIx Pulla.execute crashing when there are jobs in the queue

## Version 9.7.0 (2025-08-20)

### Features

Fix qiskit-to-pulla tests so they don't initialize IQMBackend with metrics.

## Version 9.6.0 (2025-08-20)

### Bug fixes

- Add explicit cross-component requirements

## Version 9.5.0 (2025-08-11)

Feature

- bump exa-experiments and iqm-pulse

## Version 9.4.0 (2025-08-07)

### Features

- Fix naming conventions related to Pulse level access as `Pulla`.

## Version 9.3.0 (2025-08-05)

### Features

- Require compiler instead of Pulla in qir_to_pulla

## Version 9.2.0 (2025-07-31)

### Features

- Support iqm-client with QIR support

## Version 9.1.0 (2025-07-30)

### Features

- Require compiler instead of Pulla in qir_to_pulla

## Version 9.0.0 (2025-07-16)

### Breaking changes

- `Compiler.add_implementation` signature has slightly changed.

## Version 8.3.0 (2025-07-09)

### Features

- Enable mypy type checking in CI and add temporary type ignores to the source code.

## Version 8.2.0 (2025-07-02)

### Bug fixes

- Fix type errors raised by mypy.

## Version 8.1.0 (2025-06-18)

### Bug fixes

- Fix get calibration set

## Version 8.0.0 (2025-06-13)

### Features

- Move `get_calibration_set_values` logic from Station Control to Pulla.
- Move `create_dynamic_quantum_architecture` logic from Cocos to Pulla.

## Version 7.23.0 (2025-06-12)

### Features

- Update dependency on station-control-client, taking updated JobExecutorStatus into use.

## Version 7.22.0 (2025-06-11)

### Features

- Update dependency on iqm-pulse

## Version 7.21.0 (2025-06-02)

### Features

- Update dependency to station-control-client.

## Version 7.20.0 (2025-05-30)

### Features

- Update dependency on station-control-client

## Version 7.19.0 (2025-05-28)

### Features

- Raise `ClientError` instances instead of standard Python errors from more places where appropriate.

## Version 7.18.0 (2025-05-21)

### Features

- Fix cocos path in ruff isort to run isort for cocos correctly.

## Version 7.17.0 (2025-05-19)

### Bug fixes

- Update dependency on iqm-pulse

## Version 7.16.0 (2025-05-16)

### Features

- task management now uses `JobStatus` instead of `SweepStatus`.

## Version 7.15.0 (2025-05-12)

### Features

- Update dependency on station-control-client

## Version 7.14.1 (2025-05-12)

- Test patch versioning, no functional changes.

## Version 7.14.0 (2025-04-22)

### Features

- Update dependency on station-control-client

## Version 7.13.0 (2025-04-17)

### Bug fixes

- Fix broken search input window and shortcut

## Version 7.12.0 (2025-04-17)

### Features

- Bump `iqm-client`, no functional changes.

## Version 7.11.0 (2025-04-11)

### Bug fixes

- Update dependency on iqm-client

## Version 7.10.0 (2025-04-11)

### Bug fixes

- Replace static station url to dynamic in line with other example notebooks.

## Version 7.9.0 (2025-04-10)

### Bug fixes

- Update dependency on station-control-client

## Version 7.8.0 (2025-04-10)

### Features

- Update dependency on iqm-client

## Version 7.7.0 (2025-04-09)

### Bug fixes

- Update dependency on iqm-client

## Version 7.6.0 (2025-04-09)

### Features

- Bump exa-common and iqm-pulse.

## Version 7.5.0 (2025-04-09)

### Bug fixes

- Update dependency on iqm-client

## Version 7.4.0 (2025-04-09)

### Features

- Update Cortex CLI to IQM Client CLI in documentation.

## Version 7.3.0 (2025-04-07)

### Bug fixes

- Fix docs links to `iqm.qiskit_iqm`.

## Version 7.2.0 (2025-04-07)

### Features

- Enable Pulla usage with IQM Server backends

## Version 7.1.0 (2025-04-07)

### Features

- Fix package version in published docs footers.

## Version 7.0.0 (2025-04-04)

### Features

- Replace the old quantum architecture with `DynamicQuantumArchitecture` in `IQMPullaBackend.__init__`.

## Version 6.19.0 (2025-04-03)

Feature

- Enable PEP 604 in linting rules.

## Version 6.18.0 (2025-04-03)

### Bug fixes

- Pulla QIR example now correctly remaps qubits

## Version 6.17.0 (2025-04-02)

### Features

- Update the documentation footer to display the package version.

## Version 6.16.0 (2025-04-02)

### Features

- Fix links to client library docs in docstrings.

## Version 6.15.0 (2025-03-28)

Changes

- Bump iqm-pulse

## Version 6.14.0 (2025-03-27)

### Features

- Update dependency on iqm-pulse

## Version 6.13.0 (2025-03-26)

### Features

- Update dependency on iqm-pulse

## Version 6.12.0 (2025-03-25)

### Features

- Update links to `qiskit_iqm` documentation.

## Version 6.11.0 (2025-03-24)

### Features

- Update dependencies, no functional changes.

## Version 6.10.0 (2025-03-21)

### Features

- Bump dependencies.

## Version 6.9.0 (2025-03-19)

### Bug fixes

- Update dependency on station-control-client

## Version 6.8.0 (2025-03-12)

### Bug fixes

- Small issue where extra characters were left in the notebook from SW-1005.

## Version 6.7.0 (2025-03-11)

### Bug fixes

- Only raise a warning when a custom QIR-profile is submitted, such that qiskit circuits
can be converted to QIR and submitted to our devices.

## Version 6.6.0 (2025-03-10)

Bump dependencies.

## Version 6.5.0 (2025-03-05)

### Features

- Bump version for an updated repo organization. No functional changes.

## Version 6.4.0 (2025-03-05)

### Features

- Remove general RequestError and use new specific error classes instead.
- Use HTTPStatus code names instead of numbers for better clarity. No functional changes.

## Version 6.3.0 (2025-03-04)

### Bug fixes

- Pulla compiler's station settings are now generated with correct paths.

## Version 6.2.0 (2025-03-03)

- Bump exa-common

## Version 6.1.0 (2025-02-28)

### Bug fixes

- Bump exa-common

## Version 6.0.0 (2025-02-27)

### Features

- Adapt to setting tree reorganization
- Replace deprecated usages of `DataType.NUMBER` with either new `DataType.FLOAT` or `DataType.INT`.

## Version 5.28.0 (2025-02-25)

### Features

- Fix broken Configuration and Usage guide.
- Bump Qiskit dependencies.

## Version 5.27.0 (2025-02-24)

### Bug fixes

- Do not attempt to apply dynamical decoupling (DD) sequences on components with virtual drive channels (e.g. computational resonators). This enables DD to be used on stations with star architectures.

## Version 5.26.0 (2025-02-24)

### Features

- Remove unintentional section from Quick start guide.

## Version 5.25.0 (2025-02-19)

### Features

- Require `qiskit-iqm >= 17.0` in the optional `qiskit` dependencies.

## Version 5.24.0 (2025-02-12)

### Features

- Add `iqm.pulla.utils.calset_from_observations` for converting list of observations into a Pulla calibration set.

- Add example notebook for using locally created calibration set with Pulla.

## Version 5.23.0 (2025-02-11)

### Features

- Add missing QIR example user guide to HTML rendered docs.

## Version 5.22.0 (2025-02-10)

### Features

- Bump `iqm-pulse`.

## Version 5.21.0 (2025-02-04)

### Features

- Refactor codebase to new lint rules. No functional changes.

## Version 5.20.0 (2025-02-04)

### Features

- Refactor codebase to new lint rules. No functional changes.

## Version 5.19.0 (2025-02-03)

- Bump iqm-pulse

## Version 5.18.0 (2025-01-30)

### Features

- Display progress bar(s) when user sends circuit for execution using `.Pulla.execute`.

## Version 5.17.0 (2025-01-30)

### Features

- Updated the T1 example notebook to promote a more sensible apporach.

### Bug fixes

- Fixed that modifying the calibration also modified the original calibration stored on the client-side cache.

## Version 5.16.0 (2025-01-28)

Bump iqm-pulse.

## Version 5.15.0 (2025-01-28)

### Features

- Bump `iqm-data-definitions`, no functional changes.

## Version 5.14.0 (2025-01-28)

### Features

- Bump exa-common and iqm-pulse.

## Version 5.13.0 (2025-01-27)

### Features

- Adjust dependencies.

## Version 5.12.0 (2025-01-21)

### Bug fixes

- Make QIR program example compatible with more devices.

## Version 5.11.0 (2025-01-15)

### Bug fixes

- Replace user-given measurement keys to be safe and compatible with exa-db.

## Version 5.10.0 (2025-01-09)

### Features

- `.DummyJob.job_id` of the job returned by `.IQMPullaBackend.run` now returns `sweep_id` of the executed
  sweep.
- `.StationControlResult` returned by `.Pulla.execute` now contains the `start_time` and `end_time`
  even if the task failed.

## Version 5.9.0 (2025-01-08)

### Features

- Remove gitlab links from public pages.

## Version 5.8.0 (2025-01-07)

### Features

- Revoke Station Control task when user aborts Pulla execution.

## Version 5.7.0 (2024-12-30)

### Features

- Update licensing and bump Station Control Client and IQM Pulse dependencies.

## Version 5.6.0 (2024-12-12)

### Features

- Bump exa-experiments

## Version 5.5.0 (2024-12-11)

### Features

- Improvements in the example notebooks.

## Version 5.4.0 (2024-12-10)

### Bug fixes

- Improve documentation structure.

## Version 5.3.0 (2024-12-09)

### Features

Fix extlinks to MRs and issues in sphinx docs config

## Version 5.2.0 (2024-12-05)

### Features

- Fix intersphinx reference paths in docs

## Version 5.1.0 (2024-12-05)

### Features

- Pulla now support base QIR profile as the circuit definition.

## Version 5.0.0 (2024-12-05)

### Features

- Added `.StationControlResult.sweep_id` and `.StationControlResult.task_id`.

## Version 4.8.0 (2024-12-04)

### Features

- By default, `iqm.pulla.cpc.Compiler` can now be initialized with calibration data failing validation.

## Version 4.7.0 (2024-12-04)

### Features

- Bump version for an updated repo organization. No functional changes.

## Version 4.6.0 (2024-11-29)

### Features

- Adjust the conftest calibration set for NDonis to include the parameter `detuning` of all MOVE gate
  nodes, containing the difference of the qubit and resonator frequency.

## Version 4.5.0 (2024-11-27)

### Features

- Added `iqm.pulla.utils_qiskit.IQMPullaBackend` allowing to use Pulla as a backend in Qiskit.

## Version 4.4.0 (2024-11-27)

### Features

- Implement Dynamical Decoupling as a standard compilation stage.

## Version 4.3.0 (2024-11-22)

### Features

- Update to the latest station-control-client.

## Version 4.2.0 (2024-11-21)

### Bug fixes

- Fix a CircuitExecutionError when submitting a batch of circuits measuring different qubits, with heralding enabled.

## Version 4.1.0 (2024-11-19)

### Features

- Bump version for an updated repo organization. No functional changes.

## Version 4.0 (2024-11-14)

- `prepend_reset` (TimeBox-level) standard compiler stage added (implements both reset by wait and active reset)
- added `.CircuitExecutionOptions.active_reset_cycles` that is used to control the reset functionality between.
- :meth`.Pulla.get_standard_compiler` now has an optional argument for overriding default
 circuit execution options

## Version 3.0 (2024-11-01)

- Replaced the function `iqm.pulla.utils_qiskit.qiskit_to_cpc` with
  `iqm.pulla.utils_qiskit.qiskit_circuits_to_pulla`, changing the signature.
- Added the function `iqm.pulla.utils_qiskit.qiskit_to_pulla`.
- Updated the user guide.
- Cleaned up the execution results handling.
- Bugfix: `MeasurementMode.ALL` now works properly with mid-circuit measurements.
- Require `iqm-pulse >= 6.5`, `qiskit-iqm >= 15.0`.

## Version 2.1 (2024-10-25)

- `iqm-pulse` 6.0 compatibility.

## Version 2.0 (2024-10-24)

- See `docs/migration_guide.rst` for a detailed migration guide from version 1.x to 2.0.
- Consolidate compiler code under `iqm.cpc.compiler.compiler` module.
- Remove `iqm.cpc.compiler.compiler2`.
- Do not construct qubit mapping and do not connect to CoCoS.
- Remove `register_fast_feedback` method. Conditional `cc_prx` is now natively supported in CoCoS and Qiskit-on-IQM.

## Version 1.8 (2024-10-18)

- Convert `cc_prx` args like `prx`, convert `measure` "feedback_key" to "feedback_label" for now.

## Version 1.7 (2024-10-09)

- Update `iqm-pulse` to 5.0.

## Version 1.6 (2024-10-07)

- Add trigger delays, `twpa.voltage_1` and `twpa.voltage_2` to calset whitelist.

## Version 1.5 (2024-10-03)

- `register_fast_feedback` now takes feedback signal delays from calibration data.

## Version 1.4 (2024-10-02)

- Qiskit is now an optional dependency.
- Qiskit-related utils are moved to `iqm.pulla.utils_qiskit`. Old import paths are deprecated.

## Version 1.3 (2024-09-30)

- User guides updated for Qiskit 1.x.
- Nicer error messages on authentication problems.
- Add Custom gates user guide to the HTML documentation.

## Version 1.2 (2024-09-25)

- Compilation passes of the standard stages are now by default idempotent.
- User guide updated with more detailed information on authentication.
- Allow custom initial compiler context dictionary to be passed to `Compiler.compile`.

## Version 1.1 (2024-09-23)

- The dynamical implementations created by `register_fast_feedback` are now set as special implementations (protects
  against infinite recursion).

## Version 1.0 (2024-09-20)

- See `docs/migration_guide.rst` for a detailed migration guide from version 0.x to 1.0.
- Compiler and Pulla are now separated for simplicity.
- Compiler is now always refreshed automatically when needed without user's explicit action.
- Pulla no longer needs to access `/cocos/configuration` endpoint.
- Prevent user from accidentally modifying standard stages.
- Compiler stages are all multipass now.
- User guide split into multiple files.

## Version 0.20 (2024-09-18)

- Support ragged acquisition (acquisition labels no longer need to present in every circuit of a batch).
- Circuits in a batch are no longer padded to the same length.
- Heralding is now done on the `TimeBox` level.
- Change the logic for `MeasurementMode`, controlling the final measurement in a circuit:
  - MeasurementMode.CIRCUIT now measures just the qubits that have `measure` gates on them in
    each circuit (previously it measured all the qubits *used* in *any circuit* in the batch).
  - Heralding in MeasurementMode.ALL now performs the heralding measurement (and results filtering)
    only on the qubits used in each circuit (if they *can* be measured, that is). Previously it
    heralded all the qubits used in any circuit in the batch.
- Always send settings to all the probe lines (and TWPAs) on the station, regardless of which
  components are measured in the batch circuits, in order to simplify the settings generation
  logic. This should cause no harm, and typically would happen anyway.

## Version 0.19 (2024-09-09)

- Update to `iqm-pulse` 3.0.
- Add fast feedback example notebook.

## Version 0.18 (2024-09-03)

- Fix and rework `CompositeGate` support. Add `Custom Gates` example notebook.
- `qiskit_to_cpc` no longer takes backend as argument.
- `qiskit_to_cpc` now accepts a list of circuits.
- Adjust logging to not output debug logs by default.
- Change signature of `Pulla.add_implementation` to allow any kind of gate.
- Add `Pulla.register_fast_feedback` as a temporary helper to utilize fast feedback.

## Version 0.17 (2024-08-29)

- Fix front padding of schedules in case instruments have different sampling rates.

## Version 0.16 (2024-08-20)

- Fix result handling of mid-circuit measurements.

## Version 0.15 (2024-08-20)

- Fix failure on null timestamp values

## Version 0.14 (2024-08-14)

- (internal) Rely on chip design record instead of CHADs from station.

## Version 0.13 (2024-08-12)

- Mid-circuit measurement support in the compiler.

## Version 0.12 (2024-08-05)

- Optional `MOVE` validation in the compiler.
- Update `iqm-pulse`.
- Drop support for Python 3.10.
- Drop requirement for `StrEnum` package.

## Version 0.11 (2024-07-15)

- Start using programmable readout (functionally identical to CoCoS 29.0).
- Standard compilation stages adapted to programmable readout.
- User guide and Examples updated.
- Add a decorator `@compiler_pass` that converts a function to a pass with less boilerplate.
- Remove `CompilationStage.add_pass` in favor of `.add_passes`.

## Version 0.10 (2024-07-03)

- Change dependency of `iqm-exa-pulse` to `iqm-pulse`.
- Change dependency of `iqm-exa-backend-client` to `iqm-station-control-client`.

## Version 0.9 (2024-06-28)

- Utility function `qiskit_to_cpc` can now handle Qiskit circuits containing custom composite gates.
- GraphQL support is dropped. Calibration data is now fetched only from Station Control.
- Extended logging support. The user can now set the log level.

## Version 0.8 (2024-06-20)

- New utility function: `station_control_result_to_qiskit` to convert an execution result into a Qiskit result.
- Updated documentation with examples of constructing Qiskit results.
- New optional argument `complex_readout` to `build_settings()` to set result type to complex.
- Breaking change: `circuit_operations_to_iqm` renamed to `circuit_operations_to_cpc`.
- Breaking change: `qiskit_to_iqm` renamed to `qiskit_to_cpc`.

## Version 0.7 (2024-06-18)

- Add an example on how to create T1 Experiment with Pulla.
- Add an example of defining a circuit using IQM Pulse `CircuitOperation`s directly and compiling it normally.
- New utility function: `circuit_operations_to_iqm` to convert a tuple of `CircuitOperation`s into a
  compiler-compatible circuit.
- New utility function: `map_qubit_indices` to replace qubit names in a circuits with their indices from component
  mapping; can be used as a circuit-level pass.

## Version 0.6 (2024-06-17)

- Authentication support for connecting to CoCoS and to Station Control Service.
- Mypy type checking in tests and CI.

## Version 0.5 (2024-06-17)

- Method `Pulla.execute()` returns `StationControlResult`.
- Method `Pulla.execute()` accepts an optional argument `verbose` (default: `False`) to print the measurement
  results.
- Method `Pulla.execute()` prints links to the task id and sweep id pages of the Station Control web interface.
- When execution fails, the error from Station Control is propagated and displayed to the user.

## Version 0.4 (2024-06-10)

- Station Control is now the default provider of calibration data. GraphQL URL is optional.

## Version 0.3 (2024-06-10)

- New utility function: `locate_instructions` to find the channel and index of given playlist instructions with
  optional minimum duration.
- New utility function: `replace_instruction_in_place` to replace an instruction at a given channel+index with one or
  more other instructions, given that the total durations match.
- New utility functions: `print_channel` and `print_schedule` to help visualize the playlist instructions per
  channel.
- New notebook `Examples` added with an example of using the new helper functions to replace `Wait`s with arbitrary
  sequences of pulses&waits for dynamical decoupling.

## Version 0.2 (2024-06-10)

- GraphQL url is now configurable when loading the configuration from url.
- Automatic fetching of latest calibration set on initialization can be disabled.
- Info about schedule visualization added to the user guide.

## Version 0.1 (2024-05-21)

- Initial version.
- Abstract multipass compiler interface and `STANDARD_STAGES`.
- Basic Qiskit integration.
- Circuit compilation and execution.
- Calibration data provider.
