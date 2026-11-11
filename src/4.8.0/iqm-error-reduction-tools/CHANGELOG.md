# Changelog

## Version 0.4.0 (2026-11-03)

### Bug fixes

- Every tool now derives the qubit-to-classical-bit mapping from the measurement
  keys, via the new `utils.circuit_utils.clbit_index_from_key` and
  `TwirledCircuit.qubit_to_clbit()`. `CircuitTwirler` and
  `ReadoutErrorMitigation` previously numbered measured qubits by the order
  their `measure` operations appeared, which silently permuted observables and
  readout untwirling whenever a circuit emitted measurements out of classical-bit
  order. Keys that carry no classical bit index still fall back to measurement
  order.
- `CircuitTwirler.from_dict()` silently discarded the rotation strings that
  `to_dict()` stored, so a restored twirler had no way to untwirl counts. They
  are now restored.

### Features

- Add `ErrorReductionWorkflow`, which estimates observables of target circuits
  with an error reduction technique in four stages: `plan()`, `submit()`,
  `collect()` and `estimate()`, or all at once with `run()`. It combines the
  technique with twirling and readout error mitigation, and `save()` / `load()`
  store its state between collection and estimation. Techniques plug in through
  the `Technique` and `Estimation` interfaces.
- Release of zero-noise extrapolation (ZNE).
- Release of Clifford data regression (CDR).
- `REMWorkflow` can now be serialized and restored, so a run can be submitted in
  one Python session and retrieved in another: `to_dict()` / `from_dict()` and
  the file-based `save()` / `load()`. The circuit job is the expensive part of a
  run and may queue for a long time; this removes the need to keep the
  submitting process alive to collect its results.
  The circuits themselves are not stored, since the quantum computer already has
  them. The client passed to `from_dict()` / `load()` must point at the same
  quantum computer the job was submitted to.
  The readout error characterization must already have been retrieved when the
  workflow is serialized; a pending characterization job cannot be restored,
  because retrieving one needs the `CalibrationJobInfo` produced at submission,
  which `ReadoutErrorCharacterization` does not persist. Characterization is
  fast compared with the circuit job, so retrieve it before saving.
- `CircuitTwirler.to_dict()` now also carries the submitted job id and the
  per-circuit shot count, and `from_dict()` rebuilds the job through
  `IQMClient.get_job()`. A restored twirler can therefore run
  `retrieve_counts()`, which previously required the original in-memory object.
- `CircuitTwirler.to_dict()` now also stores whether readout twirling was
  actually applied, so `readout_twirling_applied` works on a restored twirler.
  Payloads written by an earlier version, which have no such field, are read as
  "twirled as configured".
- `CircuitTwirler.get_rot_strings()` works on a restored twirler.
- `REMWorkflow.submit()` and `REMWorkflow.run()` accept a new
  `observables_per_circuit` argument, assigning a distinct set of observables
  to each input circuit. The existing `observables` argument, which applies one
  shared list to every circuit, is unchanged; the two are mutually exclusive.
- Add the `rem.ObservableSpecs` type alias for an observable specification list.

## Version 0.3.0 (2026-09-29)

### Breaking changes

- `REMResults.mitigated_counts` has been removed. Use
  `REMResults.mitigated_distributions` instead. The values are
  (quasi-)probabilities summing to approximately 1, not integer counts.
- `REMWorkflow.get_mitigated_counts()` has been removed. Use
  `REMWorkflow.get_mitigated_distributions()` instead.

### Bug fixes

- Remove the nonexistent `iqm.readout_characterization` module from the API
  reference, and fix the intersphinx links to `iqm-station-control-client`.
- Add the user guide and quick start to the documentation navigation; both were
  unreachable, and the user guide linked to a nonexistent page.
- Fix docstring rendering: ket notation, an empty `note` directive and a nested
  list.
- `REMWorkflow` now passes the correct `twirled` flag to the readout error
  mitigation step instead of always assuming `True`. The confusion matrix is
  only symmetrized when readout twirling was actually applied, so results are
  now correct with `strategy="NONE"` or when twirling is skipped by a fallback
  (missing LOCAL topology, MOVE-gate QPU).
- Add `CircuitTwirler.readout_twirling_applied` reporting whether readout
  twirling was actually applied to the counts.

### Features

- `REMWorkflow` now exposes the submitted QPU job ids via `get_job_ids()`, and
  includes `characterization_job_id` and `twirling_job_id` in `REMMetadata`.
- Add `CircuitTwirler.get_job_id()` and
  `ReadoutErrorCharacterization.get_job_id()` returning the submitted job id.
- Add `REMResults.mitigated_distributions` and
  `REMWorkflow.get_mitigated_distributions()`, replacing the removed
  `mitigated_counts` field and `get_mitigated_counts()` method.
- Bumps minimum NumPy version to 2.2 after 2.1 has reached its EOL.

## Version 0.2.0 (2026-07-07)

### Bug fixes

- Updated twirling tutorial (now working).
- Clearer explanation/handling on how "total" shots are assigned to
  individual circuit when a list of circuit is twirled and then executed.
- Handling non-working qubits when readout characterization is called on
  "all" qubits.
- Allow to run REMWorkflow also without twirling (i.e. just basic REM).

### Features

- First release of IQM Error Reduction Tools.
- Fast post-processing using the `mthree` package.
- Fast post-processing using the twirled REM framework.
- REC (Readout Error characterization) framework.

## Version 0.1.0
