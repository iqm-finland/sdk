# Changelog

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
