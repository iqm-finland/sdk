# Changelog

## Version 0.2.2 (2026-08-26)

### Bug fixes

- `REMWorkflow` now passes the correct `twirled` flag to the readout error
  mitigation step instead of always assuming `True`. The confusion matrix is
  only symmetrized when readout twirling was actually applied, so results are
  now correct with `strategy="NONE"` or when twirling is skipped by a fallback
  (missing LOCAL topology, MOVE-gate QPU).
- Add `CircuitTwirler.readout_twirling_applied` reporting whether readout
  twirling was actually applied.
- Remove the nonexistent `iqm.readout_characterization` module from the API
  reference, and fix the intersphinx links to `iqm-station-control-client`.
- Add the user guide and quick start to the documentation navigation; both were
  unreachable, and the user guide linked to a nonexistent page.

## Version 0.2.1 (2026-08-05)

### Features

- Prepare package for `4.6.1` release. No functional changes.

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
