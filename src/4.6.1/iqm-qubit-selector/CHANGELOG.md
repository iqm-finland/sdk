# Changelog

## Version 1.1.1 (2026-08-05)

### Features

- Prepare package for `4.6.1` release. No functional changes.

## Version 1.1.0 (2026-07-07)

### Bug fixes

- Domain changed from meetiqm.com to iqm.tech
- Switch from pinned dependencies to ranges for the bare package installation.

### Features

- Improvement on general documentation and example notebook for ``iqm-qubit-selector``.
- Removed deprecated ``iqm_server_url`` use.
- Add optional dependency groups with pinned versions to `iqm-qubit-selector`:
  - `pin-iqm`: pins inter-IQM package dependencies to the exact versions that
    are tested and released together.
  - `pin-all`: like `pin-iqm`, but additionally pins all transitive third-party
    dependencies to exact locked versions.
  - `notebook-pin-iqm`: includes `notebook` extra dependencies with inter-IQM
    packages pinned to co-released versions.
  - `notebook-pin-all`: like `notebook-pin-iqm`, but additionally pins all
    transitive third-party dependencies.

## Version 1.0.0

### Features

- First release of `iqm-qubit-selector` for crystal topology.
- Publish package and documentation publicly.
