# Changelog

## Version 2.0.0 (2026-11-03)

### Breaking changes

- Renamed `core/calibration.py` to `core/metrics_data.py`, and
  `CalibrationData`, `FullCalibrationData` and `CalibrationDataManager` to
  `MetricsData`, `FullMetricsData` and `MetricsDataManager`. The fetch methods
  are now `get_metrics_data(client, calibration_set_id=None)` and
  `get_full_metrics_data(client, calibration_set_id=None)`; they fetch the
  architecture and the metrics of the same calibration set. Without a client,
  `metrics_from_observations(observations, arch)` and
  `full_metrics_from_observations` parse observations already at hand.
  `FullMetricsData.distill` is now `to_metrics_data`.
- Renamed `core/architecture.py` to `core/graph_builder.py`.
- `build_coupling_graph` now returns only the graph, and
  `remove_qubits_from_graph` and `VF2Enumerator` no longer take `node_labels`.
  The qubits are `list(graph.nodes())`, in architecture order.
- `select_layout` now returns a list of `(layout, error)` pairs instead of a
  `(layouts, errors)` tuple, and its third parameter is named `metrics`.
  Instead of `arch` and `metrics`, a `client=` can be given, and both are
  fetched from it.
- Renamed the `num_trials` parameter to `max_num_layouts`.
- `MetricsData.readout` is now keyed by `ReadoutMode`, e.g.
  `metrics.readout[ReadoutMode.FIDELITY]["QB1"]`.
- Renamed `FastEdgeCost.score` to `sort_by_scores`.
- A missing fidelity now gives a layout infinite error instead of being treated
  as perfect. A fidelity outside `(0, 1]`, or a readout mode missing from the
  metrics, raises `ValueError`.

### Bug fixes

- Fixed metrics being attributed to the wrong physical qubit when a circuit
  has a qubit with only single-qubit gates.
- Single-qubit-only qubits can now be placed on any free qubit, not only ones
  adjacent to the rest of the layout.

### Features

- Added `score_layouts`, which returns every candidate as a `LayoutScore`
  ranked by an error budget in nats, split into two-qubit, single-qubit and
  readout terms. This keeps rankings meaningful for deep circuits.
- Added `LayoutCost` and `CostCalculatorBase` as a base for custom cost models.
- Added `ReadoutMode.QNDNESS` and `FullMetricsData.readout_qndness`.
  `ReadoutMode` is now a `StrEnum`.
- Added support for Star architectures, including
  `fictional_two_qubit_fidelities` for effective qubit-qubit CZ fidelities.
- Added `AllToAllEnumerator` for fully connected coupling graphs. It and
  `VF2Enumerator` share the `BaseEnumerator` interface,
  `enumerate(interaction_graph, num_circuit_qubits)`, and `make_enumerator`
  picks the right one for a coupling graph.
- Circuits whose two-qubit interactions are disconnected, e.g.
  `CZ(QB1, QB2)` and `CZ(QB3, QB4)`, can now be placed.
- `VF2Enumerator` now restarts its capped rustworkx VF2 search from every
  qubit in turn, so it returns layouts spread over the whole chip instead of
  one corner. It takes a `seed` argument for reproducible sampling.

## Version 1.2.0 (2026-09-29)

### Bug fixes

- Fix the heading levels of the example notebooks, so their sections nest
  correctly in the documentation navigation.

### Features

- Bumps minimum NumPy version to 2.2 after 2.1 has reached its EOL.

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
