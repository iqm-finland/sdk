# Changelog

## Version 20.0.0 (2026-11-03)

### Breaking changes

- Move `Rabi` to iqm-public-experiments. It is no longer importable
  from `iqm-exa-core`. The new import path is
  `from iqm_public_experiments.rabi import Rabi`.
- `exa.core.plotting.figure_helper.generate_figure_title` and
  `exa.core.plotting.figure_helper.generate_figure_filename` signatures have changed.
- `exa.core.post_processing.replace_run_data_field` removed as obsolete.
- `LocalStorage`'s query methods follow the `StorageInterface` changes
  described in `iqm-station-control-client`'s changelog: iterators instead of
  `ListWithMeta`, and `create_observations` returning `list[ObservationData]`.
- `exa.core.database_wrapper.flatten_pages` has been removed; storage
  implementations now page internally, so callers can just iterate
  `query_observations` directly.
- `exa.core.database_wrapper.batch_save_observations` and
  `save_observations_with_new_tags`, and `LocalObservationSequence.save`, now
  return `list[ObservationData]` instead of `ListWithMeta[ObservationData]`.
  A failed observation is logged by the storage backend and simply omitted
  from the returned list. `ObservationStash`-backed calibration-set and
  quality-metric-set creation (`create_calibration_set`,
  `create_quality_metric_set`) is unaffected: it still raises `RuntimeError`
  if any observation failed to save, since that operation must remain
  all-or-nothing.
- Removes `TestChipType` and `StarChipType`.
- Removes `chip_type` argument from `get_coloring` in `calibration_tools`.
- Removes `time_domain` subpackage.
- Moves `calibration_tool` to `iqm-exa-experiments`.

### Bug fixes

- `provider.reset()` (and `ExaRuntime.set()`) now close the outgoing executor
  before replacing or discarding it. Previously, repeated calls to
  `init_monolith_runtime()` (e.g. in `regenerate_graphs.py`) left the
  previous `StationEngine`'s background instrument-polling thread running,
  which could crash once the next `StationEngine` closed all QCodes
  instruments out from under it (SW-3115).

### Features

- Add function `filter_components_with_gate` that filters out components
  that are not registered for any of the implementations of a given gate.
- `ExaRuntime.set` and `ExaRuntime.clear` now also close the outgoing storage
  (see `StorageInterface.close`), not only the executor.
- `init_iqm_server_runtime` and `init_station_control_runtime` give the executor
  and the storage separate clients, each with its own HTTP session.
- Replaces function wrapper `wrap_2qb_plot_for_circuit_dimension` by
  `plot_with_circuit_index` to be used as partial function in
  `PlotDefinition`.
- Removes `CHAD` usage from `QPUPlot`, topology and histogram plotting.
- Defines interfaces for `experiment.utils`, `error_estimation`,
  `gate_implementations`, and `post_processing`.
- Defines interface for observation in `__init__.py`.
- Defines interface for sequence in `__init__.py`.
- Updates the copyright license header in all source files from
  IQM's proprietary header to the Apache License, Version 2.0.
- Moves some `calibration_tools` helpers to `util`.
- Makes iqm-exa-core independent of `calibration_tools` helpers.

## Version 19.0.0 (2026-09-29)

### Breaking changes

- `ObservationHandler` now stores the default StorageInterface
  that will be used when a loading rule does not pin a specific
  storage backend. `ObservationHandler` also stores the `dut_label`.
  These are propagated to the observation loading functions from
  the handler instead of being loaded from the global `ExaRuntime`
  variable.
- `Experiment.__init__` no longer can take `dut_label` as a parameter, but always
  queries it from `ExaRuntime.executor`.
  If you wish to create an `Experiment` instance for loading and re-analyzing
  old run data, use `Experiment.from_dut_label`.
- All experiment classes overriding `process_run_components` must take `settings`
  as an optional argument.

### Bug fixes

- `LocalStorage.query_observations` now correctly returns all unique dut fields,
  also when they have duplicate tags.
- A model's `guess` no longer overwrites what an analysis pinned with
  `set_param_hint`.
  - A parameter that fixes with `vary=False` now keeps the value the hint gave
    it, instead of the guess.
  - A parameter tied to another with `expr` now keeps the expression. `lmfit`
    dropped it as soon as the guess gave the parameter a value.
- Parameter expressions now carry the model prefix, so `GaussExpSineModel`,
  `GaussExpAbsModel` and `SineExpansionModel` no longer fail with `NameError`
  when given one.
- `NyquistSawtoothModel.guess` no longer drops its phase guess when the model
  has a prefix.
- Fixes Graph nodes in `get_connectivity_graph` when `ChipLayout` is empty.

### Features

- `ObservationHandler` and observation handling rules moved out from
  iqm-exa-core into iqm-pulla for, making observation loading the
  same in EXA and pulse-level circuit usage.
- backwards compatible imports are maintained still in
  exa.core.util.observation.* and exa.core.database_wrapper
- Add `settings` as an optional argument to `process_run_components` of `Experiment`.
- Adds the two-tone Ramsey model `Ramsey2fSeparableModel`.
- Adds `BaseModel` to Model Selection Fitter.
- Added `exa.core.local_storage.LocalStorage(StorageInterface)`, a fully
  file-based implementation of the observation and observation set related
  parts of `StorageInterface` that requires no external server or database.
  Supports creating, updating and querying observations and observation sets
  (including tag-based filtering), plus finalization. Pass a `path` to the
  constructor to also persist state to a JSON file: existing state is loaded
  from it on construction, and every create/update operation transparently
  rewrites it. All other `StorageInterface` methods (runs, sequence
  metadata/results, chip design records, calibration sets, dynamic quantum
  architecture) raise `NotImplementedError`.
- Every observation loading rule in
  `exa.core.util.observation.observation_loading_rules` now accepts an
  optional `storage` argument that overrides the ambient
  `ExaRuntime.storage()` for that rule instance, e.g. to load some
  observations from `LocalStorage` instead of (or alongside) a live backend.
  This is robust even when multiple instances of the *same* rule class use
  different storages in one call (e.g. via `ObservationHandler` or
  `load_observations_by_rules`): each rule's cache is internally partitioned
  by storage, so results never leak across instances.
  `ObservationStash.save()`, `LocalObservation.save()`, and
  `LocalObservationSequence.save()` gained the same optional `storage`
  parameter for the equivalent write-side override.
- Several relevant `exa.core.database_wrapper` functions
  (`get_latest_observation`, `get_all_latest_observations`,
  `load_get_latest_for_tags_cache`, `get_observation_set_observations`,
  `get_observation_set_type`, `get_default_calset_observations`,
  `get_latest_observation_set_observations`,
  `get_latest_observation_set_data`, `batch_save_observations`) gained the
  same optional `storage` parameter, additive and backward compatible.

## Version 18.0.0 (2026-09-15)

### Features

- This is the first public version of `iqm-exa-core`
