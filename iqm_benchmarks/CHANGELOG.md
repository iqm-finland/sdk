# Changelog

## Version 2.59.4 (2026-06-08)

### Bug fixes

- Enable ``timeslot`` mode for IQM Resonance.
- Add ``active_reset_cycles`` to circuit compilation options.
- Fix calibration data retrieval.
- Fix QScore benchmark for higher QAOA depths.

## Version 2.59.3 (2026-04-30)

### Bug fixes

- Switch from pinned dependencies to ranges for the bare package installation.
- Domain changed from meetiqm.com to iqm.tech.

### Features

- Prepare package for `4.5.2` release. No functional changes.

## Version 2.59.2 (2026-04-23)

### Features

- Prepare package for `4.5.1` release. No functional changes.

## Version 2.59.1 (2026-04-09)

### Features

- Publish package and documentation publicly.
- Migration to an internal repo from [GitHub](https://github.com/iqm-finland/iqm-benchmarks).
- Removal of pyGSTi dependency and addition of functionality replacement
  code in mgst.
- Enabled local generation of single- and two-qubit GST circuits without
  version controlled .pkl files.
- Removed obsolete `benchmark.py` (all base class functionality is now in `benchmark_definition.py`).
- Extensive linting and type hinting changes to conform with repo standards.

## Version 2.58

- Fixed a bug that would lead to circuit compilation options not being applied.

## Version 2.57

- Added calibration data for Clifford fidelities.

## Version 2.56

- Fixed import bug with iRB.
- Improved execution of star_optimal GHZ creation circuit.

## Version 2.55

- Compatibility with new job metadata formats.
- Unified fetching of calibration data via IQMClient.
- Addition of active reset and approximate transpilation options in the configurations.

## Version 2.54

- Update iqm-client and iqm-station-control-client dependency ranges for compatibility with IQM OS 4.4.

## Version 2.53

- Improvements made for QScore benchmark to take custom layout for Star processors.

## Version 2.52 (2025-11-13)

### Features

- Circuit transpilation improvements for GHZ benchmark implemented.

## Version 2.51 (2025-11-07)

### Features

- Circuit transpilation improvements for Q-score implemented.

## Version 2.50 (2025-10-30)

### Features

- Update iqm-client and iqm-station-control-client dependency versions as part of IQM OS 4.3 release.

## Version 2.49 (2025-10-20)

### Features

- Added logging of execution time to all benchmarks.

## Version 2.48 (2025-10-16)

### Features

- Updated iqm-client and supported python versions.

## Version 2.47 (2025-09-05)

### Features

- Updated qiskit and iqm-client dependencies

## Version 2.46 (2025-08-27)

### Features

- Bug fixes encorporated for Q-score benchmark.

## Version 2.45 (2025-08-25)

### Features

- GST overhaul:
- Added gate context
- Updated and fixed Hinton diagrams for process matrices
- Added dominant coherent errors plot
- Updated Jupyter tutorial: Running for different ranks, new plot, context explanation
- Properly defined idle gates with delay and added idle gate to default two qubit gate set
- Parallel post-processing of layouts
- Maximum likelihood optimization for the final estimate

## Version 2.44 (2025-08-11)

### Features

- Added qubit positions for Emerald
- Visibility improvements for Graph state and EPLG plots
- Add visualization of single qubit errors (Fidelity, readout, T1, T2) to the layout fidelity graph plot

## Version 2.43 (2025-07-27)

### Features

- Added coherence benchmark for T1 and T2 estimations.

## Version 2.42 (2025-07-22)

### Features

- Improved run-time execution for QScore benchmark.

## Version 2.41 (2025-07-22)

### Features

- Fixed deprecated colormap usage in several benchmarks.

## Version 2.40 (2025-07-04)

### Features

- Added optimise_single_qubit_gates option for crystal processors.

## Version 2.39 (2025-06-26)

### Features

- Relax `numba` dependency to allow easier interoperability with other client libraries.

## Version 2.38 (2025-06-26)

### Features

- Update `iqm-client` dependency to `29.0+`.
- Added iqm-station-control-client dependency to ensure iqm-client version works with stations on resonance.

## Version 2.37 (2025-06-11)

### Features

- Update `iqm-client` dependency to `27.0+`.

## Version 2.36 (2025-04-28)

### Features

- Fixed bugs for qscore and CLOPS.

## Version 2.35 (2025-04-22)

### Features

- Improved qiskit transpilation for Qscore benchmark.

## Version 2.34 (2025-04-16)

### Features

- Update `iqm-client` dependency to `23.8+`.

## Version 2.33 (2025-04-14)

### Features

- Added Error Per Layered Gate (EPLG) benchmark.

## Version 2.32 (2025-04-08)

### Features

- Fixed incorrect edge width assignment in qubit selection plot.

## Version 2.31 (2025-04-02)

### Features

- Fixed a bug in the bootstrapping functionality of the GST benchmark and
  updated the respective Jupyter tutorial.

## Version 2.30 (2025-04-01)

### Features

- Updated CLOPS value and plot reporting, making explicit offline values, such
  as time spent in transpilation and in parameter assigning.

## Version 2.29 (2025-03-26)

### Features

- Fixed header levels in `example_graphstate` notebook for correct pages
  rendering.

## Version 2.28 (2025-03-26)

### Features

- Added graph state (bipartite entanglement negativity) benchmark.

## Version 2.27 (2025-03-24)

### Features

- Qiskit on IQM dependency updated to > 17.0.

## Version 2.26 (2025-03-20)

### Features

- Changed benchmark observation names and identifiers to be more consistent
  with guidelines.

## Version 2.25 (2025-03-18)

### Features

- Added optional configuration parameter (`max_circuits_per_batch`) to specify
  the maximum amount of circuits per batch.

## Version 2.24 (2025-03-07)

### Features

- Added rustworkx dependency range to fix wrong edge thickness assignment in
  qubit selection plot.

## Version 2.23 (2025-03-04)

### Features

- Added dynamical decoupling parameter option to configurations of all
  benchmarks.
- Added visual aid plot for qubit selection (see, e.g., GHZ example notebook).
- Included option to run GST in parallel if the specified qubits don't overlap
- Small runtime improvements in the GST benchmark.
- Changed tensor order in GST outputs from Qiskit (bottom to top) to standard
  (top to bottom) order.

## Version 2.22 (2025-02-27)

### Features

- Fix for QScore errors when custom_qubits_array is specified.

## Version 2.21 (2025-02-25)

### Features

- Function to bootstrap counts added to utils file.

## Version 2.20 (2025-02-20)

### Features

- Standardizes observations for CLOPS and Mirror RB.

## Version 2.19 (2025-02-18)

### Features

- All functional tests extended to the fake Deneb backend.
- Added backend transpilation to REM calibration circuits to fix errors with REM
  on fake Deneb.

## Version 2.18 (2025-02-12)

### Features

- Added notebook to benchmark IQM Star QPUs and bug fixes done for Qscore.

## Version 2.17 (2025-02-10)

### Features

- Update installation command for development mode.
  `#41 <https://github.com/iqm-finland/iqm-benchmarks/pull/41>`_

## Version 2.16 (2025-02-07)

### Features

- Added readout error mitigation for Qscore benchmark.

## Version 2.15 (2025-02-06)

### Features

- Added optimal GHZ circuit generation and corresponding example notebook for
  all-to-all connected QPU topology.

## Version 2.14 (2025-01-31)

### Features

- Added devices folder in docs with notebook to benchmark IQM Spark.

## Version 2.13 (2025-01-20)

### Features

- Move all example notebooks to docs.
  `#30 <https://github.com/iqm-finland/iqm-benchmarks/pull/30>`_

## Version 2.12 (2025-01-20)

### Features

- Added compatibility with IQM-Deneb by adapting the transpilation behavior in
  several benchmarks.

## Version 2.11 (2025-01-17)

### Features

- Report average native single-qubit gate fidelity estimates in observations of
  1Q Clifford RB and 1Q IRB, and display in plots of 1Q Clifford RB.

## Version 2.10 (2025-01-17)

### Features

- Fix docs publishing by CI.

## Version 2.9 (2025-01-15)

### Features

- Add optional security-scanned lockfile.

## Version 2.8 (2025-01-10)

### Features

- Fixed a bug where optional dependencies related to gst were imported with
  other benchmarks, leading to a ModuleNotFoundError.

## Version 2.7 (2025-01-09)

### Features

- Fixed bugs in Qscore and enabled benchmark execution for pyrite.

## Version 2.6 (2025-01-08)

### Features

- Fixed bugs including wrong GHZ plot x-Axis labels and incorrect transpiled and
  untranspiled circuit storage for mGST.
- Added note about optional dependency "mgst".
- Improved display and calculation method for Hamiltonian parameter output of
  rank 1 compressive GST.

## Version 2.5 (2025-01-07)

### Features

- Changed simulation method for MRB to 'stabilizer' and simulation circuits are
  compiled in circuit generation stage.

## Version 2.4 (2024-12-23)

### Features

- Changed Qscore to operate under the new base class.

## Version 2.3 (2024-12-18)

### Features

- Reverted QV simulation circuits to untranspiled ones (fixes bug giving all
  HOPs equal to zero).

## Version 2.2 (2024-12-17)

### Features

- Added Clifford RB example notebook to docs.
  `#20 <https://github.com/iqm-finland/iqm-benchmarks/pull/20>`_

## Version 2.1 (2024-12-16)

### Features

- Fixed bug in RB plots for individual decays.

## Version 2.0 (2024-12-16)

### Features

- Adds `Circuits`, `BenchmarkCircuit` and `CircuitGroup` as a way to easily
  store and interact with multiple quantum circuits.
- `BenchmarkRunResult` now takes a `circuits` argument, expecting an instance
  of `Circuits`. `QuantumCircuit` instances can now exist there instead of
  inside xarray Datasets. All analysis methods should also expect to use an
  instance of `BenchmarkRunResult`.
- Ported all of the benchmarks subclassing from `Benchmark` to use the new
  containers.
- Updates the usage of `qiskit.QuantumCircuit` to `iqm.qiskit_iqm.IQMCircuit`
  in many places.

## Version 1.12 (2024-12-13)

### Features

- Miscellaneous small bugs fixed.

## Version 1.11 (2024-12-12)

### Features

- Relaxes dependencies to allow for ranges.

## Version 1.10 (2024-12-11)

### Features

- Added API docs building and publishing.

## Version 1.9 (2024-12-10)

### Features

- Fixed bug (overwriting observations) in Quantum Volume.
- Fixed small bug in CLOPS when calling plots in simulator execution.

## Version 1.8 (2024-12-09)

### Features

- Changed compressive GST to operate under the new base class and added
  multiple qubit layouts.
- Added plot to GHZ benchmark and applied small fixes.
- Added tutorial notebook for the GHZ benchmark.

## Version 1.7 (2024-12-09)

### Features

- Remove explicit dependency on qiskit, instead taking it from qiskit-on-iqm.

## Version 1.6 (2024-12-03)

### Features

- Minor change in dependencies for compatibility.

## Version 1.5 (2024-12-03)

### Features

- fit results are no longer `BenchmarkObservation`, and instead are moved into
  the datasets.

## Version 1.4 (2024-12-03)

### Features

- Renames:
  - AnalysisResult -> BenchmarkAnalysisResult
  - RunResult -> BenchmarkRunResult
- Adds BenchmarkObservation class, and modifies BenchmarkAnalysisResult so
  observations now accepts a list[BenchmarkObservation].
- Adds BenchmarkObservationIdentifier class.
- Rebases RandomizedBenchmarking benchmarks, QuantumVolume, GHZ and CLOPS to
  use the new Observation class.
- Fixes serialization of some circuits.
- Adds AVAILABLE_BENCHMARKS to map a benchmark name to its class in __init__.
- Adds benchmarks and configurations to __init__ for public import.
- Other fixes.

## Version 1.3 (2024-11-29)

### Features

- Further improvements to type hints, docstrings, and error messages.

## Version 1.2

### Features

- Minor improvements to type hints, docstrings, and error messages.

## Version 1.1

### Features

- Fixed bug preventing execution on a generic IQM Backend.
- Randomized Benchmarking (Clifford, Interleaved and Mirror), Quantum Volume,
  CLOPS and GHZ state fidelity all functioning exclusively under new Benchmark
  base class.
- Updated separate example Jupyter notebooks.

## Version 1.0

### Features

- Published Randomized Benchmarking (Clifford, Interleaved and Mirror), Quantum
  Volume, CLOPS and GHZ state fidelity all functioning exclusively under new
  Benchmark base class.
- Updated separate example Jupyter notebooks.
