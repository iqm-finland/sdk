Version 3.0 (2026-08-04)
========================
- Add `sweep_results_format` to `SweepTaskRequest` to support selecting legacy vs streaming sweep results artifacts.
- Add `SweepResultsChunk` protobuf message for chunked streaming sweep results payloads.
- Add `CosineRiseFallDerivative`, `CosineRiseFallDerivative` and `Slepian` to waveforms.
- Add `CosineRiseFallDerivative`, `CosineRiseFallDerivative` and `Slepian` to protobuf definitions.
- Update docs links to https://docs.iqm.tech/iqm-data-definitions/.
- Fix missing CHANGELOG.rst file.

Version 2.24 (2026-03-25)
=========================
- Relax protobuf version range to >=6, <8.

Version 2.23 (2026-03-25)
=========================
- Bump protobuf dependency to 7.34.1 as the previous version had reached its end of life.

Version 2.22 (2026-01-15)
=========================
- Increase upper version limit of scipy to 2.0

Version 2.21 (2026-01-05)
=========================
- :attr:`.Setting.read_only` is included in the proto.

Version 2.20 (2025-12-18)
=========================
- Fix incorrect output of :meth:`.Playlist.view`.

Version 2.19 (2025-09-30)
=========================
- Supply mypy type information for :mod:`iqm.data_definitions` generated sources.

Version 2.18 (2025-09-29)
=========================
- Add protobuf descriptor_set.bin to the package under generated source.

Version 2.17 (2025-09-26)
=========================
- Fix package publishing step in CI.

Version 2.16 (2025-09-25)
=========================
- Supply mypy type information for :mod:`iqm.models`.
- Require Python version greater than 3.11. Testing is done with 3.13.

Version 2.15 (2025-08-19)
=========================
- Improve docstrings and type annotations in :class:`Playlist`.

Version 2.14 (2025-08-12)
=========================
- Add method :meth:`view` to :class:`Playlist` to view operations of a given playlist, for simple inspection and
  debugging purposes.

Version 2.13 (2025-05-15)
=========================
- Bump minimum NumPy version to 1.26.4.

Version 2.12 (2025-04-03)
=========================
- Fix :class:`ChannelProperties` `instruction_duration_granularity` and `instruction_duration_min` typehints to integers.
- Add fields `instruction_duration_granularity_samples`  and `instruction_duration_min_samples` to
  message ChannelProperties.


Version 2.11 (2025-04-01)
=========================
- Fix :class:`ReadoutProperties` `integration_start_dead_time` and `integration_stop_dead_time` typehints to integers.
- Add fields `integration_start_dead_time_samples`  and `integration_stop_dead_time_samples` to
  message ReadoutProperties.

Version 2.10 (2025-03-26)
=========================
- Fix version string in documentation and include .proto files in sdist.

Version 2.9 (2025-03-24)
========================
- Add option to specify numcodecs-compatible compression stages for protobuf spot_result array data. :issue:`SW-1307`

Version 2.8 (2025-02-17)
========================
- Fix links.

Version 2.7 (2025-01-21)
========================
- Add protobuf definitions for channel properties.
- Add python classes of channel properties.

Version 2.6 (2025-01-09)
========================
- Support NumPy 2

Version 2.5 (2025-01-02)
========================
- Update licensing information.

Versions 2.4 (2024-10-24)
=========================
- Bump NumPy version

Versions 2.3 (2024-10-10)
=========================
- Fix docs generation.

Versions 2.2 (2024-10-10)
=========================

- Update scipy requirement from 1.11.1 to 1.11.4. The new version includes
  bugfixes and a Python 3.12 wheel.

Version 2.1 (2024-10-10)
========================

Features
--------

- Add HTML documentation. :issue:`SW-703`.

Version 2.0 (2024-10-01)
========================

Features
--------
- Waveform canonicality is no longer an inheritable property of the class, but instead there is a decorator
  ``register_canonical_waveform`` that can be used to make a waveform canonical.

Version 1.18 (2024-07-18)
=========================

Features
--------

- Add new attribute ``feedback_signal_label`` for ThresholdStateDiscrimination.

Versions 1.15-1.17 (2024-06-06)
===============================

- Update protobuf version to increase serialization speed.

Version 1.14 (2024-06-05)
=========================

Features
--------

- Add new attribute ``phase_increment`` for IQ pulses in both :class:`Playlist` and protobuf definition.
  :issue:`EXA-1751`

Version 1.13 (2024-05-31)
=========================

Features
--------

- Add boolean and array of booleans and strings to ``ObservationValue``.

Version 1.12 (2024-05-27)
=========================

Features
--------

- Add ``ObservationValue``, ``ObservationUncertainty``, and ``SpotResultValue``, originating from exa-data,
  as well as ``Int64Array``, ``Float64Array``, and ``Complex128Array``, which are required for observations
  and spot results.

Version 1.11 (2024-05-24)
=========================

Features
--------

- Add ``iqm.models`` module which contains definition of the iqm.pulse.playlist.

Version 1.10 (2024-05-06)
=========================

Features
--------

- Add field ``sweep_definition_payload`` to ``RunDefintion`` v2.


Version 1.9 (2024-05-06)
========================

Features
--------

- Add messages for programmable readout.

Version 1.8 (2024-01-09)
========================

Features
--------

- Add ``TruncatedGaussian``, ``TruncatedGaussianSmoothedSquare``, ``TruncatedGaussianDerivative``
  and ``CosineRiseFall`` as `oneof` types for the Waveform.

Version 1.7 (2023-12-01)
========================

Features
--------

- Add v2 RunDefinition model using custom Struct implementation with integer support.

Version 1.6 (2023-09-21)
========================

Features
--------

- Add protobuf model for ``RunDefinition``.

Version 1.5 (2023-09-15)
========================

Features
--------

- Add ``playlist.proto`` file which defines ``Playlist`` message for exa.pulse2 ``Schedule``
- Add ``playlist`` field to ``SweepRequest``

Version 1.4 (2023-08-29)
========================

Features
----------------

- Add v2 SweepResultsResponse, which changes results format from ``dict[str,np.ndarray]`` to
  ``dict[str, list[np.ndarray]]``.

Version 1.3 (2023-08-17)
========================

Features
--------

- Add optional ``element_indices``, ``parent_name``, and ``parent_label`` fields to parameter.

Version 1.2 (2023-05-22)
========================

Features
--------

- Add optional dut_label field to sweep requests.

Version 1.1 (2023-03-28)
========================

Features
--------

- Added optional full parameter spec to sweeps.


Version 1.0 (2023-02-01)
========================

Features
--------

- Added first data definitions aimed to change Station Control Service communication from JSON to Protobuf.
- Added automatic generation of `.pyi` files by using `mypy-protobuf`.
- Allow local installation with pip <21.3.


Breaking changes
----------------

- Dummy definitions removed.


Version 0.5 (2022-12-19)
========================

Breaking changes
----------------

- Use protoc version < 21 in order to be compatible with the transitive dependency brought by qcodes in exa-experiment.
  Protobuf versioning has changed recently a little bit, such that libprotoc's version is kept in sync with the minor
  version of different language package distributions, and those distributions have their own major versions to indicate
  breaking changes. Compiler version 21 brings a breaking change to the Python package `protobuf` and bumps its major
  version to 4. With quick testing the changes appear to be backwards compatible in simple cases, however, it's best
  to limit the Python package's major version to have more confidence in compatibility with more complex protobuf
  definitions.

- Note that this breaking change does not bump major version because we are still in alpha.

Version 0.4 (2022-12-19)
========================

Bug fixes
---------

- Pin protoc version in the pipeline. This makes the generated Python package to be compatible with the newest
  `protobuf` package major version 4 that is currently required in pyproject.toml.

Version 0.3 (2022-12-16)
========================

Features
--------

- Protobuf definition directory paths are changed.
- The official ``protoc`` compiler is used instead of betterproto.
- A non-breaking change is introduced in DummyTaskServiceMessage.

Version 0.2 (2022-12-14)
========================

Features
--------

- Add a non-breaking change to StringSequence
- Add another placeholder proto, BoolSequence

Version 0.1 (2022-11-30)
========================

Features
--------

- Initialize new repository
- Add first placeholder .proto definition
