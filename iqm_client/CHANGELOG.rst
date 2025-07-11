=========
Changelog
=========

Version 29.5.0 (2025-07-09)
===========================

Features
--------

- Enable mypy type checking in CI and add temporary type ignores to the source code. :issue:`SW-1615`

Version 29.4.0 (2025-07-07)
===========================

Features
--------

- Bugfix: Fixed the iqm-client cirq documentation such that the instruction is to route circuits before decomposing.

Version 29.3.0 (2025-07-02)
===========================

Bug fixes
---------

- Fix type errors raised by mypy.

Version 29.2.0 (2025-06-30)
===========================

Features
--------

- Bugfix: Fixed case where the transpiler would sometimes replace Parameters in the circuit with another Parameter.

Version 29.1.0 (2025-06-30)
===========================

Bug fixes
---------

- Add job id to APITimeoutError

Version 29.0.0 (2025-06-13)
===========================

Features
--------

- Major :class:`IQMClient` refactoring. Many methods now use :class:`StationControlClient` internally to retrieve the data,
  instead of using direct HTTP requests to REST API. This work is going to continue in the future, and the end goal
  is to remove all HTTP logic from :class:`IQMClient` and handle HTTP requests/responses only in :class:`StationControlClient`.
  :issue:`SW-1084`

Breaking Changes
----------------

- Remove :class:`APIVariant` from :class:`IQMClient`, it will support only one version at any given time from now on.
  Upgrade/downgrade software if you want to use different versions of :class:`IQMClient`. :issue:`SW-1084`
- You may have to pass a different `url` to :class:`IQMClient` than before:
  - If you previously used :class:`APIVariant.V1`, you typically need to change `/cocos` to `/station` in the URL.
  - If you previously used :class:`APIVariant.V2`, you typically need to add `/station` to the URL.
  - If you previously used :class:`APIVariant.RESONANCE_COCOS_V1`, no changes are needed to the URL.
- Support for ``timeout_secs`` parameter has been dropped from some methods (:meth:`get_static_quantum_architecture`,
  :meth:`get_quality_metric_set`, :meth:`get_calibration_set`, :meth:`get_dynamic_quantum_architecture`, :meth:`get_feedback_groups`).
  :class:`IQMClient` uses :class:`StationControlClient` to retrieve the data now instead of HTTP requests directly, and we have no
  plans to support different request timeout for each endpoint separately in :class:`StationControlClient`. Instead, all
  endpoints uses the same timeout which is currently set to default 120 seconds.

Version 28.0.0 (2025-06-12)
===========================

Breaking changes
----------------

- Updated `iqm_client.models.Status` to conform with changed station-control-client job statuses. :issue:`SW-1513`.

Version 27.1.0 (2025-06-09)
===========================

Bug Fixes
---------

- Added a backwards compatibility fix for Resonance. :issue:`SW-1532`
- Made Resonance integration tests mandatory again. :issue:`SW-1532`

Version 27.0.0 (2025-06-02)
===========================

Features
--------

- Revert changes from :issue:`SW-1513`.

Version 26.0.0 (2025-05-30)
===========================

Features
--------

- Altered `iqm_client.models.status` to cover changed station-control-client job statuses. :issue:`SW-1513`.

Version 25.5.0 (2025-05-30)
===========================

Features
--------

- Bump NumPy to 1.26.4.

Version 25.4.0 (2025-05-30)
===========================

Bug fixes
---------

- Improve auth error message

Version 25.3.0 (2025-05-28)
===========================

Features
--------

- Use new error log artifact format when obtaining job error message from station using V2 API.

Version 25.2.0 (2025-05-23)
===========================

Features
--------

- ``IQMClient.submit_run_request`` uses UTF-8 encoding for the JSON payload.

Version 25.1.0 (2025-05-21)
===========================

Features
--------

- Fix cocos path in ruff isort to run isort for cocos correctly.

Version 25.0.0 (2025-05-16)
===========================

Features
--------

- Extended `iqm_client.models.status` to cover new station-control job statuses. :issue:`SW-948`.
- Updated to work with station-control version 42.0. Earlier versions of station-control will not work.
  :issue:`SW-948`

Breaking Changes
----------------
- :attr:`iqm_client.models.Status.PENDING_EXECUTION` changed from "pending execution" to "pending_execution".
  :issue:`SW-948`.
- :attr:`iqm_client.models.status.PENDING_COMPILATION` changed from "pending compilation" to "pending_compilation".
  :issue:`SW-948.`

Version 24.3.0 (2025-05-14)
===========================

Feature
-------

- Allow the :class:`.IQMMoveLayout` transpiler to handle gates that are not in the native gateset, done by skipping the marking
  of Circuit Qubits as either qubit or resonator when the gate is not natively supported. :issue:`SW-1390`.

Version 24.2.0 (2025-05-12)
===========================

Features
--------

- Update dependency on exa-common

Version 24.1.1 (2025-05-12)
===========================

- Test patch versioning, no functional changes. :issue:`SW-1429`

Version 24.1.0 (2025-04-17)
===========================

Features
--------

- Support the :class:`cirq.R` reset operation in ``cirq_iqm``. :issue:`SW-795`

Version 24.0.0 (2025-04-17)
===========================

Features
--------

- Add ``timeout`` argument and flag to :meth:`IQMJob.result`, :issue:`SW-1308`.

Breaking changes
----------------

- Remove ``timeout_seconds`` from :class:`IQMJob` and :meth:`IQMBackend.run`.


Version 23.8.0 (2025-04-11)
===========================

Bug fixes
---------

- Fix broken link in docs to Cirq user guide

Version 23.7.0 (2025-04-11)
===========================

Bug fixes
---------

- Update license

Version 23.6.0 (2025-04-10)
===========================

Features
--------

- fix flaky e2e tests

Version 23.5.0 (2025-04-09)
===========================

Bug fixes
---------

- Add STATIC_QUANTUM_ARCHITECTURE to RESONANCE_COCOS_V1 api

Version 23.4.0 (2025-04-09)
===========================

Bug fixes
---------

- Fix missing api docs for :mod:`iqm.qiskit_iqm` and :mod:`iqm.cirq_iqm`.

Version 23.3.0 (2025-04-09)
===========================

Bug fixes
---------

- Fix links in readme to be compatible with PyPI publishing.

Version 23.2.0 (2025-04-09)
===========================

Features
--------

- ``iqm.cortex_cli`` is moved inside ``iqm-client`` to a new submodule ``iqm.iqm_client.cli``. The corresponding ``cli``
  extra dependency can be installed as ``iqm-client[cli]``. :issue:`SW-1145`

Version 23.1.0 (2025-04-07)
===========================

Features
--------

- Fix package version in published docs footers, :issue:`SW-1392`. 

Version 23.0.0 (2025-04-04)
===========================

Features
--------

- Replaced the old quantum architecture in :class:`IQMBackendBase`.

Version 22.16.0 (2025-04-03)
============================

Feature
*******

- Enable PEP 604 in linting rules, :issue:`SW-1230`.

Version 22.15.0 (2025-04-03)
============================

Features
--------

- Add versioning for station control API. :issue:`SW-898`

Version 22.14.0 (2025-04-02)
============================

Features
********

- Update the documentation footer to display the package version.

Version 22.13.0 (2025-04-02)
============================

Features
--------

- Add ``cirq_iqm`` to ``iqm-client`` distribution package as an optional feature. :issue:`SW-1145`

Version 22.12.0 (2025-03-31)
============================

Features
--------

- :meth:`IQMClient.get_static_quantum_architecture` added. :issue:`SW-706`
- :class:`iqm.qiskit_iqm.fake_backends.IQMFakeBackend` uses the static quantum architecture. :issue:`SW-706`

Version 22.11.0 (2025-03-25)
============================

Features
--------

- Improve ``qiskit_iqm`` installation instructions and update links to ``qiskit_iqm`` documentation.

Version 22.10.0 (2025-03-24)
============================

Features
--------

- Add ``qiskit_iqm`` to ``iqm-client`` distribution package as an optional feature. :issue:`SW-1146`

Version 22.9.0 (2025-03-17)
===========================

Features
--------

- Restore Python 3.10 support

Version 22.8.0 (2025-03-10)
===========================

Features
--------

- IQMClient tolerates unrecognized job statuses from the server to ensure forward compatibility.

Version 22.7.0 (2025-03-07)
===========================

Features
--------

- :attr:`Instruction.args` now has a default value (an empty dict) for convenience. :issue:`SW-1289`

Bug fixes
---------

- :func:`transpile_insert_moves` now handles ``barrier`` instructions properly. :issue:`SW-1289`

Version 22.6.0 (2025-03-07)
===========================

Features
--------

* Improve docstrings.
* Remove boilerplate code in :class:`IQMClient` endpoint requests, make the error behavior more uniform.
* Speed up the unit tests by mocking ``sleep`` calls, do not send out actual HTTP requests.

Version 22.5.0 (2025-03-05)
===========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-1015`
- Drop support for Python 3.10.

Version 22.4 (2025-02-25)
=========================

* Python 3.10 support is deprecated and will be removed in the future.
  `#173 <https://github.com/iqm-finland/iqm-client/pull/173>`_


Version 22.3 (2025-02-24)
=========================

* Add the native ``reset`` operation. `#170 <https://github.com/iqm-finland/iqm-client/pull/170>`_


Version 22.2 (2025-02-20)
=========================

* Improve the documentation of the ``delay`` operation.
* Improve the performance of :func:`.transpile_insert_moves`.
  `#168 <https://github.com/iqm-finland/iqm-client/pull/168>`_


Version 22.1 (2025-02-18)
=========================

* The V1 API is deprecated and will be removed in a future release. Please use the V2 API instead. `#167 <https://github.com/iqm-finland/iqm-client/pull/167>`_


Version 22.0 (2025-02-14)
=========================

* Refactor the transpilation code, make its details private, improve the docs.
  `#156 <https://github.com/iqm-finland/iqm-client/pull/156>`_
* By default :func:`.transpile_insert_moves` now keeps any existing MOVE gates in the circuit.
  `#156 <https://github.com/iqm-finland/iqm-client/pull/156>`_
* Add the ``delay`` operation.  `#156 <https://github.com/iqm-finland/iqm-client/pull/156>`_


Version 20.17 (2025-02-14)
==========================

* Fix DQA deserialization when ``override_default_implementations`` is not empty.
  `#166 <https://github.com/iqm-finland/iqm-client/pull/166>`_


Version 20.16 (2025-02-07)
==========================

* Define ``DDStrategy`` as Pydantic ``BaseModel`` `#153 <https://github.com/iqm-finland/iqm-client/pull/153>`_
* Add unit tests to test ``RunRequest`` with dynamical decoupling `#153 <https://github.com/iqm-finland/iqm-client/pull/153>`_


Version 20.15 (2025-02-03)
==========================

* Relax version ranges of ``numpy``, ``packaging``. `#165 <https://github.com/iqm-finland/iqm-client/pull/165>`_


Version 20.14 (2025-01-28)
==========================

* Add ``IQMClient::get_feedback_groups`` method. `#162 <https://github.com/iqm-finland/iqm-client/pull/162>`_


Version 20.13 (2025-01-13)
==========================

* Fix package publishing from ci, no functional changes. `#160 <https://github.com/iqm-finland/iqm-client/pull/160>`_


Version 20.12 (2025-01-13)
==========================

* Drop support for Python 3.9. `#159 <https://github.com/iqm-finland/iqm-client/pull/159>`_
* Add optional security-scanned lockfile. `#159 <https://github.com/iqm-finland/iqm-client/pull/159>`_


Version 20.11 (2025-01-03)
==========================

* Add ``RESONANCE_COCOS_V1`` API variant option for Resonance Cocos API v1. `#158 <https://github.com/iqm-finland/iqm-client/pull/158>`_
* Add ``IQMClient::get_run_counts`` method. `#158 <https://github.com/iqm-finland/iqm-client/pull/158>`_
* Add ``IQMClient::get_supported_client_libraries`` method. `#158 <https://github.com/iqm-finland/iqm-client/pull/158>`_


Version 20.10 (2024-12-17)
==========================

* Fix Sphinx documentation build warnings `#155 <https://github.com/iqm-finland/iqm-client/pull/155>`_
* Enable Sphinx documentation build option to treat warnings as errors `#155 <https://github.com/iqm-finland/iqm-client/pull/155>`_


Version 20.9 (2024-12-14)
=========================

* Added Python 3.12 support `#154 <https://github.com/iqm-finland/iqm-client/pull/154>`_
* Python 3.9 support is deprecated and will be removed in the future


Version 20.8 (2024-11-29)
=========================

* Add ``dd_mode`` and ``dd_strategy`` to ``CircuitCompilationOptions`` and ``RunRequest`` `#152 <https://github.com/iqm-finland/iqm-client/pull/152>`_


Version 20.7 (2024-11-26)
=========================

* Fix typo of `QUALITY` in `src/iqm/iqm_client/api.py` `#149 <https://github.com/iqm-finland/iqm-client/pull/149>`_


Version 20.6 (2024-11-21)
=========================

* Improve version compatibility check to avoid it preventing usage of the client in any situation. `#150 <https://github.com/iqm-finland/iqm-client/pull/150>`_


Version 20.5 (2024-11-19)
=========================

* Fixed client version compatibility check. `#148 <https://github.com/iqm-finland/iqm-client/pull/148>`_


Version 20.4 (2024-11-18)
=========================
* ``active_reset_cycles`` added to ``CircuitCompilationOptions`` (in 20.2 it was only added to ``RunRequest`` making it
  difficult to use).


Version 20.3 (2024-11-15)
=========================

* Add warning when initializing client with server that has incompatible version. `#145 <https://github.com/iqm-finland/iqm-client/pull/145>`_
* Improve error message when an endpoint returns a 404 error due to the server version not supporting the endpoint. `#145 <https://github.com/iqm-finland/iqm-client/pull/145>`_


Version 20.2 (2024-11-15)
=========================

* Add ``active_reset_cycles`` circuit execution option, used for deciding between reset-by-wait and active reset (and how
  active reset cycles). `#146 <https://github.com/iqm-finland/iqm-client/pull/146>`_


Version 20.1 (2024-10-30)
=========================

* Disable attestations on ``gh-action-pypi-publish`` to fix failing PyPI publishing `#143 <https://github.com/iqm-finland/iqm-client/pull/143>`_


Version 20.0 (2024-10-30)
=========================

* Use dynamic quantum architecture for transpilation and validation. `#140 <https://github.com/iqm-finland/iqm-client/pull/140>`_
* Bugfix: ``cc_prx`` params fixed. `#140 <https://github.com/iqm-finland/iqm-client/pull/140>`_


Version 19.0 (2024-12-16)
=========================

* Allow mid-circuit measurements and classically controlled PRX gates.
  `#136 <https://github.com/iqm-finland/iqm-client/pull/136>`_
* Deprecated native operations names ``phased_rx`` and ``measurement`` removed,
  use ``prx`` and ``measure`` instead.
  `#136 <https://github.com/iqm-finland/iqm-client/pull/136>`_


Version 18.8 (2024-10-17)
=========================

* Fix MOVE gate validation for qubit mappings containing only some of the architecture qubits `#137 <https://github.com/iqm-finland/iqm-client/pull/137>`_


Version 18.7 (2024-10-16)
=========================

* Fix list of endpoints supported by the V1 API. `#138 <https://github.com/iqm-finland/iqm-client/pull/138>`_


Version 18.6 (2024-10-16)
=========================

* Add IQM Server API versioning support. `#135 <https://github.com/iqm-finland/iqm-client/pull/135>`_


Version 18.5 (2024-10-15)
=========================

* Added ``isort`` formatting to the tox configuration, so it is automatically run when running
  ``tox -e format``. `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_
* Bugfix: Fix the issue where the :class:`CircuitCompilationOptions` was not used in local circuit
  validation when using the :meth:`submit_circuit` method. Improved testing to catch the bug.
  `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_
* Bugfix: MOVE gate validation now also works with more than one resonator. `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_
* More specific validation and transpilation errors. `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_
* Docs updated: mid-circuit measurements are allowed on stations with ``cocos >= 30.2``. `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_
* Integration guide updated. `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_
* Circuit validation: All measurement keys must be unique. `#130 <https://github.com/iqm-finland/iqm-client/pull/130>`_


Version 18.4 (2024-10-04)
=========================

* Do not verify external auth token expiration. This fixes IQM Resonance authentication. `#134 <https://github.com/iqm-finland/iqm-client/pull/134>`_


Version 18.3 (2024-10-01)
=========================

* Remove unnecessary build files when publishing documentation. `#133 <https://github.com/iqm-finland/iqm-client/pull/133>`_


Version 18.2 (2024-10-01)
=========================

* Add mitigation for failed authentication sessions. `#132 <https://github.com/iqm-finland/iqm-client/pull/132>`_


Version 18.1 (2024-09-19)
=========================

* Add :meth:`IQMClient.get_dynamic_quantum_architecture`. `#131 <https://github.com/iqm-finland/iqm-client/pull/131>`_


Version 18.0 (2024-10-16)
=========================

* Added the naive MOVE transpilation method for unified transpilation behavior for different external APIs. `#124 <https://github.com/iqm-finland/iqm-client/pull/124>`_
* Added class for compilation options :class:`CircuitCompilationOptions` to allow for more fine-grained control over the compilation process. (breaking change)

  * :meth:`IQMClient.submit_circuit` now takes a :class:`CircuitCompilationOptions` parameter instead of ``max_circuit_duration_over_t2`` and ``heralding_mode``.
  * Moved the existing ``max_circuit_duration_over_t2`` parameter to :class:`CircuitCompilationOptions`.
  * Moved the existing ``heralding_mode`` parameter to :class:`CircuitCompilationOptions`.
  * Introduced new option ``move_gate_validation`` to turn off MOVE gate validation during compilation (ADVANCED).
  * Introduced new option ``move_gate_frame_tracking`` to turn off frame tracking for the MOVE gate (ADVANCED).
  * New options can only be used on stations with ``CoCoS`` version 29.9 or later that support the MOVE gate instruction. Otherwise, the options will be ignored.


Version 17.8 (2024-08-26)
=========================

* Allow inspecting a run request before submitting it for execution. `#129 <https://github.com/iqm-finland/iqm-client/pull/129>`_


Version 17.7 (2024-06-11)
=========================

* Update documentation. `#128 <https://github.com/iqm-finland/iqm-client/pull/128>`_


Version 17.6 (2024-05-21)
=========================

* Move all data models to ``iqm.iqm_client.models``. `#125 <https://github.com/iqm-finland/iqm-client/pull/125>`_
* Refactor user authentication and check authentication parameters for conflicts. `#125 <https://github.com/iqm-finland/iqm-client/pull/125>`_


Version 17.5 (2024-05-06)
=========================

* Show full response error in all cases of receiving a HTTP 4xx error response. `#123 <https://github.com/iqm-finland/iqm-client/pull/123>`_


Version 17.4 (2024-04-26)
=========================

* Raise ClientConfigurationError and display the details of the error upon receiving a HTTP 400 error response. `#120 <https://github.com/iqm-finland/iqm-client/pull/120>`_


Version 17.3 (2024-04-24)
=========================

* Add new job states to support job delete operation in the backend. `#119 <https://github.com/iqm-finland/iqm-client/pull/119>`_


Version 17.2 (2024-03-18)
=========================

* Use GitHub Action as a Trusted Publisher to publish packages to PyPI. `#116 <https://github.com/iqm-finland/iqm-client/pull/116>`_


Version 17.1 (2024-03-08)
=========================

* Support both extended and simple quantum architecture specification. `#117 <https://github.com/iqm-finland/iqm-client/pull/117>`_


Version 17.0 (2024-03-07)
=========================

* Extend quantum architecture specification to allow different loci for each operation. `#112 <https://github.com/iqm-finland/iqm-client/pull/112>`_
* Allow the ``move`` instruction natively.
* Validate instructions loci based on quantum architecture.
* Auto-rename deprecated instruction names to current names.


Version 16.1 (2024-02-26)
=========================

* Remove multiversion documentation. `#115 <https://github.com/iqm-finland/iqm-client/pull/115>`_


Version 16.0 (2024-02-07)
=========================

* Remove ``circuit_duration_check`` parameter from ``RunRequest``. `#114 <https://github.com/iqm-finland/iqm-client/pull/114>`_
* Add ``max_circuit_duration_over_t2`` parameter to ``RunRequest`` to control circuit disqualification threshold. `#114 <https://github.com/iqm-finland/iqm-client/pull/114>`_


Version 15.4 (2024-01-30)
=========================

* Add testing with python 3.11. `#113 <https://github.com/iqm-finland/iqm-client/pull/113>`_


Version 15.3 (2024-01-12)
=========================

* Make network request timeouts reconfigurable for ``abort_job``, ``get_quantum_architecture``, ``get_run``, and ``get_run_status`` via keyword argument ``timeout_secs``. `#110 <https://github.com/iqm-finland/iqm-client/pull/110>`_
* Make network request timeouts reconfigurable globally via environment variable ``IQM_CLIENT_REQUESTS_TIMEOUT``. `#110 <https://github.com/iqm-finland/iqm-client/pull/110>`_


Version 15.2 (2023-12-20)
=========================

* Allow construction of ``Circuit.instructions``  from a ``tuple`` of ``dict``. `#109 <https://github.com/iqm-finland/iqm-client/pull/109>`_


Version 15.1 (2023-12-19)
=========================

* Bump ``pydantic`` version to ``2.4.2``. `#108 <https://github.com/iqm-finland/iqm-client/pull/108>`_


Version 15.0 (2023-12-15)
=========================

* Update project setup to use ``pyproject.toml``. `#107 <https://github.com/iqm-finland/iqm-client/pull/107>`_
* New instruction names: ``phased_rx`` -> ``prx``, ``measurement`` -> ``measure`` (the old names are deprecated
  but still supported). `#107 <https://github.com/iqm-finland/iqm-client/pull/107>`_


Version 14.7 (2023-12-07)
=========================

* Add API token support. `#102 <https://github.com/iqm-finland/iqm-client/pull/102>`_


Version 14.6 (2023-11-17)
=========================

* Add CoCoS version to job metadata. `#104 <https://github.com/iqm-finland/iqm-client/pull/104>`_


Version 14.5 (2023-11-15)
=========================

* Add platform version and python version to user agent. `#103 <https://github.com/iqm-finland/iqm-client/pull/103>`_


Version 14.4 (2023-11-14)
=========================

* Require number of shots to be greater than zero. `#101 <https://github.com/iqm-finland/iqm-client/pull/101>`_


Version 14.3 (2023-11-08)
=========================

* Update integration guide. `#99 <https://github.com/iqm-finland/iqm-client/pull/99>`_


Version 14.2 (2023-11-08)
=========================

* Use ``get_run_status`` instead of ``get_run`` to check job status in ``wait_for_compilation`` and ``wait_for_results``. `#100 <https://github.com/iqm-finland/iqm-client/pull/100>`_


Version 14.1 (2023-10-19)
=========================

* Use latest version of ``sphinx-multiversion-contrib`` to fix documentation version sorting. `#98 <https://github.com/iqm-finland/iqm-client/pull/98>`_


Version 14.0 (2023-09-15)
=========================

* Move ``iqm_client`` package to ``iqm`` namespace. `#96 <https://github.com/iqm-finland/iqm-client/pull/96>`_


Version 13.4 (2023-09-11)
=========================

* Update integration guide. `#95 <https://github.com/iqm-finland/iqm-client/pull/95>`_



Version 13.3 (2023-08-30)
=========================

* Improve tests. `#94 <https://github.com/iqm-finland/iqm-client/pull/94>`_


Version 13.2 (2023-08-25)
=========================

* Use ISO 8601 format timestamps in RunResult metadata. `#93 <https://github.com/iqm-finland/iqm-client/pull/93>`_


Version 13.1 (2023-08-11)
=========================

* Add execution timestamps in RunResult metadata. `#92 <https://github.com/iqm-finland/iqm-client/pull/92>`_


Version 13.0 (2023-07-03)
=========================

* Add ability to abort jobs. `#89 <https://github.com/iqm-finland/iqm-client/pull/89>`_


Version 12.5 (2023-05-25)
=========================

* Add parameter ``heralding`` to ``RunRequest``. `#87 <https://github.com/iqm-finland/iqm-client/pull/87>`_


Version 12.4 (2023-05-25)
=========================

* Add parameter ``circuit_duration_check`` allowing to control server-side maximum circuit duration check. `#85 <https://github.com/iqm-finland/iqm-client/pull/85>`_


Version 12.3 (2023-05-03)
=========================

* Generate license information for dependencies on every release `#84 <https://github.com/iqm-finland/iqm-client/pull/84>`_


Version 12.2 (2023-04-21)
=========================

* Revert moving Pydantic model definitions into ``models.py`` file. `#81 <https://github.com/iqm-finland/iqm-client/pull/81>`_


Version 12.1 (2023-04-20)
=========================

* Add function ``validate_circuit`` to validate a submitted circuit for input argument correctness. `#80 <https://github.com/iqm-finland/iqm-client/pull/80>`_


Version 12.0 (2023-04-18)
=========================

* Split ``PENDING`` job status into ``PENDING_COMPILATION`` and ``PENDING_EXECUTION`` `#79 <https://github.com/iqm-finland/iqm-client/pull/79>`_
* Add ``wait_for_compilation`` method. `#79 <https://github.com/iqm-finland/iqm-client/pull/79>`_


Version 11.8 (2023-03-28)
=========================

* Bugfix: multiversion documentation has incomplete lists to available documentation versions `#76 <https://github.com/iqm-finland/iqm-client/pull/76>`_


Version 11.7 (2023-03-10)
=========================

* Add utility function ``to_json_dict`` to convert a dict to a JSON dict. `#77 <https://github.com/iqm-finland/iqm-client/pull/77>`_


Version 11.6 (2023-02-23)
=========================

* Improve error reporting on unexpected server responses. `#74 <https://github.com/iqm-finland/iqm-client/pull/74>`_


Version 11.5 (2023-02-23)
=========================

* Improve multiversion docs builds. `#75 <https://github.com/iqm-finland/iqm-client/pull/75>`_


Version 11.4 (2023-02-10)
=========================

* Add user agent header to requests. `#72 <https://github.com/iqm-finland/iqm-client/pull/72>`_


Version 11.3 (2023-02-09)
=========================

* Fix multiversion docs publication. `#73 <https://github.com/iqm-finland/iqm-client/pull/73>`_


Version 11.2 (2023-02-06)
=========================

* Reduce docs size. `#71 <https://github.com/iqm-finland/iqm-client/pull/71>`_


Version 11.1 (2023-01-26)
=========================

* Fix docs version sort. `#70 <https://github.com/iqm-finland/iqm-client/pull/70>`_


Version 11.0 (2023-01-20)
=========================

* Change type of ``calibration_set_id`` to be opaque UUID. `#69 <https://github.com/iqm-finland/iqm-client/pull/69>`_


Version 10.3 (2023-01-04)
=========================

* Remove ``description`` from pydantic model fields. `#68 <https://github.com/iqm-finland/iqm-client/pull/68>`_


Version 10.2 (2022-12-29)
=========================

* Add optional ``implementation`` field to ``Instruction``. `#67 <https://github.com/iqm-finland/iqm-client/pull/67>`_


Version 10.1 (2022-12-28)
=========================

* Raise an error while fetching quantum architecture if authentication is not provided. `#66 <https://github.com/iqm-finland/iqm-client/pull/66>`_


Version 10.0 (2022-12-28)
=========================

* ``RunResult.metadata.request`` now contains a copy of the original request. `#65 <https://github.com/iqm-finland/iqm-client/pull/65>`_


Version 9.8 (2022-12-20)
========================

* Bugfix: ``Circuit.metadata`` Pydantic field needs default value. `#64 <https://github.com/iqm-finland/iqm-client/pull/64>`_


Version 9.7 (2022-12-20)
========================

* Add optional ``metadata`` field to ``Circuit``. `#63 <https://github.com/iqm-finland/iqm-client/pull/63>`_


Version 9.6 (2022-12-14)
========================

* Reduce wait interval between requests to the IQM Server and make it configurable with the ``IQM_CLIENT_SECONDS_BETWEEN_CALLS`` environment var. `#62 <https://github.com/iqm-finland/iqm-client/pull/66>`_


Version 9.5 (2022-12-05)
========================

* Retry requests to the IQM Server if the server is busy. `#61 <https://github.com/iqm-finland/iqm-client/pull/61>`_


Version 9.4 (2022-11-30)
========================

* Add integration guide. `#60 <https://github.com/iqm-finland/iqm-client/pull/60>`_


Version 9.3 (2022-11-23)
========================

* Support OpenTelemetry trace propagation. `#59 <https://github.com/iqm-finland/iqm-client/pull/59>`_


Version 9.2 (2022-11-17)
========================

* New external token is now obtained from tokens file if old token expired. `#58 <https://github.com/iqm-finland/iqm-client/pull/58>`_


Version 9.1 (2022-10-20)
========================

* Update documentation. `#57 <https://github.com/iqm-finland/iqm-client/pull/57>`_


Version 9.0 (2022-10-19)
========================

* The method ``IQMClient.get_quantum_architecture`` now return the architecture specification instead of the top level object. `#56 <https://github.com/iqm-finland/iqm-client/pull/56>`_


Version 8.4 (2022-10-17)
========================

* Update documentation of Metadata. `#54 <https://github.com/iqm-finland/iqm-client/pull/54>`_


Version 8.3 (2022-10-17)
========================

* Improved error message when ``qubit_mapping`` does not cover all qubits in a circuit. `#53 <https://github.com/iqm-finland/iqm-client/pull/53>`_
* Better type definitions and code cleanup. `#53 <https://github.com/iqm-finland/iqm-client/pull/53>`_, `#52 <https://github.com/iqm-finland/iqm-client/pull/52>`_


Version 8.2 (2022-10-10)
========================

* Add method ``IQMClient.get_quantum_architecture``. `#51 <https://github.com/iqm-finland/iqm-client/pull/51>`_


Version 8.1 (2022-09-30)
========================

* Change ``Circuit.instructions`` and ``Instruction.qubits`` from list to tuple. `#49 <https://github.com/iqm-finland/iqm-client/pull/49>`_


Version 8.0 (2022-09-28)
========================

* Remove settings from RunRequest, add custom_settings. `#48 <https://github.com/iqm-finland/iqm-client/pull/48>`_


Version 7.3 (2022-09-28)
========================

* Increase job result poll interval while waiting for circuit execution. `#47 <https://github.com/iqm-finland/iqm-client/pull/47>`_


Version 7.2 (2022-09-08)
========================

* Add description of calibration set ID of RunResult metadata in the documentation. `#45 <https://github.com/iqm-finland/iqm-client/pull/45>`_


Version 7.1 (2022-09-08)
========================

* Increase timeout of requests. `#43 <https://github.com/iqm-finland/iqm-client/pull/43>`_


Version 7.0 (2022-09-02)
========================

* Add calibration set ID to RunResult metadata. `#42 <https://github.com/iqm-finland/iqm-client/pull/42>`_


Version 6.2 (2022-08-29)
========================

* Enable mypy checks. `#41 <https://github.com/iqm-finland/iqm-client/pull/41>`_
* Update source code according to new checks in pylint v2.15.0. `#41 <https://github.com/iqm-finland/iqm-client/pull/41>`_


Version 6.1 (2022-08-16)
========================

* Add optional ``calibration_set_id`` parameter to ``IQMClient.submit_circuit``. `#40 <https://github.com/iqm-finland/iqm-client/pull/40>`_


Version 6.0 (2022-08-12)
========================

* ``IQMClient.close`` renamed to ``IQMClient.close_auth_session`` and raises an exception when asked to close an externally managed authentication session. `#39 <https://github.com/iqm-finland/iqm-client/pull/39>`_
* Try to automatically close the authentication session when the client is deleted. `#39 <https://github.com/iqm-finland/iqm-client/pull/39>`_
* Show CoCoS error on 401 response. `#39 <https://github.com/iqm-finland/iqm-client/pull/39>`_


Version 5.0 (2022-08-09)
========================

* ``settings`` are moved from the constructor of ``IQMClient`` to ``IQMClient.submit_circuit``. `#31 <https://github.com/iqm-finland/iqm-client/pull/31>`_
* Changed the type of ``qubit_mapping`` argument of ``IQMClient.submit_circuit`` to ``dict[str, str]``. `#31 <https://github.com/iqm-finland/iqm-client/pull/31>`_
* User can now import from iqm_client using `from iqm_client import x` instead of `from iqm_client.iqm_client import x`. `#31 <https://github.com/iqm-finland/iqm-client/pull/31>`_


Version 4.3 (2022-08-03)
========================

* Parse new field metadata for job result requests to the IQM quantum computer. `#37 <https://github.com/iqm-finland/iqm-client/pull/37>`_


Version 4.2 (2022-07-20)
========================

* Update documentation to include development version and certain released versions in a subdirectory. `#36 <https://github.com/iqm-finland/iqm-client/pull/36>`_


Version 4.1 (2022-07-12)
========================

* Add support for authentication without username/password, using externally managed tokens file. `#35 <https://github.com/iqm-finland/iqm-client/pull/35>`_


Version 4.0 (2022-06-28)
========================

* Implement functionality to submit a batch of circuits in one job. `#34 <https://github.com/iqm-finland/iqm-client/pull/34>`_


Version 3.3 (2022-06-02)
========================

* Make ``settings`` an optional parameter for ``IQMClient``. `#30 <https://github.com/iqm-finland/iqm-client/pull/30>`_


Version 3.2 (2022-06-02)
========================

* Add function ``get_run_status`` to check status of execution without getting measurement results. `#29 <https://github.com/iqm-finland/iqm-client/pull/29>`_


Version 3.1 (2022-11-17)
========================

* Update documentation to mention barriers. `#28 <https://github.com/iqm-finland/iqm-client/pull/28>`_


Version 3.0 (2022-05-17)
========================

* Update HTTP endpoints for circuit execution and results retrieval. `#26 <https://github.com/iqm-finland/iqm-client/pull/26>`_
* Requires CoCoS 4.0


Version 2.2 (2022-04-26)
========================

* Publish JSON schema for the circuit run request sent to an IQM server. `#24 <https://github.com/iqm-finland/iqm-client/pull/24>`_


Version 2.1 (2022-04-19)
========================

* Add support for Python 3.10. `#23 <https://github.com/iqm-finland/iqm-client/pull/23>`_


Version 2.0 (2022-03-25)
========================

* Update user authentication to use access token. `#22 <https://github.com/iqm-finland/iqm-client/pull/22>`_
* Add token management to IQMClient. `#22 <https://github.com/iqm-finland/iqm-client/pull/22>`_


Version 1.10 (2022-02-22)
=========================

* Make ``qubit_mapping`` an optional parameter in ``IQMClient.submit_circuit``. `#21 <https://github.com/iqm-finland/iqm-client/pull/21>`_


Version 1.9 (2022-02-22)
========================

* Validate that the schema of IQM server URL is http or https. `#20 <https://github.com/iqm-finland/iqm-client/pull/20>`_


Version 1.8 (2022-02-01)
========================

* Add 'Expect: 100-Continue' header to the post request. `#18 <https://github.com/iqm-finland/iqm-client/pull/18>`_
* Bump pydantic dependency. `#13 <https://github.com/iqm-finland/iqm-client/pull/13>`_
* Minor updates in docs. `#13 <https://github.com/iqm-finland/iqm-client/pull/13>`_


Version 1.7 (2022-01-25)
========================

* Emit warnings in server response as python UserWarning. `#15 <https://github.com/iqm-finland/iqm-client/pull/15>`_


Version 1.6 (2021-12-15)
========================

* Configure automatic tagging and releasing. `#7 <https://github.com/iqm-finland/iqm-client/pull/7>`_


Version 1.5 (2021-11-23)
========================

* Implement HTTP Basic auth. `#9 <https://github.com/iqm-finland/iqm-client/pull/9>`_


Version 1.4 (2021-11-05)
========================

* Increase default timeout. `#8 <https://github.com/iqm-finland/iqm-client/pull/8>`_


Version 1.3 (2021-10-20)
========================

Features
--------

* Document the native instruction types. `#5 <https://github.com/iqm-finland/iqm-client/pull/5>`_



Version 1.2 (2021-10-19)
========================

Fixes
-----

* Remove unneeded args field from Circuit. `#4 <https://github.com/iqm-finland/iqm-client/pull/4>`_



Version 1.1 (2021-10-08)
========================

Fixes
-----

* Changed example instruction phased_rx to measurement. `#2 <https://github.com/iqm-finland/iqm-client/pull/2>`_



Version 1.0 (2021-08-27)
========================

Features
--------

* Split IQM client from the Cirq on IQM library
