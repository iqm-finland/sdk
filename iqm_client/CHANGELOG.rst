=========
Changelog
=========

Version 22.4
============

* Python 3.10 support is deprecated and will be removed in the future.
  `#173 <https://github.com/iqm-finland/iqm-client/pull/173>`_

Version 22.3
============

* Add the native ``reset`` operation. `#170 <https://github.com/iqm-finland/iqm-client/pull/170>`_

Version 22.2
============

* Improve the documentation of the ``delay`` operation.
* Improve the performance of :func:`.transpile_insert_moves`.
  `#168 <https://github.com/iqm-finland/iqm-client/pull/168>`_

Version 22.1
=============

* The V1 API is deprecated and will be removed in a future release. Please use the V2 API instead. `#167 <https://github.com/iqm-finland/iqm-client/pull/167>`_

Version 22.0
============

* Refactor the transpilation code, make its details private, improve the docs.
  `#156 <https://github.com/iqm-finland/iqm-client/pull/156>`_
* By default :func:`.transpile_insert_moves` now keeps any existing MOVE gates in the circuit.
  `#156 <https://github.com/iqm-finland/iqm-client/pull/156>`_
* Add the ``delay`` operation.  `#156 <https://github.com/iqm-finland/iqm-client/pull/156>`_

Version 20.17
=============

* Fix DQA deserialization when ``override_default_implementations`` is not empty.
  `#166 <https://github.com/iqm-finland/iqm-client/pull/166>`_

Version 20.16
=============

* Define ``DDStrategy`` as Pydantic ``BaseModel`` `#153 <https://github.com/iqm-finland/iqm-client/pull/153>`_
* Add unit tests to test ``RunRequest`` with dynamical decoupling `#153 <https://github.com/iqm-finland/iqm-client/pull/153>`_

Version 20.15
=============

* Relax version ranges of ``numpy``, ``packaging``. `#165 <https://github.com/iqm-finland/iqm-client/pull/165>`_

Version 20.14
=============

* Add ``IQMClient::get_feedback_groups`` method. `#162 <https://github.com/iqm-finland/iqm-client/pull/162>`_

Version 20.13
=============

* Fix package publishing from ci, no functional changes. `#160 <https://github.com/iqm-finland/iqm-client/pull/160>`_

Version 20.12
=============

* Drop support for Python 3.9. `#159 <https://github.com/iqm-finland/iqm-client/pull/159>`_
* Add optional security-scanned lockfile. `#159 <https://github.com/iqm-finland/iqm-client/pull/159>`_

Version 20.11
=============

* Add ``RESONANCE_COCOS_V1`` API variant option for Resonance Cocos API v1. `#158 <https://github.com/iqm-finland/iqm-client/pull/158>`_
* Add ``IQMClient::get_run_counts`` method. `#158 <https://github.com/iqm-finland/iqm-client/pull/158>`_
* Add ``IQMClient::get_supported_client_libraries`` method. `#158 <https://github.com/iqm-finland/iqm-client/pull/158>`_

Version 20.10
=============

* Fix Sphinx documentation build warnings `#155 <https://github.com/iqm-finland/iqm-client/pull/155>`_
* Enable Sphinx documentation build option to treat warnings as errors `#155 <https://github.com/iqm-finland/iqm-client/pull/155>`_

Version 20.9
============

* Added Python 3.12 support `#154 <https://github.com/iqm-finland/iqm-client/pull/154>`_
* Python 3.9 support is deprecated and will be removed in the future

Version 20.8
============

* Add ``dd_mode`` and ``dd_strategy`` to ``CircuitCompilationOptions`` and ``RunRequest`` `#152 <https://github.com/iqm-finland/iqm-client/pull/152>`_

Version 20.7
============

* Fix typo of `QUALITY` in `src/iqm/iqm_client/api.py` `#149 <https://github.com/iqm-finland/iqm-client/pull/149>`_

Version 20.6
============

* Improve version compatibility check to avoid it preventing usage of the client in any situation. `#150 <https://github.com/iqm-finland/iqm-client/pull/150>`_

Version 20.5
============

* Fixed client version compatibility check. `#148 <https://github.com/iqm-finland/iqm-client/pull/148>`_

Version 20.4
============
* ``active_reset_cycles`` added to ``CircuitCompilationOptions`` (in 20.2 it was only added to ``RunRequest`` making it
  difficult to use).

Version 20.3
============

* Add warning when initializing client with server that has incompatible version. `#145 <https://github.com/iqm-finland/iqm-client/pull/145>`_
* Improve error message when an endpoint returns a 404 error due to the server version not supporting the endpoint. `#145 <https://github.com/iqm-finland/iqm-client/pull/145>`_

Version 20.2
============

* Add ``active_reset_cycles`` circuit execution option, used for deciding between reset-by-wait and active reset (and how
  active reset cycles). `#146 <https://github.com/iqm-finland/iqm-client/pull/146>`_

Version 20.1
============

* Disable attestations on ``gh-action-pypi-publish`` to fix failing PyPI publishing `#143 <https://github.com/iqm-finland/iqm-client/pull/143>`_

Version 20.0
============

* Use dynamic quantum architecture for transpilation and validation. `#140 <https://github.com/iqm-finland/iqm-client/pull/140>`_
* Bugfix: ``cc_prx`` params fixed. `#140 <https://github.com/iqm-finland/iqm-client/pull/140>`_

Version 19.0
============

* Allow mid-circuit measurements and classically controlled PRX gates.
  `#136 <https://github.com/iqm-finland/iqm-client/pull/136>`_
* Deprecated native operations names ``phased_rx`` and ``measurement`` removed,
  use ``prx`` and ``measure`` instead.
  `#136 <https://github.com/iqm-finland/iqm-client/pull/136>`_

Version 18.8
============

* Fix MOVE gate validation for qubit mappings containing only some of the architecture qubits `#137 <https://github.com/iqm-finland/iqm-client/pull/137>`_

Version 18.7
============

* Fix list of endpoints supported by the V1 API. `#138 <https://github.com/iqm-finland/iqm-client/pull/138>`_

Version 18.6
============

* Add IQM Server API versioning support. `#135 <https://github.com/iqm-finland/iqm-client/pull/135>`_

Version 18.5
============

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

Version 18.4
============

* Do not verify external auth token expiration. This fixes IQM Resonance authentication. `#134 <https://github.com/iqm-finland/iqm-client/pull/134>`_

Version 18.3
============

* Remove unnecessary build files when publishing documentation. `#133 <https://github.com/iqm-finland/iqm-client/pull/133>`_

Version 18.2
============

* Add mitigation for failed authentication sessions. `#132 <https://github.com/iqm-finland/iqm-client/pull/132>`_

Version 18.1
============

* Add :meth:`IQMClient.get_dynamic_quantum_architecture`. `#131 <https://github.com/iqm-finland/iqm-client/pull/131>`_

Version 18.0
============

* Added the naive MOVE transpilation method for unified transpilation behavior for different external APIs. `#124 <https://github.com/iqm-finland/iqm-client/pull/124>`_
* Added class for compilation options :class:`CircuitCompilationOptions` to allow for more fine-grained control over the compilation process. (breaking change)

  * :meth:`IQMClient.submit_circuit` now takes a :class:`CircuitCompilationOptions` parameter instead of ``max_circuit_duration_over_t2`` and ``heralding_mode``.
  * Moved the existing ``max_circuit_duration_over_t2`` parameter to :class:`CircuitCompilationOptions`.
  * Moved the existing ``heralding_mode`` parameter to :class:`CircuitCompilationOptions`.
  * Introduced new option ``move_gate_validation`` to turn off MOVE gate validation during compilation (ADVANCED).
  * Introduced new option ``move_gate_frame_tracking`` to turn off frame tracking for the MOVE gate (ADVANCED).
  * New options can only be used on stations with ``CoCoS`` version 29.9 or later that support the MOVE gate instruction. Otherwise, the options will be ignored.

Version 17.8
============

* Allow inspecting a run request before submitting it for execution. `#129 <https://github.com/iqm-finland/iqm-client/pull/129>`_

Version 17.7
============

* Update documentation. `#128 <https://github.com/iqm-finland/iqm-client/pull/128>`_

Version 17.6
============

* Move all data models to ``iqm.iqm_client.models``. `#125 <https://github.com/iqm-finland/iqm-client/pull/125>`_
* Refactor user authentication and check authentication parameters for conflicts. `#125 <https://github.com/iqm-finland/iqm-client/pull/125>`_

Version 17.5
============

* Show full response error in all cases of receiving a HTTP 4xx error response. `#123 <https://github.com/iqm-finland/iqm-client/pull/123>`_

Version 17.4
============

* Raise ClientConfigurationError and display the details of the error upon receiving a HTTP 400 error response. `#120 <https://github.com/iqm-finland/iqm-client/pull/120>`_

Version 17.3
============

* Add new job states to support job delete operation in the backend. `#119 <https://github.com/iqm-finland/iqm-client/pull/119>`_

Version 17.2
============

* Use GitHub Action as a Trusted Publisher to publish packages to PyPI. `#116 <https://github.com/iqm-finland/iqm-client/pull/116>`_

Version 17.1
============

* Support both extended and simple quantum architecture specification. `#117 <https://github.com/iqm-finland/iqm-client/pull/117>`_

Version 17.0
============

* Extend quantum architecture specification to allow different loci for each operation. `#112 <https://github.com/iqm-finland/iqm-client/pull/112>`_
* Allow the ``move`` instruction natively.
* Validate instructions loci based on quantum architecture.
* Auto-rename deprecated instruction names to current names.

Version 16.1
============

* Remove multiversion documentation. `#115 <https://github.com/iqm-finland/iqm-client/pull/115>`_

Version 16.0
============

* Remove ``circuit_duration_check`` parameter from ``RunRequest``. `#114 <https://github.com/iqm-finland/iqm-client/pull/114>`_
* Add ``max_circuit_duration_over_t2`` parameter to ``RunRequest`` to control circuit disqualification threshold. `#114 <https://github.com/iqm-finland/iqm-client/pull/114>`_

Version 15.4
============

* Add testing with python 3.11. `#113 <https://github.com/iqm-finland/iqm-client/pull/113>`_

Version 15.3
============

* Make network request timeouts reconfigurable for ``abort_job``, ``get_quantum_architecture``, ``get_run``, and ``get_run_status`` via keyword argument ``timeout_secs``. `#110 <https://github.com/iqm-finland/iqm-client/pull/110>`_
* Make network request timeouts reconfigurable globally via environment variable ``IQM_CLIENT_REQUESTS_TIMEOUT``. `#110 <https://github.com/iqm-finland/iqm-client/pull/110>`_

Version 15.2
============

* Allow construction of ``Circuit.instructions``  from a ``tuple`` of ``dict``. `#109 <https://github.com/iqm-finland/iqm-client/pull/109>`_

Version 15.1
============

* Bump ``pydantic`` version to ``2.4.2``. `#108 <https://github.com/iqm-finland/iqm-client/pull/108>`_

Version 15.0
============

* Update project setup to use ``pyproject.toml``. `#107 <https://github.com/iqm-finland/iqm-client/pull/107>`_
* New instruction names: ``phased_rx`` -> ``prx``, ``measurement`` -> ``measure`` (the old names are deprecated
  but still supported). `#107 <https://github.com/iqm-finland/iqm-client/pull/107>`_

Version 14.7
============

* Add API token support. `#102 <https://github.com/iqm-finland/iqm-client/pull/102>`_

Version 14.6
============

* Add CoCoS version to job metadata. `#104 <https://github.com/iqm-finland/iqm-client/pull/104>`_

Version 14.5
============

* Add platform version and python version to user agent. `#103 <https://github.com/iqm-finland/iqm-client/pull/103>`_

Version 14.4
============

* Require number of shots to be greater than zero. `#101 <https://github.com/iqm-finland/iqm-client/pull/101>`_

Version 14.3
============

* Update integration guide. `#99 <https://github.com/iqm-finland/iqm-client/pull/99>`_

Version 14.2
============

* Use ``get_run_status`` instead of ``get_run`` to check job status in ``wait_for_compilation`` and ``wait_for_results``. `#100 <https://github.com/iqm-finland/iqm-client/pull/100>`_

Version 14.1
============

* Use latest version of ``sphinx-multiversion-contrib`` to fix documentation version sorting. `#98 <https://github.com/iqm-finland/iqm-client/pull/98>`_

Version 14.0
============

* Move ``iqm_client`` package to ``iqm`` namespace. `#96 <https://github.com/iqm-finland/iqm-client/pull/96>`_

Version 13.4
============

* Update integration guide. `#95 <https://github.com/iqm-finland/iqm-client/pull/95>`_


Version 13.3
============

* Improve tests. `#94 <https://github.com/iqm-finland/iqm-client/pull/94>`_

Version 13.2
============

* Use ISO 8601 format timestamps in RunResult metadata. `#93 <https://github.com/iqm-finland/iqm-client/pull/93>`_

Version 13.1
============

* Add execution timestamps in RunResult metadata. `#92 <https://github.com/iqm-finland/iqm-client/pull/92>`_

Version 13.0
============

* Add ability to abort jobs. `#89 <https://github.com/iqm-finland/iqm-client/pull/89>`_

Version 12.5
============

* Add parameter ``heralding`` to ``RunRequest``. `#87 <https://github.com/iqm-finland/iqm-client/pull/87>`_

Version 12.4
============

* Add parameter ``circuit_duration_check`` allowing to control server-side maximum circuit duration check. `#85 <https://github.com/iqm-finland/iqm-client/pull/85>`_

Version 12.3
============

* Generate license information for dependencies on every release `#84 <https://github.com/iqm-finland/iqm-client/pull/84>`_

Version 12.2
============

* Revert moving Pydantic model definitions into ``models.py`` file. `#81 <https://github.com/iqm-finland/iqm-client/pull/81>`_

Version 12.1
============

* Add function ``validate_circuit`` to validate a submitted circuit for input argument correctness. `#80 <https://github.com/iqm-finland/iqm-client/pull/80>`_

Version 12.0
============

* Split ``PENDING`` job status into ``PENDING_COMPILATION`` and ``PENDING_EXECUTION`` `#79 <https://github.com/iqm-finland/iqm-client/pull/79>`_
* Add ``wait_for_compilation`` method. `#79 <https://github.com/iqm-finland/iqm-client/pull/79>`_

Version 11.8
============

* Bugfix: multiversion documentation has incomplete lists to available documentation versions `#76 <https://github.com/iqm-finland/iqm-client/pull/76>`_

Version 11.7
============

* Add utility function ``to_json_dict`` to convert a dict to a JSON dict. `#77 <https://github.com/iqm-finland/iqm-client/pull/77>`_

Version 11.6
============

* Improve error reporting on unexpected server responses. `#74 <https://github.com/iqm-finland/iqm-client/pull/74>`_

Version 11.5
============

* Improve multiversion docs builds. `#75 <https://github.com/iqm-finland/iqm-client/pull/75>`_

Version 11.4
============

* Add user agent header to requests. `#72 <https://github.com/iqm-finland/iqm-client/pull/72>`_

Version 11.3
============

* Fix multiversion docs publication. `#73 <https://github.com/iqm-finland/iqm-client/pull/73>`_

Version 11.2
============

* Reduce docs size. `#71 <https://github.com/iqm-finland/iqm-client/pull/71>`_

Version 11.1
============

* Fix docs version sort. `#70 <https://github.com/iqm-finland/iqm-client/pull/70>`_

Version 11.0
============

* Change type of ``calibration_set_id`` to be opaque UUID. `#69 <https://github.com/iqm-finland/iqm-client/pull/69>`_

Version 10.3
============

* Remove ``description`` from pydantic model fields. `#68 <https://github.com/iqm-finland/iqm-client/pull/68>`_

Version 10.2
============

* Add optional ``implementation`` field to ``Instruction``. `#67 <https://github.com/iqm-finland/iqm-client/pull/67>`_

Version 10.1
============

* Raise an error while fetching quantum architecture if authentication is not provided. `#66 <https://github.com/iqm-finland/iqm-client/pull/66>`_

Version 10.0
============

* ``RunResult.metadata.request`` now contains a copy of the original request. `#65 <https://github.com/iqm-finland/iqm-client/pull/65>`_

Version 9.8
===========

* Bugfix: ``Circuit.metadata`` Pydantic field needs default value. `#64 <https://github.com/iqm-finland/iqm-client/pull/64>`_

Version 9.7
===========

* Add optional ``metadata`` field to ``Circuit``. `#63 <https://github.com/iqm-finland/iqm-client/pull/63>`_

Version 9.6
===========

* Reduce wait interval between requests to the IQM Server and make it configurable with the ``IQM_CLIENT_SECONDS_BETWEEN_CALLS`` environment var. `#62 <https://github.com/iqm-finland/iqm-client/pull/66>`_

Version 9.5
===========

* Retry requests to the IQM Server if the server is busy. `#61 <https://github.com/iqm-finland/iqm-client/pull/61>`_

Version 9.4
===========

* Add integration guide. `#60 <https://github.com/iqm-finland/iqm-client/pull/60>`_

Version 9.3
===========

* Support OpenTelemetry trace propagation. `#59 <https://github.com/iqm-finland/iqm-client/pull/59>`_

Version 9.2
===========

* New external token is now obtained from tokens file if old token expired. `#58 <https://github.com/iqm-finland/iqm-client/pull/58>`_

Version 9.1
===========

* Update documentation. `#57 <https://github.com/iqm-finland/iqm-client/pull/57>`_

Version 9.0
===========

* The method ``IQMClient.get_quantum_architecture`` now return the architecture specification instead of the top level object. `#56 <https://github.com/iqm-finland/iqm-client/pull/56>`_

Version 8.4
===========

* Update documentation of Metadata. `#54 <https://github.com/iqm-finland/iqm-client/pull/54>`_

Version 8.3
===========

* Improved error message when ``qubit_mapping`` does not cover all qubits in a circuit. `#53 <https://github.com/iqm-finland/iqm-client/pull/53>`_
* Better type definitions and code cleanup. `#53 <https://github.com/iqm-finland/iqm-client/pull/53>`_, `#52 <https://github.com/iqm-finland/iqm-client/pull/52>`_

Version 8.2
===========

* Add method ``IQMClient.get_quantum_architecture``. `#51 <https://github.com/iqm-finland/iqm-client/pull/51>`_

Version 8.1
===========

* Change ``Circuit.instructions`` and ``Instruction.qubits`` from list to tuple. `#49 <https://github.com/iqm-finland/iqm-client/pull/49>`_

Version 8.0
===========

* Remove settings from RunRequest, add custom_settings. `#48 <https://github.com/iqm-finland/iqm-client/pull/48>`_

Version 7.3
===========

* Increase job result poll interval while waiting for circuit execution. `#47 <https://github.com/iqm-finland/iqm-client/pull/47>`_

Version 7.2
===========

* Add description of calibration set ID of RunResult metadata in the documentation. `#45 <https://github.com/iqm-finland/iqm-client/pull/45>`_

Version 7.1
===========

* Increase timeout of requests. `#43 <https://github.com/iqm-finland/iqm-client/pull/43>`_

Version 7.0
===========

* Add calibration set ID to RunResult metadata. `#42 <https://github.com/iqm-finland/iqm-client/pull/42>`_

Version 6.2
===========

* Enable mypy checks. `#41 <https://github.com/iqm-finland/iqm-client/pull/41>`_
* Update source code according to new checks in pylint v2.15.0. `#41 <https://github.com/iqm-finland/iqm-client/pull/41>`_

Version 6.1
===========

* Add optional ``calibration_set_id`` parameter to ``IQMClient.submit_circuit``. `#40 <https://github.com/iqm-finland/iqm-client/pull/40>`_

Version 6.0
===========

* ``IQMClient.close`` renamed to ``IQMClient.close_auth_session`` and raises an exception when asked to close an externally managed authentication session. `#39 <https://github.com/iqm-finland/iqm-client/pull/39>`_
* Try to automatically close the authentication session when the client is deleted. `#39 <https://github.com/iqm-finland/iqm-client/pull/39>`_
* Show CoCoS error on 401 response. `#39 <https://github.com/iqm-finland/iqm-client/pull/39>`_

Version 5.0
===========

* ``settings`` are moved from the constructor of ``IQMClient`` to ``IQMClient.submit_circuit``. `#31 <https://github.com/iqm-finland/iqm-client/pull/31>`_
* Changed the type of ``qubit_mapping`` argument of ``IQMClient.submit_circuit`` to ``dict[str, str]``. `#31 <https://github.com/iqm-finland/iqm-client/pull/31>`_
* User can now import from iqm_client using `from iqm_client import x` instead of `from iqm_client.iqm_client import x`. `#31 <https://github.com/iqm-finland/iqm-client/pull/31>`_

Version 4.3
===========

* Parse new field metadata for job result requests to the IQM quantum computer. `#37 <https://github.com/iqm-finland/iqm-client/pull/37>`_

Version 4.2
===========

* Update documentation to include development version and certain released versions in a subdirectory. `#36 <https://github.com/iqm-finland/iqm-client/pull/36>`_

Version 4.1
===========

* Add support for authentication without username/password, using externally managed tokens file. `#35 <https://github.com/iqm-finland/iqm-client/pull/35>`_

Version 4.0
===========

* Implement functionality to submit a batch of circuits in one job. `#34 <https://github.com/iqm-finland/iqm-client/pull/34>`_

Version 3.3
===========

* Make ``settings`` an optional parameter for ``IQMClient``. `#30 <https://github.com/iqm-finland/iqm-client/pull/30>`_

Version 3.2
===========

* Add function ``get_run_status`` to check status of execution without getting measurement results. `#29 <https://github.com/iqm-finland/iqm-client/pull/29>`_

Version 3.1
===========

* Update documentation to mention barriers. `#28 <https://github.com/iqm-finland/iqm-client/pull/28>`_

Version 3.0
===========

* Update HTTP endpoints for circuit execution and results retrieval. `#26 <https://github.com/iqm-finland/iqm-client/pull/26>`_
* Requires CoCoS 4.0

Version 2.2
===========

* Publish JSON schema for the circuit run request sent to an IQM server. `#24 <https://github.com/iqm-finland/iqm-client/pull/24>`_

Version 2.1
===========

* Add support for Python 3.10. `#23 <https://github.com/iqm-finland/iqm-client/pull/23>`_

Version 2.0
===========

* Update user authentication to use access token. `#22 <https://github.com/iqm-finland/iqm-client/pull/22>`_
* Add token management to IQMClient. `#22 <https://github.com/iqm-finland/iqm-client/pull/22>`_

Version 1.10
============

* Make ``qubit_mapping`` an optional parameter in ``IQMClient.submit_circuit``. `#21 <https://github.com/iqm-finland/iqm-client/pull/21>`_

Version 1.9
===========

* Validate that the schema of IQM server URL is http or https. `#20 <https://github.com/iqm-finland/iqm-client/pull/20>`_

Version 1.8
===========

* Add 'Expect: 100-Continue' header to the post request. `#18 <https://github.com/iqm-finland/iqm-client/pull/18>`_
* Bump pydantic dependency. `#13 <https://github.com/iqm-finland/iqm-client/pull/13>`_
* Minor updates in docs. `#13 <https://github.com/iqm-finland/iqm-client/pull/13>`_

Version 1.7
===========

* Emit warnings in server response as python UserWarning. `#15 <https://github.com/iqm-finland/iqm-client/pull/15>`_

Version 1.6
===========

* Configure automatic tagging and releasing. `#7 <https://github.com/iqm-finland/iqm-client/pull/7>`_

Version 1.5
===========

* Implement HTTP Basic auth. `#9 <https://github.com/iqm-finland/iqm-client/pull/9>`_

Version 1.4
===========

* Increase default timeout. `#8 <https://github.com/iqm-finland/iqm-client/pull/8>`_

Version 1.3
===========

Features
--------

* Document the native instruction types. `#5 <https://github.com/iqm-finland/iqm-client/pull/5>`_


Version 1.2
===========

Fixes
-----

* Remove unneeded args field from Circuit. `#4 <https://github.com/iqm-finland/iqm-client/pull/4>`_


Version 1.1
===========

Fixes
-----

* Changed example instruction phased_rx to measurement. `#2 <https://github.com/iqm-finland/iqm-client/pull/2>`_


Version 1.0
===========

Features
--------

* Split IQM client from the Cirq on IQM library
