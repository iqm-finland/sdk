=========
Changelog
=========

Version 9.7.0 (2025-07-10)
==========================

Bug fix
-------

- Better error handling when using IQM Server.

Version 9.6.0 (2025-07-09)
==========================

Features
--------

- Enable mypy type checking in CI and add temporary type ignores to the source code. :issue:`SW-1615`

Version 9.5.0 (2025-07-07)
==========================

Features
--------

- Improve documentation.

Version 9.4.0 (2025-06-26)
==========================

Bug fixes
---------

- Bump default timeout from 120 to 600

Version 9.3.0 (2025-06-25)
==========================

Bug fixes
---------

- Fix DQA deserialization when ``override_default_implementations`` is not empty. :issue:`SW-1625`

Version 9.2.0 (2025-06-25)
==========================

Bug fixes
---------

- Fix failing type checks in mypy.

Version 9.1.0 (2025-06-24)
==========================

Bug fixes
---------

- Requests now always have User-Agent header.

Version 9.0.0 (2025-06-13)
==========================

Features
--------

- Reintroduce :class:`.StationControlInterface` and make both :class:`.StationControlClient` and :class:`.IQMServerClient` inherit it
  to implement the same interface.
- Add new methods to :class:`StationControlInterface`, implemented by :class:`StationControlClient`. :class:`IQMServerClient` doesn't
  implement these new methods yet, but raises ``NotImplementedError`` instead. :issue:`SW-1078`

  - :meth:`get_default_calibration_set`
  - :meth:`get_default_calibration_set_observations`
  - :meth:`get_dynamic_quantum_architecture`
  - :meth:`get_default_dynamic_quantum_architecture`
  - :meth:`get_static_quantum_architecture`

- Support also string UUIDs in :class:`StationControlClient` methods, instead of only :class:`UUID` objects. This always worked in
  practice, since UUIDs were instantly serialized to strings anyway. However, it gave type warnings for the users
  when string given. No functional changes, but no more type warnings.

Breaking changes
----------------

- Remove :meth:`StationControlClient.init`, :meth:`StationControlClient.get_calibration_set_values` and
  :meth:`StationControlClient.get_latest_calibration_set_id` methods. Those aren't REST API related
  and needed only by Pulla for now, and they don't belong to :class:`StationControlInterface` since there is already
  :meth:`StationControlInterface.query_observation_sets`,
  :meth:`StationControlInterface.get_observation_set` and
  :meth:`StationControlInterface.get_observation_set_observations`
  which can be used for the same purpose. :class:`IQMServerClient` still has the original methods
  and it should be later refactored to implement the same interface as :class:`StationControlClient`.

Version 8.1.0 (2025-06-13)
==========================

Features
--------

- Improve docstring for `interface.models.JobExecutorStatus`.

Version 8.0.0 (2025-06-12)
==========================

Breaking changes
----------------

- Rename `JobStatus` to `JobExecutorStatus` in order to reduce confusion with CoCos models.
- Rename some `JobExecutorStatus` statuses to be more descriptive.

Version 7.1.0 (2025-06-10)
==========================

Features
--------

- Update dependency on exa-common


Version 7.0.0 (2025-06-02)
==========================

Features
--------

- Revert changes from :issue:`SW-1513`.

Version 6.0.0 (2025-05-30)
==========================

Features
--------

- Rename JobStatus entries to be more descriptive. :issue:`SW-1513`

Version 5.0.0 (2025-05-28)
==========================

Breaking changes
----------------

- :attr:`.JobData.job_error` now has separate fields for user error message and full error log, instead of only
  containing the full error log. :issue:`SW-1179`

Version 4.2.0 (2025-05-23)
==========================

Bug fixes
---------

- Use explicit UTF-8 encoding for JSON data in HTTP requests.

Version 4.1.0 (2025-05-21)
==========================

Features
--------

- Fix cocos path in ruff isort to run isort for cocos correctly.

Version 4.0.0 (2025-05-16)
==========================

Breaking changes
----------------

- Renamed two methods in station_control.client.serializers.task_serializers:
  - `serialize_run_task_request` to `serialize_run_job_request`
  - `serialize_sweep_task_request` to `serialize_sweep_job_request`
- Removed `SweepStatus`

Features
--------

- Reworked client methods to work with new station-control endpoints for job management.
- Added `JobStatus` model (previously in station-control).

Version 3.17.0 (2025-05-12)
===========================

Features
--------

- Update dependency on exa-common

Version 3.16.1 (2025-05-12)
===========================

- Test patch versioning, no functional changes. :issue:`SW-1429`

Version 3.16.0 (2025-04-22)
===========================

Features
--------

- Update dependency on exa-common

Version 3.15.0 (2025-04-11)
===========================

Bug fixes
---------

- Update license

Version 3.14.0 (2025-04-10)
===========================

Bug fixes
---------

- Fix broken iqm-server-client initialization

Version 3.13.0 (2025-04-07)
===========================

Features
--------

- Add partial IQM Server backend support to Station Control client to enable Pulla usage through IQM Server

Version 3.12.0 (2025-04-07)
===========================

Features
--------

- Fix package version in published docs footers, :issue:`SW-1392`. 

Version 3.11.0 (2025-04-03)
===========================

Feature
*******

- Enable PEP 604 in linting rules, :issue:`SW-1230`.

Version 3.10.0 (2025-04-03)
===========================

Features
--------

- Add versioning for station control API. :issue:`SW-898`

Version 3.9.0 (2025-04-02)
==========================

Features
********

- Update the documentation footer to display the package version.

Version 3.8.0 (2025-03-19)
==========================

Bug fixes
---------

- Update dependency on exa-common

Version 3.7.0 (2025-03-11)
==========================

Features
--------

- Bump pulla

Version 3.6.0 (2025-03-07)
==========================

Bug fix
-------

- Fix error message formatting of server-side errors.

Version 3.5.0 (2025-03-05)
==========================

- Bump station control dependencies

Version 3.4.0 (2025-03-05)
==========================

Features
--------

- Remove general RequestError and use new specific error classes instead, and improve error handling in general.
- Start using new "DELETE sweeps/{sweep_id}" endpoint instead of the deprecated one.
- Use HTTPStatus code names instead of numbers for better clarity. No functional changes.
- The ``/docs`` endpoint shows relevant metadata, e.g. the package version.

Version 3.3.0 (2025-02-28)
==========================

Features
--------

- Add ``StationControlClient.get_exa_configuration`` that returns the recommended EXA configuration from the server.
  :issue:`SW-1078`


Version 3.2.0 (2025-02-28)
==========================


Bug fix
-------
- Bump exa-common

Version 3.1.0 (2025-02-27)
==========================

Features
--------

* Bump EXA version.

Version 3.0.0 (2025-02-27)
==========================

Breaking changes
****************

- It is no longer possible to submit sweeps in the legacy readout format (settings without readout instructions
  in the playlist). :issue:`SW-690`
- Removed deprecated ``StationControlClient.get_chad`` and ``StationControlClient.get_qubit_design_properties``.
- Removed deprecated field ``SweepDefinition.playlists`` use ``.playlist`` instead.

Version 2.20.0 (2025-02-19)
===========================

- Bump station-control dependencies.

Version 2.19.0 (2025-02-18)
===========================

Bump exa-data, iqm-data-definitions versions.

Version 2.18.0 (2025-02-04)
===========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 2.17.0 (2025-02-04)
===========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 2.16.0 (2025-01-30)
===========================

Features
--------

- Implement callback to display progress bars for task execution :issue:`SW-881`

Version 2.15.0 (2025-01-28)
===========================

Features
********
- Support broader range of `numpy` versions and verify compatibily with ruff, see migration guide `https://numpy.org/doc/stable/numpy_2_0_migration_guide.html`.

Version 2.14.0 (2025-01-28)
===========================

Features
--------

- Bump exa-common.

Version 2.13.0 (2025-01-27)
===========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-1042`

Version 2.12.0 (2025-01-24)
===========================

Features
--------

* Add serialisation and deserialisation for channel properties

Version 2.11.0 (2025-01-10)
===========================

Features
--------

- Log meta.errors is station control responses automatically. :issue:`SW-514`.

Version 2.10.0 (2025-01-08)
===========================

Features
--------

- Remove gitlab links from public pages. :issue:`SW-776`

Version 2.9.0 (2024-12-30)
==========================

Features
--------

- Minor typos fixed. :issue:`SW-776`

Version 2.8.0 (2024-12-30)
==========================

Features
--------

- Change license info to Apache 2.0. :issue:`SW-776`

Version 2.7.0 (2024-12-17)
==========================

Features
--------

- Remove hardcoded iqm-internal links from docstrings :issue:`SW-977`

Version 2.6.0 (2024-12-12)
==========================

Features
--------

- Bump exa-experiments

Version 2.5.0 (2024-12-10)
==========================

Features
********

- Make `observation_ids` in `ObservationSetUpdate` optional. `SW-926`

Version 2.4.0 (2024-12-09)
==========================

Features
--------

Fix extlinks to MRs and issues in sphinx docs config :issue:`SW-916`

Version 2.3.0 (2024-12-05)
==========================

Features
--------

- Fix intersphinx reference paths in docs :issue:`SW-916`

Version 2.2.0 (2024-12-05)
==========================

Features
--------

- Add `characterization-set` to observation set pydantic model. `SW-845`

Version 2.1.0 (2024-12-04)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-665`

Version 2.0.0 (2024-11-22)
==========================

Breaking changes
****************
- This is only a bug fix MR, however it's technically a breaking change for station-control since the fix
  required us to change how list objects are serialized/deserialized and thus, station-control need to use
  different syntax for that from now on. exa-repo or other clients shouldn't be affected, so from their perspective
  this should be considered a minor release.

Bug fixes
*********

- Fix `NaN` and `Inf` serialization for float values, serializing them to `NaN` and `Inf` instead of `None`.
  This affected only when serializing list of objects at once (for example when saving multiple observations),
  endpoints dealing with single object were working as expected. :issue:`SW-865`

Version 1.23.0 (2024-11-19)
===========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-774`

Version 1.22.0 (2024-11-15)
===========================

Bug fixes
---------

- Remove iqm-internal web links in customer docs artifacts.

Version 1.21.0 (2024-11-12)
===========================

Features
********

- Allow extra attributes for ObservationDefinition 1.) to allow older server versions to accept newer versions
  with added attributes, and 2.) to make it possible to convert inheriting classes to ObservationDefinition without
  removing extra attributes. Part of :issue:`SW-774`.

Version 1.20.0 (2024-11-11)
===========================

Features
--------

- Adds "gbc-set" observation set type. :issue:`GBC-672`

Version 1.19.0 (2024-11-08)
===========================

Features
--------

- New changelog workflow, no functional changes. :issue:`SW-774`

Version 1.18 (2024-10-30)
=========================

- Bump Pydantic to version 2.9.2, :issue:`SW-804`.
- Bump `iqm-exa-common` to version 25.14.


Version 1.17 (2024-10-25)
=========================

- Update `iqm-exa-common` to version 25.13 and bump NumPy to version 1.25.2.


Version 1.16 (2024-10-24)
=========================

- Update `iqm-exa-common` to 25.12
- Use function :func:`convert_sweeps_to_list_of_tuples` from exa-common to sweep conversion


Version 1.15 (2024-10-23)
=========================

Features
--------
- Add optional `wait_task_completion` boolean parameter (default `True`) to station control client's `run()`.
  If set to `False`, `run()` won't wait/poll for the task completion, but instead returns immediately after it
  receives the initial response. This feature can be used to implement async-like workflows which is not blocked
  by the task execution. :issue:`EXA-1244`


Version 1.14 (2024-10-11)
=========================

- Update `iqm-exa-common` to version 25.11.


Version 1.13 (2024-10-02)
=========================

- Update `iqm-data-definitions` to version 2.0.


Version 1.12 (2024-09-26)
=========================

- Bugfix: JSON serialization error when saving array-valued observations that are non-contiguous memory. (in particular eg this happened from the IntegrationWeights experiment analysis).


Version 1.11 (2024-09-23)
=========================

Features
--------
- Update `iqm-exa-common` to version 25.9.


Version 1.10 (2024-09-20)
=========================

Features
--------
- Add optional client side fallback to fetch chip design records from QCM API. :issue:`SW-570`


Version 1.9 (2024-09-11)
========================

Features
--------
- Update exa-common.


Version 1.8 (2024-08-26)
========================

- Update ``exa-common`` to 25.7.


Version 1.7 (2024-08-23)
========================

Features
--------
- Support empty settings field in sweep serialization. :issue:`EXA-2099`


Version 1.6 (2024-08-16)
========================

Features
--------
- Update `iqm-exa-common`` to 25.6.


Version 1.5 (2024-08-15)
========================

Features
--------

- Add method `get_chip_design_record` to `StationControlClient`


Version 1.4 (2024-07-23)
========================

Features
--------
- Field ``feedback_signal_label`` added to ``ThresholdStateDiscrimination`` (an acquisition
  method in programmable readout).
  The label is used to specify a signal that a `ConditionalInstruction` can act on. :issue:`EXA-1923`



Version 1.3 (2024-07-12)
========================

Features
--------
- Bump exa-common to 25.4


Version 1.2 (2024-07-05)
========================

Features
--------
- Bump exa-common to 25.3 


Version 1.1 (2024-07-04)
========================

Features
--------

- Bump exa-common to 25.2. :issue:`EXA-2056`


Version 1.0 (2024-07-02)
========================

Features
--------

- Package `iqm-exa-backend-client` is renamed to `iqm-station-control-client`.
  No functional changes to `iqm-exa-backend-client` version 59.4.
