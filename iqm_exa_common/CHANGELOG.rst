=========
Changelog
=========

Version 26.9 (2025-04-03)
=========================

Feature
*******

- Format code and enable PEP 604 in linting rules, :issue:`SW-1230`.

Version 26.8 (2025-04-02)
=========================

Features
********

- Update the documentation footer to display the package version.

Version 26.7 (2025-04-01)
=========================

Features
--------

- Use standard process of deprecation. No functional changes. :issue:`SW-450`

Version 26.6 (2025-03-21)
=========================

Features
********

* Rename QPU chip types, based on either "crystal" or "star" architecture and number of qubits. For example,
  "crystal_5" or "star_6". For "mini" chips, like "mini_crystal_20", the number is not based on the actual number
  of qubits but to the chip it's trying to "minimize" instead, like "crystal_20". :issue:`SW-1059`

Version 26.5 (2025-03-19)
=========================

Bug fixes
---------

- Fix Parameters with element_indices having those indices duplicated in the name when deserialised

Version 26.4 (2025-03-11)
=========================

Features
--------

- Bump pulla

Version 26.3 (2025-03-05)
=========================

Features
--------

- Add new error classes designed for client-server communication via station control client.
- Remove general RequestError and use new specific error classes instead.

Version 26.2 (2025-03-03)
=========================

Bug fix
-------
- Fix numpy numeric types serialization in sweeps

Version 26.1 (2025-02-28)
=========================


Bug fix
-------
- Fix protobuf deserialisation to not align SettingNode names

Version 26.0 (2025-02-27)
=========================


Features
--------

Breaking changes
****************

- Remove Pydantic based ``ParameterModel``, ``SettingModel``, ``SettingNodeModel``, and ``SweepModel``,
  and inherit ``Parameter``, ``Setting``, ``SettingNode``, and ``Sweep`` from Pydantic model directly.
  As a result, the signature of usage has changed. :issue:`SW-798`
    - Instead of ``copy``, ``model_copy`` (with an optional ``update`` parameter) should be used.
    - For deserialization, for now, use :meth:`SettingNode.deserialize` (Pydantic native approach will be implemented later)
- ``FunctionSweep`` and ``FunctionOptions`` has been deleted.
- :class:`.SettingNode` now supports "the path notation" when inserting/getting nodes

  - Example: ``settings["flux.my.new.path.foo"] = SettingNode(...)`` adds the specified node under the specified path.
    Any missing subnodes will be added.

- :class:`.SettingNode` and :class:`.Setting` now have an attribute ``path`` which will be populated by their relative
  path within the settings tree when inserting/creating the node. The base class will also automatically align the name
  of a node with its path, but there is an attribute ``align_name`` which can be set to ``False`` to not align
  (used e.g. in the controllers section of the EXA settings tree).
- Remove deprecated ``QCMClient.get_chad`` and ``QCMClient.get_qubit_design_properties``.

Features
--------

- Settings can now be declared read-only by setting ``read_only = True`` when
  initialising the setting.
- Modified the html representation of settings tree to support read_only settings.
- Methods :meth:`.SettingNode.get_gate_node_for_locus`, :meth:`.SettingNode.get_gate_properties_for_locus`,
  :meth:`.SettingNode.get_default_implementation`, and :meth:`.SettingNode.get_locus_node_paths_for` added
  for accessing EXA-specific gate/characterization nodes in the settings tree.
- Methods :meth:`.SettingNode.add_for_path` and :meth:`.get_node_for_path` that facilitate dealing with long paths
  in setting trees.
  in setting trees.
- ``Sweep.options`` is deprecated, use ``Sweep.data`` instead. ``data`` can be still generated using different
  sweep options.
- ``ExponentialSweep``, ``FixedSweep``, and ``LinearSweep`` are deprecated, use ``Sweep`` instead.
- Split ``DataType.NUMBER`` to ``DataType.FLOAT`` and ``DataType.INT``. ``DataType.NUMBER`` is now handled as a
  deprecated alias for ``DataType.FLOAT``.

Version 25.34 (2025-02-06)
==========================

Bug fixes
---------

- Bump mechanize test dependency.

Version 25.33 (2025-02-04)
==========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 25.32 (2025-02-04)
==========================

Features
--------

- Refactor codebase to new lint rules. No functional changes. :issue:`SW-467`


Version 25.31 (2025-01-28)
==========================

Features
********
- Support broader range of `numpy` versions and verify compatibily with ruff, see migration guide `https://numpy.org/doc/stable/numpy_2_0_migration_guide.html`.

Version 25.30 (2025-01-28)
==========================

Bug Fixes
---------

- Method ``ChipTopology.get_all_common_resonators`` can never return a set containing components which are not
  computational resonators.

Version 25.29 (2025-01-27)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-1042`

Version 25.28 (2025-01-08)
==========================

Features
--------

- Remove gitlab links from public pages. :issue:`SW-776`

Version 25.27 (2024-12-19)
==========================

Features
********
- Bumps xarray

Version 25.26 (2024-12-12)
==========================

Features
--------

- Bump exa-experiments

Version 25.25 (2024-12-11)
==========================

Features
--------

- Fix public PyPI publishing. :issue:`SW-776`

Version 25.24 (2024-12-11)
==========================

Features
--------

- Change license info to Apache 2.0. :issue:`SW-776`

Version 25.23 (2024-12-09)
==========================

Features
--------

Fix extlinks to MRs and issues in sphinx docs config :issue:`SW-916`

Version 25.22 (2024-12-05)
==========================

Features
--------

- Fix intersphinx reference paths in docs :issue:`SW-916`

Version 25.21 (2024-12-04)
==========================

Test
****
- Adds unit test for sorting couplers

Version 25.20 (2024-12-04)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-665`

Version 25.19 (2024-11-29)
==========================

Features
--------

- Include computational resonators as possible locus components for `DEFAULT_2QB_MAPPING`, which is used for slow CZ
  gates, required for :issue:`GBC-589`.

Version 25.18 (2024-11-27)
==========================

Features
--------

- Expand allowed CHEDDAR versions in `qcm_data_client` to include versions 2.x.

Version 25.17 (2024-11-19)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-774`

Version 25.16 (2024-11-15)
==========================

Bug fixes
---------

- Remove iqm-internal web links in customer docs artifacts.

Version 25.15 (2024-11-08)
==========================

Features
--------

- New changelog workflow, no functional changes. :issue:`SW-774`

Version 25.14 (2024-10-30)
==========================

- Bump Pydantic to version 2.9.2, :issue:`SW-804`.


Version 25.13 (2024-10-28)
==========================

- Bump NumPy to version 1.25.2.


Version 25.12 (2024-10-24)
==========================

- Add sweep validation to :func:`convert_sweeps_to_list_of_tuples` function.


Version 25.11 (2024-10-11)
==========================

- Add :func:`get_all_common_resonators`


Version 25.10 (2024-10-02)
==========================

- Bump `iqm-data-definitions` to 2.0.


Version 25.9 (2024-09-23)
=========================

- Bump dependency `requests` to version 2.32.3


Version 25.8 (2024-09-10)
=========================

Features
--------
- Add ``EmptyComponentListError``.



Version 25.7 (2024-08-23)
=========================

Bug fix
-------
- Fix :meth:`Setting.__eq__` not working between values of type ``np.ndarray`` and ``None``.


Version 25.6 (2024-08-16)
=========================

Bug fix
-------

- Fix `QCMDataClient.get_chip_design_record` not working on remote targets.


Version 25.5 (2024-08-15)
=========================

Features
--------

- Add `QCMDataClient.get_chip_design_record`. Can be used in place of `get_chad`. :issue:`EXA-2077`
- Deprecate `QCMDataClient.get_qubit_design_properties` as the chip design_record (CHEDDAR) contains the same data.
- Add more utility methods to `ChipTopology`.


Version 25.4 (2024-07-12)
=========================

Features
--------

- Add an optional fallback URL to `QCMDataClient`.


Version 25.3 (2024-07-05)
=========================

Features
--------

- Add `_repr_html_` method for :class:`SettingNode`. This method overrides the
 default `repr` in notebooks. :issue:`EXA-1975`



Version 25.2 (2024-07-04)
=========================

Features
--------

- Add couplers to data components of :class:`ChipTopology`. :issue:`EXA-2056`


Version 25.1 (2024-06-27)
=========================

Features
--------

- First changelog for exa-common. No functional changes.
