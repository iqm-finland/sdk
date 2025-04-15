====================================
Changelog for old cortex-cli package
====================================

Version 5.10
============

* Use GitHub Action as a Trusted Publisher to publish packages to PyPI. `#68 <https://github.com/iqm-finland/cortex-cli/pull/68>`_

Version 5.9
===========

* Remove multiversion documentation. `#67 <https://github.com/iqm-finland/cortex-cli/pull/67>`_

Version 5.8
===========

* Bugfix: join URLs properly. `#66 <https://github.com/iqm-finland/cortex-cli/pull/66>`_

Version 5.7
===========

* Bump dependencies and dev tools. `#65 <https://github.com/iqm-finland/cortex-cli/pull/65>`_

Version 5.6
===========

* Enable ``cortex-cli`` to be used in non-interactive mode `#63 <https://github.com/iqm-finland/cortex-cli/pull/63>`_

Version 5.5
===========

* Update user documentation with an example how to set environment variable `#62 <https://github.com/iqm-finland/cortex-cli/pull/62>`_

Version 5.4
===========

* Use default values of ``realm`` and ``client-id`` silently during ``cortex init``. Allow to overwrite the default values with command line switches ``--realm`` and ``--client-id``. `#61 <https://github.com/iqm-finland/cortex-cli/pull/61>`_

Version 5.3
===========

* Use latest version of ``sphinx-multiversion-contrib`` to fix documentation version sorting. `#60 <https://github.com/iqm-finland/cortex-cli/pull/60>`_

Version 5.2
===========

* Fix ``cortex-cli`` logout crash when token manager process is not running. `#59 <https://github.com/iqm-finland/cortex-cli/pull/59>`_

Version 5.1
===========

* Minor language update in user instructions. `#58 <https://github.com/iqm-finland/cortex-cli/pull/58>`_

Version 5.0
===========

* Move ``cortex_cli`` package to ``iqm`` namespace. `#57 <https://github.com/iqm-finland/cortex-cli/pull/57>`_

Version 4.7
===========

* Remove temporary password update functionality from ``auth login`` command. `#56 <https://github.com/iqm-finland/cortex-cli/pull/56>`_

Version 4.6
===========

* Automatically use ``--no-daemon`` flag on Windows `#55 <https://github.com/iqm-finland/cortex-cli/pull/55>`_

Version 4.5
===========

* Support python 3.11 `#53 <https://github.com/iqm-finland/cortex-cli/pull/53>`_

Version 4.4
===========

* Support relative paths during init `#52 <https://github.com/iqm-finland/cortex-cli/pull/52>`_

Version 4.3
===========

* Fix logging for multi-user systems `#48 <https://github.com/iqm-finland/cortex-cli/pull/48>`_

Version 4.2
===========

* Fix ``init`` command non-interactive mode failing for some orders of options `#45 <https://github.com/iqm-finland/cortex-cli/pull/45>`_

Version 4.1
===========

* Generate license information for dependencies on every release `#44 <https://github.com/iqm-finland/cortex-cli/pull/44>`_

Version 4.0
===========

* Remove circuit execution command `#43 <https://github.com/iqm-finland/cortex-cli/pull/43>`_

Version 3.6
============

* Upgrade to IQMClient version 12.0 `#42 <https://github.com/iqm-finland/cortex-cli/pull/42>`_

Version 3.5
===========

* Allow user to update a temporary password in ``auth login`` command. `#41 <https://github.com/iqm-finland/cortex-cli/pull/41>`_

Version 3.4
===========

* "Pin down" supported Python versions to 3.9 and 3.10. `#38 <https://github.com/iqm-finland/cortex-cli/pull/38>`_
* Configure Tox to skip missing versions of Python interpreters when running tests. `#38 <https://github.com/iqm-finland/cortex-cli/pull/38>`_
* Move project metadata and configuration to ``pyproject.toml``. `#38 <https://github.com/iqm-finland/cortex-cli/pull/38>`_

Version 3.3
===========

* Provide version information to IQMClient. `#39 <https://github.com/iqm-finland/cortex-cli/pull/39>`_

Version 3.2
===========

* Build docs for older versions. `#37 <https://github.com/iqm-finland/cortex-cli/pull/37>`_

Version 3.1
===========

* Use ``calibration_set_id`` from metadata directly and not from the request. `#36 <https://github.com/iqm-finland/cortex-cli/pull/36>`_

Version 3.0
===========

* Use new opaque UUID for ``calibration_set_id``. `#35 <https://github.com/iqm-finland/cortex-cli/pull/35>`_

Version 2.4
===========

* Remove ``description`` from pydantic model fields. `#34 <https://github.com/iqm-finland/cortex-cli/pull/34>`_

Version 2.3
===========

* Cortex init command will validate the entered ``auth_server_url`` and ``realm``. `#33 <https://github.com/iqm-finland/cortex-cli/pull/33>`_

Version 2.2
===========

* Tokens file gets deleted and logout succeeds even if the authentication server is not available `#32 <https://github.com/iqm-finland/cortex-cli/pull/32>`_

Version 2.1
===========

* The user can now specify the output format for ``cortex circuit run``: human-readable ``--output frequencies``, ``--output shots`` or machine-readable raw ``RunResult`` ``--output json``. `#31 <https://github.com/iqm-finland/cortex-cli/pull/31>`_

Version 2.0
===========

* Replace qubit mapping with QASM qubit placement. `#30 <https://github.com/iqm-finland/cortex-cli/pull/30>`_

Version 1.6
===========

* Gracefully handle token files with outdated format. `#28 <https://github.com/iqm-finland/cortex-cli/pull/28>`_

Version 1.5
===========

* Make circuit execution an optional functionality, which requires to install additional dependencies. `#27 <https://github.com/iqm-finland/cortex-cli/pull/27>`_

Version 1.4
===========

* Fix typing issue related to upgrading to cirq-on-iqm 7.7. `#25 <https://github.com/iqm-finland/cortex-cli/pull/25>`_

Version 1.3
===========

* Remove settings from circuit run command. `#24 <https://github.com/iqm-finland/cortex-cli/pull/24>`_
* Upgrade to iqm-client 8.0. `#24 <https://github.com/iqm-finland/cortex-cli/pull/24>`_

Version 1.2
===========

* Fix a bug in handling tokens received from auth server. `#23 <https://github.com/iqm-finland/cortex-cli/pull/23>`_

Version 1.1
===========

* Token manager will keep trying indefinitely to re-connect to auth server. `#22 <https://github.com/iqm-finland/cortex-cli/pull/22>`_

Version 1.0
===========

* Enable foreground mode for token manager. `#20 <https://github.com/iqm-finland/cortex-cli/pull/20>`_
* Flag ``--no-daemon`` of the ``cortex auth login`` command is renamed to ``--no-refresh``. `#20 <https://github.com/iqm-finland/cortex-cli/pull/20>`_
* Flag ``--no-daemon`` now starts the token manager in foreground mode. `#20 <https://github.com/iqm-finland/cortex-cli/pull/20>`_
* Breaking change. The format of the configuration file is changed: ``base_url`` renamed to ``auth_server_url``. `#20 <https://github.com/iqm-finland/cortex-cli/pull/20>`_
* Breaking change. The format of the tokens file is changed: ``timestamp`` format is changed to ISO. `#20 <https://github.com/iqm-finland/cortex-cli/pull/20>`_
* Configuration and tokens files' formats are now validated by Cortex CLI. `#20 <https://github.com/iqm-finland/cortex-cli/pull/20>`_

Version 0.11
============

* ``--no-auth`` and ``--config-file`` are now mutually exclusive `#19 <https://github.com/iqm-finland/cortex-cli/pull/19>`_

Version 0.10
============

* Upgrade to iqm-client 7.0 `#18 <https://github.com/iqm-finland/cortex-cli/pull/18>`_
* Report the ID of the calibration set that was used in circuit run when no settings were specified. `#18 <https://github.com/iqm-finland/cortex-cli/pull/18>`_

Version 0.9
===========

* Enable mypy checks. `#17 <https://github.com/iqm-finland/cortex-cli/pull/17>`_
* Update source code according to new checks in pylint v2.15.0. `#17 <https://github.com/iqm-finland/cortex-cli/pull/17>`_

Version 0.8
===========

* Upgrade ``cirq-iqm`` to 7.3. `#15 <https://github.com/iqm-finland/cortex-cli/pull/15>`_

Version 0.7
===========

* iqm-client 6.1 support. `#13 <https://github.com/iqm-finland/cortex-cli/pull/13>`_
* Allow user to provide a custom ``calibration_set_id`` when using ``cortex circuit run``. `#13 <https://github.com/iqm-finland/cortex-cli/pull/13>`_
* Update documentation regarding the use of Cortex CLI. `#13 <https://github.com/iqm-finland/cortex-cli/pull/13>`_

Version 0.6
===========

* iqm-client 6.0 support. `#14 <https://github.com/iqm-finland/cortex-cli/pull/14>`_

Version 0.5
===========

* Partial Windows support (no token manager daemon)
* Performance improvements for faster loading time

Version 0.4
===========

* Bump iqm-client dependency to 5.0
* Remind the user to login before using operations requiring authentication

Version 0.3
===========

* Fix tests for iqm-client 4.3

Version 0.2
===========

* Added circuit commands ``cortex circuit validate`` and ``cortex circuit run``

Version 0.1
===========

* Authentication token manager daemon
* Multiple configurations support
