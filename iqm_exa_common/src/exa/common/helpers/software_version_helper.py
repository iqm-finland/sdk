# Copyright 2024 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import reload
from importlib.metadata import distribution
import os
import platform
import subprocess

import pkg_resources  # type:ignore[import-untyped]


def _is_editable(pkg_name: str) -> bool:
    """Determine if the package is installed in the editable/development mode.

    Currently happens by checking if the first file in the installed distribution has the name
    __editable__.<pkgname>-<version>.pth.
    TODO this is a hacky way of figuring out editability, not an official feature of ``setuptools`` or
    ``importlib.metadata``, so it might break anytime.
    """
    dist = distribution(pkg_name)
    return dist.files is not None and dist.files[0].name.startswith("__editable__.")


def get_all_software_versions(reload_module: bool = False) -> dict[str, str]:
    """Get all available software version information.

    Currently, this function collects all Python package versions and Python interpreter version.

    Args:
        reload_module: Whether to reload the ``pkg_resources`` module or not. By default,
            it is disabled because reloading the module is not thread safe!
            This function should be called with ``reload_module=True`` when IPython autoreload is in use.

    Example:
            1. You have numpy==1.21.0 installed, and in the notebook you have executed the following IPython magic:

                .. code-block:: python

                    %load_ext autoreload
                    %autoreload 2

            2. You install numpy==1.21.1

            3. You call this function with ``reload_module=False``. This will result in some warning printouts and
               the function will succeed by returning 1.21.0 for numpy, which is wrong because in reality IPython
               autoreload has reloaded the newly installed numpy 1.21.1.
               With ``reload_module=True`` the correct version 1.21.1 is returned and no warnings are printed.


    Returns: All software components in a dictionary that maps each package name to its version
        information. A package's version information contains the base version, and the string
        "(local editable)" in the case the package is a local editable installation.

    """
    python_version = platform.python_version()
    software_versions = {"python": python_version}
    # TODO use of pkg_resources is discouraged, replace it with importlib.metadata below.
    # https://setuptools.pypa.io/en/latest/pkg_resources.html
    if reload_module:
        reload(pkg_resources)
    for pkg in pkg_resources.working_set:
        value = pkg.parsed_version.base_version
        if _is_editable(pkg.project_name):
            value += " (local editable)"
        software_versions[pkg.project_name] = value
    return software_versions


def get_vcs_description(root_directory: str) -> str | None:
    """Get Version Control System (VCS) description for the caller's current working directory.

    The description is used to verify if a directory is installed under VCS and whether changes to the files have
    been made. First, the existence of ``.git`` directory will be checked from :attr:`root_directory`.
    Only if it exists, a ``git`` command is executed in a subprocess with a timeout
    of 1 seconds as the best effort only.

    Attributes:
        root_directory: The path to the directory where the command will be executed. For instance when called from
            exa-experiment, it can be the exa-experiment root directory or any directory under it.

    Returns:
        If :attr:`root_directory` is not installed under git, None will be returned. Otherwise, the output of
        ``git describe --dirty --tags --long`` is returned. In case of errors in executing the command, the caught
        ``subprocess.CalledProcessError`` will be converted to string and returned.

    Raises:
        If the command fails or timeouts, an exception will be raised directly from ``subprocess.check_output``.

    """
    # TODO does not seem very robust, consider using the full version number (e.g. 17.0.post1.dev9+g4748363.d20221003)
    # provided by setuptools_scm and accessed using importlib.metadata.version instead, if
    # it can be made to work with importlib.reload.
    if not os.path.isdir(os.path.join(root_directory, ".git")):
        return None

    # git command is similar to what setuptools_scm uses for describing versions:
    # SETUPTOOLS_SCM_DEBUG=1 python3 setup.py --version
    return subprocess.check_output(
        ["git", "describe", "--dirty", "--tags", "--long"], timeout=5, text=True, cwd=root_directory
    ).strip()
