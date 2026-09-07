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

from importlib.metadata import distributions
import platform

from packaging.version import Version


def get_all_software_versions() -> dict[str, str]:
    """Get all available software version information.

    Currently, this function collects all Python package versions and Python interpreter version.

    Returns: All software components in a dictionary that maps each package name to its version
        information. A package's version information contains the base version, and the string
        "(local editable)" in the case the package is a local editable installation.

    """
    python_version = platform.python_version()
    software_versions = {"python": python_version}
    for dist in distributions():
        pkg_name = dist.metadata["Name"]
        version = dist.version
        try:
            base_version = Version(version).base_version
        except Exception:
            base_version = version
        value = base_version
        software_versions[pkg_name] = value
    return software_versions
