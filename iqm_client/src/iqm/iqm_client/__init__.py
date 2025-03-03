# Copyright 2021 IQM client developers
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
"""Client-side library for connecting to and executing quantum circuits on IQM quantum computers."""
from importlib.metadata import PackageNotFoundError, version
import sys
import warnings

from iqm.iqm_client.api import *
from iqm.iqm_client.authentication import *
from iqm.iqm_client.errors import *
from iqm.iqm_client.iqm_client import *
from iqm.iqm_client.models import *
from iqm.iqm_client.transpile import *

try:
    DIST_NAME = "iqm-client"
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

if sys.version_info < (3, 11):
    warnings.warn(DeprecationWarning("Python 3.10 will no longer be supported in a later release of IQM client."))
