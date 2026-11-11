#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Module for handling the database ID of persisted software version data."""

import sys


def detect_ipython_autoreload() -> bool:
    """Detect if IPython autoreload is in use.

    IPython autoreload is in use, if the following magic lines have been used:

    .. code::

        %load_ext autoreload
        %autoreload 2

    Returns:
        True if IPython autoreload is in use, or in case of any errors trying to access IPython internals. Otherwise
            False.

    """
    # This function uses a hacky way to determine if IPython autoreload is in use
    try:
        import IPython  # noqa: PLC0415 (import-outside-top-level)
    except ModuleNotFoundError:
        return False

    ipy = IPython.get_ipython()

    # Some time earlier, the module was named 'autoreload', but now it seems to be 'IPython.extensions.autoreload'.
    # FIXME: check only the relevant module from sys.modules, after we have implemented some kind of a test for this
    # method to ensure it doesn't break when IPython dependencies are upgraded.
    if ipy and ("IPython.extensions.autoreload" in sys.modules or "autoreload" in sys.modules):
        try:
            if ipy.magics_manager.magics["line"]["autoreload"].__self__._reloader.enabled:
                return True
        except Exception:
            # To be in the safe side, return True here if the check above (accessing ipython internals) fails
            return True
    return False
