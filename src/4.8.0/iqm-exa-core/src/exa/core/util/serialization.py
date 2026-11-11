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
"""Serialization utils."""

import importlib
from typing import Any


def fully_qual_name(obj: Any) -> str:
    """Returns the fully qualified name (module + qualified name) of any object (object, type, or func).

    Args:
        obj: The object whose fully qualified name is to be returned (should have __module__ and __qualname__).

    Returns:
        Fully qualified name of ``obj``.

    """
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    else:
        return f"{obj.__module__}.{type(obj).__qualname__}"


def fqn_to_object(fqn: str) -> Any:
    """Returns an object (object, type, or func) from its fully qualified name.

    Args:
        fqn: Fully qualified name of the object that should be returned.

    Returns:
        Object behind the fully qualified name.

    """
    mod, name = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)
