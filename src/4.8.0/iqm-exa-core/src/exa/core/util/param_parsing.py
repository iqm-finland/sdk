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
"""Helper functions for parsing query parameters."""

from typing import Any


def chunk_list(values: list[Any], chunk_size: int) -> list[list[Any]]:
    """Takes a list of objects and splits it into several lists of size chunk_size."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be larger than zero.")
    chunked_list = [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]
    return chunked_list
