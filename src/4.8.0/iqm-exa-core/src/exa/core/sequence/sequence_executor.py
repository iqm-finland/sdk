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
"""Class for customizing Sequence execution."""

from .result import SequenceExecuteResult


class SequenceExecutor:
    """Sequence executors can be used to modify the sequence execution logic.

    This is the default sequence executor, which simply executes the given
    sequence. More complicated logic can be implemented by overriding the
    execute method.
    """

    def __init__(self, sequence):  # noqa: ANN001
        """Initializes the executor."""
        self.sequence = sequence

    def execute(self) -> SequenceExecuteResult:
        """Executes the given sequence.

        This method can be overridden in order to implement a more
        complicated execution logic.
        """
        return self.sequence.execute()
