# Copyright 2025 IQM
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
"""Static quantum architecture (SQA) related interface models."""

from pydantic import Field

from iqm.station_control.interface.pydantic_base import PydanticBase


class StaticQuantumArchitecture(PydanticBase):
    """The static quantum architecture (SQA) provides information about the QPU.

    For example, the names of its components and the connections between them.
    """

    qubits: list[str] = Field(
        examples=[["QB1", "QB2"]],
    )
    """Names of the qubits on the QPU, sorted."""

    computational_resonators: list[str] = Field(
        examples=[["CR1"]],
    )
    """Names of the computational resonators on the QPU, sorted."""

    connectivity: list[tuple[str, ...]] = Field(
        examples=[[("QB1", "QB2"), ("QB1", "CR1")]],
    )
    """Components (qubits and computational resonators) connected by a coupler on the QPU, sorted."""
