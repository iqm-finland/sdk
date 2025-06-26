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
"""Dynamic quantum architecture (DQA) related interface models."""

from typing import Any
from uuid import UUID

from pydantic import Field, StrictStr, field_validator

from iqm.station_control.interface.pydantic_base import PydanticBase

Locus = tuple[StrictStr, ...]
"""Names of the QPU components (typically qubits) a quantum operation instance is acting on, e.g. `("QB1", "QB2")`."""


class GateImplementationInfo(PydanticBase):
    """Information about an implementation of a quantum gate/operation."""

    loci: tuple[Locus, ...] = Field(
        examples=[(("COMP_R", "QB1"), ("COMP_R", "QB2"))],
    )
    """Loci for which this gate implementation has been calibrated."""


class GateInfo(PydanticBase):
    """Information about a quantum gate/operation."""

    implementations: dict[str, GateImplementationInfo] = Field(
        examples=[
            {
                "tgss": GateImplementationInfo(loci=(("COMP_R", "QB1"), ("COMP_R", "QB2"))),
                "crf": GateImplementationInfo(loci=(("COMP_R", "QB1"), ("COMP_R", "QB2"))),
            }
        ],
    )
    """Mapping of available implementation names to information about the implementations."""

    default_implementation: str = Field(
        examples=["tgss"],
    )
    """Default implementation for the gate.
    
    Used unless overridden by :attr:`override_default_implementation`,
    or unless the user requests a specific implementation for a particular gate in the circuit using
    :attr:`iqm.cocos.app.api.request_models.Instruction.implementation`."""

    override_default_implementation: dict[Locus, str] = Field(
        examples=[{("COMP_R", "QB2"): "crf"}],
    )
    """Mapping of loci to implementation names that override ``default_implementation`` for those loci."""

    @field_validator("override_default_implementation", mode="before")
    @classmethod
    def override_default_implementation_validator(cls, value: Any) -> dict[Locus, str]:
        """Converts locus keys to tuples if they are encoded as strings."""
        new_value = {}
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, tuple):
                    new_value[k] = v
                # When Pydantic serializes a dict with tuple keys into JSON, the keys are turned into
                # comma-separated strings (because JSON only supports string keys). Here we convert
                # them back into tuples.
                elif isinstance(k, str):
                    new_k = tuple(k.split(","))
                    new_value[new_k] = v
                else:
                    raise ValueError("'override_default_implementation' keys must be strings or tuples.")
            return new_value
        raise ValueError("'override_default_implementation' must be a dict.")


class DynamicQuantumArchitecture(PydanticBase):
    """The dynamic quantum architecture (DQA).

    Describes gates/operations for which calibration data exists in the calibration set.
    """

    calibration_set_id: UUID = Field(
        examples=["cd4dd889-b88b-4370-ba01-eb8262ad9c53"],
    )
    """ID of the calibration set from which this DQA was generated."""

    qubits: list[str] = Field(
        examples=[["QB1", "QB2"]],
    )
    """Qubits that appear in at least one gate locus in the calibration set."""

    computational_resonators: list[str] = Field(
        examples=[["COMP_R"]],
    )
    """Computational resonators that appear in at least one gate locus in the calibration set."""

    gates: dict[str, GateInfo] = Field(
        examples=[
            {
                "cz": GateInfo(
                    implementations={
                        "tgss": GateImplementationInfo(loci=(("COMP_R", "QB1"), ("COMP_R", "QB2"))),
                        "crf": GateImplementationInfo(loci=(("COMP_R", "QB1"), ("COMP_R", "QB2"))),
                    },
                    default_implementation="tgss",
                    override_default_implementation={("COMP_R", "QB2"): "crf"},
                )
            }
        ],
    )
    """Mapping of gate names to information about the gates."""
