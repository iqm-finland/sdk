# Copyright 2025 IQM client developers
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
"""Data models used by IQMServerClient."""

from __future__ import annotations

from uuid import UUID

from iqm.station_control.interface.pydantic_base import PydanticBase


class QuantumComputer(PydanticBase):
    """Quantum computer attributes."""

    id: UUID
    """Unique ID of the quantum computer."""
    alias: str
    """Quantum computer alias that can be used as a substitute for id in API calls."""
    display_name: str
    """Quantum computer name that can be displayed in the UI."""
    proxy_path: str | None = None
    """URL path prefix for station-proxied requests to this quantum computer.

    ``None`` if station proxy access is disabled or not permitted for this quantum computer. Absent
    entirely (defaults to ``None``) when the server predates this field.
    """


class ListQuantumComputersResponse(PydanticBase):
    """Response of GET /v1/quantum-computers."""

    quantum_computers: list[QuantumComputer]
    """List of available quantum computers."""
