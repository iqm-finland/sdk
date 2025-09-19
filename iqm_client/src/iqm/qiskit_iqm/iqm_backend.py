# Copyright 2022-2025 Qiskit on IQM developers
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
"""Qiskit backend for IQM quantum computers."""

from __future__ import annotations

from abc import ABC
import logging
from typing import Final

from iqm.iqm_client import (
    DynamicQuantumArchitecture,
    ObservationFinder,
)
from iqm.qiskit_iqm.iqm_target import IQMTarget
from qiskit.providers import BackendV2
from qiskit.transpiler import Target

IQM_TO_QISKIT_GATE_NAME: Final[dict[str, str]] = {"prx": "r", "cz": "cz"}
logger = logging.getLogger(__name__)


class IQMBackendBase(BackendV2, ABC):
    """Abstract base class for various IQM-specific backends.

    Args:
        architecture: Dynamic quantum architecture associated with the backend instance.
        metrics: Optional calibration data and related quality metrics for the transpilation target.
        name: Optional name for the backend instance.

    """

    def __init__(
        self,
        architecture: DynamicQuantumArchitecture,
        *,
        metrics: ObservationFinder | None = None,
        name: str = "IQMBackend",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.architecture = architecture
        self.metrics = metrics
        # Qiskit uses integer indices to refer to qubits, so we need to map component names to indices.
        # Because of the way the Target and the transpiler interact, the resonators need to have higher indices than
        # qubits, or else transpiling with optimization_level=0 will fail because of lacking resonator indices.
        qb_to_idx = {qb: idx for idx, qb in enumerate(architecture.qubits + architecture.computational_resonators)}

        # NOTE: both targets include fictional CZs
        self._target = IQMTarget(
            architecture=architecture,
            component_to_idx=qb_to_idx,
            include_resonators=False,
            include_fictional_czs=True,
            metrics=metrics,
        )

        self._target_with_resonators = (
            IQMTarget(
                architecture=architecture,
                component_to_idx=qb_to_idx,
                include_resonators=True,
                include_fictional_czs=True,
                metrics=metrics,
            )
            if "move" in architecture.gates
            else None
        )

        self._qb_to_idx = qb_to_idx
        self._idx_to_qb = {v: k for k, v in qb_to_idx.items()}
        self._coupling_map = self.target.build_coupling_map()

    @property
    def target(self) -> Target:
        """Return the target without computational resonators."""
        return self._target

    @property
    def target_with_resonators(self) -> Target:
        """Return the target with MOVE gates and resonators included.

        Raises:
            ValueError: The backend does not have resonators.

        """
        target = self._target_with_resonators
        if target is None:
            raise ValueError("The backend does not have computational resonators.")
        return target

    @property
    def physical_qubits(self) -> list[str]:
        """Return the list of physical qubits in the backend."""
        return list(self._qb_to_idx)

    def has_resonators(self) -> bool:
        """True iff the backend QPU has computational resonators."""
        return bool(self.architecture.computational_resonators)

    def get_real_target(self) -> Target:
        """Return the real physical target of the backend without fictional CZ gates."""
        return IQMTarget(
            architecture=self.architecture,
            component_to_idx=self._qb_to_idx,
            include_resonators=True,
            include_fictional_czs=False,
            metrics=self.metrics,
        )

    def qubit_name_to_index(self, name: str) -> int:
        """Given an IQM-style qubit name, return the corresponding index in the register.

        Args:
            name: IQM-style qubit name ('QB1', 'QB2', etc.)

        Returns:
            Index of the given qubit in the quantum register.

        Raises:
            ValueError: Qubit name cannot be found on the backend.

        """
        if name not in self._qb_to_idx:
            raise ValueError(f"Qubit '{name}' is not found on the backend.")
        return self._qb_to_idx[name]

    def index_to_qubit_name(self, index: int) -> str:
        """Given a quantum register index, return the corresponding IQM-style qubit name.

        Args:
            index: Qubit index in the quantum register.

        Returns:
            Corresponding IQM-style qubit name ('QB1', 'QB2', etc.).

        Raises:
            ValueError: Qubit index cannot be found on the backend.

        """
        if index not in self._idx_to_qb:
            raise ValueError(f"Qubit index {index} is not found on the backend.")
        return self._idx_to_qb[index]

    def get_scheduling_stage_plugin(self) -> str:
        """Return the plugin that should be used for scheduling the circuits on this backend."""
        return "iqm_default_scheduling"
