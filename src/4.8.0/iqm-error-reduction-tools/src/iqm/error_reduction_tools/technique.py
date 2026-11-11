# Copyright 2022-2026 IQM
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

"""Interfaces an error reduction technique implements to plug into the workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from .circuits import CircuitVariant, TargetCircuit
    from .observables import ObservableSet


class Technique(Protocol):
    """Builds the circuit variants of a target. Its configuration fixes which circuits are executed."""

    def build_variants(self, target: TargetCircuit, observables: ObservableSet) -> tuple[CircuitVariant, ...]:
        """Build the variants of one target.

        Variant 0 must be the target circuit itself, and variant ``i`` must have ``variant_index == i``. Leave
        ``shots`` and ``num_instances`` unset to use the workflow's, or set them to override it for that variant.

        Args:
            target: The target, without variants.
            observables: The observables to be estimated.

        Returns:
            The variants, not yet twirled.

        """
        ...


@dataclass(frozen=True)
class TargetEstimate:
    """The estimate of every observable of one target."""

    values: np.ndarray
    """Estimated expectation value of each observable, shape ``(num_observables,)``."""

    diagnostics: object
    """Technique-specific data on how the estimate was obtained, e.g. the fits."""


class Estimation(Protocol):
    """Turns the measured values of a target's variants into an estimate. Can be changed after execution."""

    def accepts(self, technique: Technique) -> bool:
        """Return whether this estimation can process the variants built by ``technique``."""
        ...

    def estimate(self, target: TargetCircuit, term_values: np.ndarray, observables: ObservableSet) -> TargetEstimate:
        """Estimate the observables of one target.

        Args:
            target: The planned target, with its variants.
            term_values: Measured expectation value of each Pauli term of each variant, shape
                ``(num_variants, num_terms)``, with terms ordered as in ``observables.pauli_terms``.
            observables: The observables.

        Returns:
            The estimate.

        """
        ...
