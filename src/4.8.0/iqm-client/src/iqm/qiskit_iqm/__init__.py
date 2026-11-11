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
"""Qiskit adapter for IQM's quantum computers."""

__all__ = [
    "MOVE_GATE_UNITARY",
    "TOLERANCE",
    "ExistingMoveHandlingOptions",
    "IQMBackend",
    "IQMCircuit",
    "IQMDefaultSchedulingPlugin",
    "IQMErrorProfile",
    "IQMFacadeBackend",
    "IQMFakeAdonis",
    "IQMFakeAphrodite",
    "IQMFakeApollo",
    "IQMFakeBackend",
    "IQMFakeDeneb",
    "IQMFakeGarnet",
    "IQMJob",
    "IQMNaiveResonatorMoving",
    "IQMOptimizeSingleQubitGates",
    "IQMProvider",
    "IQMReplaceGateWithUnitaryPass",
    "IQMSchedulingPlugin",
    "IQMTarget",
    "MeasurementKey",
    "MoveGate",
    "MoveGateRoutingIgnoreBarriersKeepExistingMovesPlugin",
    "MoveGateRoutingIgnoreBarriersRemoveExistingMovesPlugin",
    "MoveGateRoutingIgnoreBarriersTrustExistingMovesPlugin",
    "MoveGateRoutingKeepExistingMovesPlugin",
    "MoveGateRoutingOnlyKeepExistingMovesPlugin",
    "MoveGateRoutingOnlyPlugin",
    "MoveGateRoutingOnlyRemoveExistingMovesPlugin",
    "MoveGateRoutingOnlyTrustExistingMovesPlugin",
    "MoveGateRoutingPlugin",
    "MoveGateRoutingRemoveExistingMovesPlugin",
    "MoveGateRoutingTrustExistingMovesPlugin",
    "MoveGateRoutingWithDroppedRZIgnoreBarriersKeepExistingMovesPlugin",
    "MoveGateRoutingWithDroppedRZIgnoreBarriersPlugin",
    "MoveGateRoutingWithDroppedRZIgnoreBarriersRemoveExistingMovesPlugin",
    "MoveGateRoutingWithDroppedRZIgnoreBarriersTrustExistingMovesPlugin",
    "MoveGateRoutingWithDroppedRZKeepExistingMovesPlugin",
    "MoveGateRoutingWithDroppedRZPlugin",
    "MoveGateRoutingWithDroppedRZRemoveExistingMovesPlugin",
    "MoveGateRoutingWithDroppedRZTrustExistingMovesPlugin",
    "MoveGateRoutingWithExactRZPlugin",
    "MoveGateRoutingWithRZOptimizationIgnoreBarriersPlugin",
    "OnlyRZOptimizationDropFinalRZIgnoreBarriersPlugin",
    "OnlyRZOptimizationDropFinalRZPlugin",
    "OnlyRZOptimizationExactIgnoreBarriersPlugin",
    "OnlyRZOptimizationExactPlugin",
    "OnlyRZOptimizationIgnoreBarriersPlugin",
    "OnlyRZOptimizationPlugin",
    "PassManager",
    "PassManagerConfig",
    "PassManagerStagePlugin",
    "bell_measure",
    "deserialize_instructions",
    "generate_initial_layout",
    "optimize_single_qubit_gates",
    "serialize_instructions",
    "transpile_example",
    "transpile_to_IQM",
    "validate_circuit",
]

import warnings

from iqm.iqm_client import ExistingMoveHandlingOptions
from packaging.version import Version
from qiskit import __version__ as qiskit_version
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.builtin_plugins import PassManagerStagePlugin

from .examples import bell_measure, transpile_example
from .fake_backends import IQMErrorProfile, IQMFakeAdonis, IQMFakeAphrodite, IQMFakeApollo, IQMFakeDeneb
from .fake_backends.fake_garnet import IQMFakeGarnet
from .fake_backends.iqm_fake_backend import IQMFakeBackend
from .iqm_backend import IQMTarget
from .iqm_circuit import IQMCircuit
from .iqm_circuit_validation import validate_circuit
from .iqm_job import IQMJob
from .iqm_move_layout import generate_initial_layout
from .iqm_naive_move_pass import IQMNaiveResonatorMoving, transpile_to_IQM
from .iqm_provider import IQMBackend, IQMFacadeBackend, IQMProvider, __version__
from .iqm_transpilation import (
    TOLERANCE,
    IQMOptimizeSingleQubitGates,
    IQMReplaceGateWithUnitaryPass,
    optimize_single_qubit_gates,
)
from .move_gate import MOVE_GATE_UNITARY, MoveGate
from .qiskit_to_iqm import MeasurementKey, deserialize_instructions, serialize_instructions
from .transpiler_plugins import (
    IQMSchedulingPlugin,
    MoveGateRoutingKeepExistingMovesPlugin,
    MoveGateRoutingOnlyPlugin,
    MoveGateRoutingPlugin,
    MoveGateRoutingRemoveExistingMovesPlugin,
    MoveGateRoutingTrustExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZKeepExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZPlugin,
    MoveGateRoutingWithExactRZPlugin,
)

if Version(qiskit_version) < Version("2.0.0"):
    warnings.warn(
        DeprecationWarning("Support for Qiskit versions < 2.0 is deprecated and will be removed in a future release.")
    )


from .transpiler_plugins import (
    IQMDefaultSchedulingPlugin,
    MoveGateRoutingIgnoreBarriersKeepExistingMovesPlugin,
    MoveGateRoutingIgnoreBarriersRemoveExistingMovesPlugin,
    MoveGateRoutingIgnoreBarriersTrustExistingMovesPlugin,
    MoveGateRoutingOnlyKeepExistingMovesPlugin,
    MoveGateRoutingOnlyRemoveExistingMovesPlugin,
    MoveGateRoutingOnlyTrustExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZIgnoreBarriersKeepExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZIgnoreBarriersPlugin,
    MoveGateRoutingWithDroppedRZIgnoreBarriersRemoveExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZIgnoreBarriersTrustExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZRemoveExistingMovesPlugin,
    MoveGateRoutingWithDroppedRZTrustExistingMovesPlugin,
    MoveGateRoutingWithRZOptimizationIgnoreBarriersPlugin,
    OnlyRZOptimizationDropFinalRZIgnoreBarriersPlugin,
    OnlyRZOptimizationDropFinalRZPlugin,
    OnlyRZOptimizationExactIgnoreBarriersPlugin,
    OnlyRZOptimizationExactPlugin,
    OnlyRZOptimizationIgnoreBarriersPlugin,
    OnlyRZOptimizationPlugin,
)
