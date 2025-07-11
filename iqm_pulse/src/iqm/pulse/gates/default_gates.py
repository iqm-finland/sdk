# Copyright 2024 IQM
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
"""This file defines the default quantum gates and operations for IQM's pulse control system"""

import numpy as np

from iqm.pulse.gates.barrier import Barrier
from iqm.pulse.gates.conditional import (
    CCPRX_Composite,
    CCPRX_Composite_DRAGCosineRiseFall,
    CCPRX_Composite_DRAGGaussian,
)
from iqm.pulse.gates.cz import (
    CZ_CRF,
    CZ_CRF_ACStarkCRF,
    CZ_GaussianSmoothedSquare,
    CZ_Slepian,
    CZ_Slepian_ACStarkCRF,
    CZ_Slepian_CRF,
    CZ_TruncatedGaussianSmoothedSquare,
    FluxPulseGate_CRF_CRF,
    FluxPulseGate_TGSS_CRF,
)
from iqm.pulse.gates.delay import Delay
from iqm.pulse.gates.flux_multiplexer import FluxMultiplexer_SampleLinear
from iqm.pulse.gates.measure import Measure_Constant, Shelved_Measure_Constant
from iqm.pulse.gates.move import MOVE_CRF_CRF, MOVE_SLEPIAN_CRF, MOVE_TGSS_CRF
from iqm.pulse.gates.prx import (
    PRX_DRAGCosineRiseFall,
    PRX_DRAGCosineRiseFallSX,
    PRX_DRAGGaussian,
    PRX_DRAGGaussianSX,
    PRX_ModulatedDRAGCosineRiseFall,
    get_unitary_prx,
)
from iqm.pulse.gates.reset import Reset_Conditional, Reset_Wait
from iqm.pulse.gates.rz import (
    RZ_ACStarkShift_CosineRiseFall,
    RZ_PRX_Composite,
    RZ_Virtual,
    get_unitary_rz,
)
from iqm.pulse.gates.sx import SXGate
from iqm.pulse.gates.u import UGate, get_unitary_u
from iqm.pulse.quantum_ops import QuantumOp, QuantumOpTable

"""A collection of mappings between default implementation names and their GateImplementation classes,
for different gates."""
_default_implementations = {
    "barrier": {"": Barrier},
    "delay": {"wait": Delay},
    "measure": {
        "constant": Measure_Constant,
        "constant_qnd": Measure_Constant,
        "shelved_constant": Shelved_Measure_Constant,
    },
    "prx": {
        "drag_gaussian": PRX_DRAGGaussian,
        "drag_crf": PRX_DRAGCosineRiseFall,
        "drag_crf_sx": PRX_DRAGCosineRiseFallSX,
        "drag_gaussian_sx": PRX_DRAGGaussianSX,
    },
    "prx_12": {
        "modulated_drag_crf": PRX_ModulatedDRAGCosineRiseFall,
    },
    "u": {"prx_u": UGate},
    "sx": {"prx_sx": SXGate},
    "rz": {"virtual": RZ_Virtual, "prx_composite": RZ_PRX_Composite},
    "rz_physical": {"ac_stark_crf": RZ_ACStarkShift_CosineRiseFall},
    "cz": {
        "tgss": CZ_TruncatedGaussianSmoothedSquare,
        "tgss_crf": FluxPulseGate_TGSS_CRF,
        "crf_crf": FluxPulseGate_CRF_CRF,
        "crf": CZ_CRF,
        "gaussian_smoothed_square": CZ_GaussianSmoothedSquare,
        "slepian": CZ_Slepian,
        "slepian_crf": CZ_Slepian_CRF,
        "crf_acstarkcrf": CZ_CRF_ACStarkCRF,
        "slepian_acstarkcrf": CZ_Slepian_ACStarkCRF,
    },
    "move": {"tgss_crf": MOVE_TGSS_CRF, "crf_crf": MOVE_CRF_CRF, "slepian_crf": MOVE_SLEPIAN_CRF},
    "cc_prx": {
        "prx_composite": CCPRX_Composite,
        "prx_composite_drag_crf": CCPRX_Composite_DRAGCosineRiseFall,
        "prx_composite_drag_gaussian": CCPRX_Composite_DRAGGaussian,
    },
    "reset": {"reset_conditional": Reset_Conditional},
    "reset_wait": {"reset_wait": Reset_Wait},
    "flux_multiplexer": {"sample_linear": FluxMultiplexer_SampleLinear},
}

"""A table of quantum operations (`_default_operations`) defining characteristics:
    - Number of qubits involved (arity)
    - Required parameters
    - GateImplementation classes
    - Unitary matrices where applicable
    - Properties like symmetry and factorizability
This table is here so that we retain information about operations, even though they might 
be deleted in the future.
"""
_default_operations: QuantumOpTable = {
    op.name: op
    for op in [
        QuantumOp(
            "barrier",
            0,
            implementations=_default_implementations["barrier"],  # type: ignore[arg-type]
            symmetric=True,
        ),
        QuantumOp(
            "delay",
            0,
            ("duration",),
            implementations=_default_implementations["delay"],  # type: ignore[arg-type]
            symmetric=True,
        ),
        QuantumOp(
            "measure",
            0,
            ("key",),
            implementations=_default_implementations["measure"],  # type: ignore[arg-type]
            factorizable=True,
        ),
        QuantumOp(
            "prx",
            1,
            ("angle", "phase"),
            implementations=_default_implementations["prx"],  # type: ignore[arg-type]
            unitary=get_unitary_prx,
        ),
        QuantumOp(
            "prx_12",
            1,
            ("angle", "phase"),
            implementations=_default_implementations["prx_12"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "u",
            1,
            ("theta", "phi", "lam"),
            implementations=_default_implementations["u"],  # type: ignore[arg-type]
            unitary=get_unitary_u,
        ),
        QuantumOp(
            "sx",
            1,
            implementations=_default_implementations["sx"],  # type: ignore[arg-type]
            unitary=lambda: get_unitary_prx(np.pi / 2, 0),
        ),
        QuantumOp(
            "rz",
            1,
            ("angle",),
            implementations=_default_implementations["rz"],  # type: ignore[arg-type]
            unitary=get_unitary_rz,
        ),
        QuantumOp(
            "rz_physical",
            1,
            implementations=_default_implementations["rz_physical"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "cz",
            2,
            (),
            implementations=_default_implementations["cz"],  # type: ignore[arg-type]
            symmetric=True,
            unitary=lambda: np.diag([1.0, 1.0, 1.0, -1.0]),
        ),
        QuantumOp(
            "move",
            2,
            implementations=_default_implementations["move"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "cc_prx",
            1,
            ("angle", "phase", "feedback_qubit", "feedback_key"),
            implementations=_default_implementations["cc_prx"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "reset",
            0,
            implementations=_default_implementations["reset"],  # type: ignore[arg-type]
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "reset_wait",
            0,
            implementations=_default_implementations["reset_wait"],  # type: ignore[arg-type]
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "flux_multiplexer",
            0,
            implementations=_default_implementations["flux_multiplexer"],  # type: ignore[arg-type]
        ),
    ]
}

"""A library for all canonical Quantum Operations (gates)"""
_quantum_ops_library = {
    op.name: op
    for op in [
        QuantumOp(
            "barrier",
            0,
            implementations=_default_implementations["barrier"],  # type: ignore[arg-type]
            symmetric=True,
        ),
        QuantumOp(
            "delay",
            0,
            ("duration",),
            implementations=_default_implementations["delay"],  # type: ignore[arg-type]
            symmetric=True,
        ),
        QuantumOp(
            "measure",
            0,
            ("key",),
            implementations=_default_implementations["measure"],  # type: ignore[arg-type]
            factorizable=True,
        ),
        QuantumOp(
            "prx",
            1,
            ("angle", "phase"),
            implementations=_default_implementations["prx"],  # type: ignore[arg-type]
            unitary=get_unitary_prx,
        ),
        QuantumOp(
            "prx_12",
            1,
            ("angle", "phase"),
            implementations=_default_implementations["prx_12"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "u",
            1,
            ("theta", "phi", "lam"),
            implementations=_default_implementations["u"],  # type: ignore[arg-type]
            unitary=get_unitary_u,
        ),
        QuantumOp(
            "sx",
            1,
            implementations=_default_implementations["sx"],  # type: ignore[arg-type]
            unitary=lambda: get_unitary_prx(np.pi / 2, 0),
        ),
        QuantumOp(
            "rz",
            1,
            ("angle",),
            implementations=_default_implementations["rz"],  # type: ignore[arg-type]
            unitary=get_unitary_rz,
        ),
        QuantumOp(
            "rz_physical",
            1,
            implementations=_default_implementations["rz_physical"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "cz",
            2,
            (),
            implementations=_default_implementations["cz"],  # type: ignore[arg-type]
            symmetric=True,
            unitary=lambda: np.diag([1.0, 1.0, 1.0, -1.0]),
        ),
        QuantumOp(
            "move",
            2,
            implementations=_default_implementations["move"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "cc_prx",
            1,
            ("angle", "phase", "feedback_qubit", "feedback_key"),
            implementations=_default_implementations["cc_prx"],  # type: ignore[arg-type]
        ),
        QuantumOp(
            "reset",
            0,
            implementations=_default_implementations["reset"],  # type: ignore[arg-type]
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "reset_wait",
            0,
            implementations=_default_implementations["reset_wait"],  # type: ignore[arg-type]
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "flux_multiplexer",
            0,
            implementations=_default_implementations["flux_multiplexer"],  # type: ignore[arg-type]
        ),
    ]
}
