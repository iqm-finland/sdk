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
"""Canonical quantum operations and implementations provided by iqm-pulse."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

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
from iqm.pulse.gates.measure import (
    Fast_Measure_Constant,
    Measure_Constant,
    Shelved_Measure_Constant,
)
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
from iqm.pulse.quantum_ops import QuantumOp

if TYPE_CHECKING:
    from iqm.pulse.gate_implementation import GateImplementation


_implementation_library: dict[str, dict[str, type[GateImplementation]]] = {
    "barrier": {"": Barrier},
    "delay": {"wait": Delay},
    "measure": {
        "constant": Measure_Constant,
        "fast_constant": Fast_Measure_Constant,
    },
    "measure_fidelity": {
        "constant": Measure_Constant,
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
"""For each canonical quantum operation name, maps its canonical implementation implementation names
to their GateImplementation classes.

Canonical names are reserved, and the users cannot redefine them.
"""

_quantum_ops_library = {
    op.name: op
    for op in [
        QuantumOp(
            "barrier",
            0,
            implementations=_implementation_library["barrier"],
            symmetric=True,
        ),
        QuantumOp(
            "delay",
            0,
            {"duration": (float,)},
            implementations=_implementation_library["delay"],
            symmetric=True,
        ),
        QuantumOp(
            "measure",
            0,
            {"key": (str,)},
            optional_params={"feedback_key": (str,)},
            implementations=_implementation_library["measure"],
            factorizable=True,
        ),
        QuantumOp(
            "measure_fidelity",
            0,
            {"key": (str,)},
            implementations=_implementation_library["measure_fidelity"],
            factorizable=True,
        ),
        QuantumOp(
            "prx",
            1,
            {
                "angle": (float,),
                "phase": (float,),
            },
            implementations=_implementation_library["prx"],
            unitary=get_unitary_prx,
        ),
        QuantumOp(
            "prx_12",
            1,
            {
                "angle": (float,),
                "phase": (float,),
            },
            implementations=_implementation_library["prx_12"],
        ),
        QuantumOp(
            "u",
            1,
            {
                "theta": (float,),
                "phi": (float,),
                "lam": (float,),
            },
            implementations=_implementation_library["u"],
            unitary=get_unitary_u,
        ),
        QuantumOp(
            "sx",
            1,
            implementations=_implementation_library["sx"],
            unitary=lambda: get_unitary_prx(np.pi / 2, 0),
        ),
        QuantumOp(
            "rz",
            1,
            {"angle": (float,)},
            implementations=_implementation_library["rz"],
            unitary=get_unitary_rz,
        ),
        QuantumOp(
            "rz_physical",
            1,
            implementations=_implementation_library["rz_physical"],
        ),
        QuantumOp(
            "cz",
            2,
            {},
            implementations=_implementation_library["cz"],
            symmetric=True,
            unitary=lambda: np.diag([1.0, 1.0, 1.0, -1.0]),
        ),
        QuantumOp(
            "move",
            2,
            implementations=_implementation_library["move"],
        ),
        QuantumOp(
            "cc_prx",
            1,
            {
                "angle": (float,),
                "phase": (float,),
                "feedback_qubit": (str,),
                "feedback_key": (str,),
            },
            implementations=_implementation_library["cc_prx"],
        ),
        QuantumOp(
            "reset",
            0,
            implementations=_implementation_library["reset"],
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "reset_wait",
            0,
            implementations=_implementation_library["reset_wait"],
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "flux_multiplexer",
            0,
            implementations=_implementation_library["flux_multiplexer"],
        ),
    ]
}
"""Canonical quantum operations provided by iqm-pulse.

Their names are reserved, and the users cannot redefine them.
"""

_deprecated_ops: Final[set[str]] = set()
"""Names of canonical quantum operations that are deprecated.

They are not included by default in ScheduleBuilder unless the user specifically requests them."""
_deprecated_implementations: Final[dict[str, set[str]]] = {}
"""For each canonical quantum operation name, canonical implementation names that are deprecated.

They are not included by default in ScheduleBuilder unless the user specifically requests them."""
# TODO: deprecate gaussian_smoothed_square and everything with the tgss waveform as that is considered inferior to crf.
# TODO: deprecate PRX_drag_gaussian
