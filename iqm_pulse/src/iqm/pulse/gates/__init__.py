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
"""Implementations of quantum gates.

The :class:`.GateImplementation` subclasses in this subpackage construct :class:`.TimeBox` instances to
implement specific native gates, using the calibration data that the class has been initialized with.
Each GateImplementation instance encapsulates the calibration data for a specific implementation of a specific
native gate acting on a specific locus.

Several different implementations and calibration schemes can be supported for a given gate,
each represented by its own GateImplementation subclass.
Likewise, a single GateImplementation subclass can be sometimes used to implement several different gates
through different calibration data.
"""

from dataclasses import replace

import numpy as np

from iqm.pulse.gate_implementation import GateImplementation
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
from iqm.pulse.gates.measure import Measure_Constant
from iqm.pulse.gates.move import MOVE_CRF_CRF, MOVE_TGSS_CRF
from iqm.pulse.gates.prx import (
    PRX_DRAGCosineRiseFall,
    PRX_DRAGCosineRiseFallSX,
    PRX_DRAGGaussian,
    PRX_DRAGGaussianSX,
    get_unitary_prx,
)
from iqm.pulse.gates.reset import Reset_Conditional, Reset_Wait
from iqm.pulse.gates.rz import RZ_ACStarkShift_CosineRiseFall, RZ_Virtual, get_unitary_rz
from iqm.pulse.gates.sx import SXGate
from iqm.pulse.gates.u import UGate, get_unitary_u
from iqm.pulse.quantum_ops import QuantumOp, QuantumOpTable

_exposed_implementations: dict[str, type[GateImplementation]] = {
    cls.__name__: cls  # type: ignore[misc]
    for cls in (
        Barrier,
        PRX_DRAGGaussian,
        PRX_DRAGCosineRiseFall,
        PRX_DRAGGaussianSX,
        PRX_DRAGCosineRiseFallSX,
        SXGate,
        UGate,
        RZ_Virtual,
        CZ_CRF_ACStarkCRF,
        CZ_Slepian_ACStarkCRF,
        CZ_GaussianSmoothedSquare,
        CZ_Slepian,
        CZ_Slepian_CRF,
        CZ_CRF,
        CZ_TruncatedGaussianSmoothedSquare,
        FluxPulseGate_TGSS_CRF,
        FluxPulseGate_CRF_CRF,
        Measure_Constant,
        MOVE_CRF_CRF,
        MOVE_TGSS_CRF,
        RZ_ACStarkShift_CosineRiseFall,
        CCPRX_Composite,
        CCPRX_Composite_DRAGCosineRiseFall,
        CCPRX_Composite_DRAGGaussian,
        Reset_Conditional,
    )
}
"""These GateImplementations can be referred to in the configuration YAML."""


def get_implementation_class(class_name: str) -> type[GateImplementation] | None:
    """Get gate implementation class by class name."""
    return _exposed_implementations.get(class_name, None)


def expose_implementation(implementation: type[GateImplementation], overwrite: bool = False) -> None:
    """Add the given gate implementation to the list of known implementations.

    Args:
        implementation: GateImplementation to add so that it can be found with :func:`.get_implementation_class`.
        overwrite: If True, does not raise an error if implementation already exists.

    """
    name = implementation.__name__
    if name in _exposed_implementations:
        if not overwrite and _exposed_implementations[name] is not implementation:
            raise ValueError(f"GateImplementation '{name}' has already been defined.")
    _exposed_implementations[name] = implementation


def register_implementation(
    operations: dict[str, QuantumOp],
    gate_name: str,
    impl_name: str,
    impl_class: type[GateImplementation],
    set_as_default: bool = False,
    overwrite: bool = False,
    quantum_op_specs: QuantumOp | dict | None = None,
) -> None:
    """Register a new gate implementation, and a new gate if needed.

    TODO: split the method for adding a new gate implementation and a new gate + implementation

    Args:
        operations: Known operations, mapping gate names to QuantumOps. A new QuantumOp is inserted here.
        gate_name: The gate name for which to register a new implementation.
        impl_name: The "human-readable" name with which the new gate implementation will be found e.g. in settings.
        impl_class: The python class of the new gate implementation to be addded.
        set_as_default: Whether to set the new implementation as the default implementation for the gate.
        overwrite: If True, allows replacing any existing implementation of the same name.
        quantum_op_specs: The quantum operation this gate represents. If a QuantumOp is given, it is used as is.
            If None is given and the same gate has been registered before, the previously registered properties are
            used.
            Otherwise, the given dict values are given to the constructor of :class:`~iqm.pulse.quantum_ops.QuantumOp`.
            For any missing constructor values, some defaults suitable for a 1-QB gate are used.

    """
    if isinstance(quantum_op_specs, QuantumOp):
        new = quantum_op_specs
    elif quantum_op_specs is None and gate_name in operations:
        new = operations[gate_name]
    else:
        new_kwargs = {
            "name": gate_name,
            "arity": 1,
            "params": tuple(),
            "implementations": {},
            "symmetric": impl_class.symmetric,
            "factorizable": False,
        }
        if quantum_op_specs:
            new_kwargs |= quantum_op_specs
            # cast iterables to tuple for consistency
            # TODO: fix the QuantumOp datclass comparisons and field typings properly
            new_kwargs["params"] = tuple(new_kwargs["params"])
        new = QuantumOp(**new_kwargs)

    # TODO: fix the QuantumOp datclass comparisons and field typings properly
    new_without_impls = replace(new, params=tuple(new.params), implementations={}, defaults_for_locus={}, unitary=None)
    if gate_name in _default_operations:
        default_op = _default_operations[gate_name]
        default_without_impls = replace(default_op, implementations={}, defaults_for_locus={}, unitary=None)
        if new_without_impls != default_without_impls:
            raise ValueError(
                f"{gate_name} is one of the default operation defined in iqm-pulse where it has different "
                f"properties. {default_op=}, {new=}"
            )
        if new.unitary is None and default_op.unitary is not None:
            # it is impossible to compare unitary generating functions, so the only thing we can do is to ensure
            # if no unitary was provided, we will still retain the old unitary
            new = replace(new, unitary=default_op.unitary)

    if not overwrite and gate_name in operations:
        old = operations[gate_name]
        # TODO: fix the QuantumOp datclass comparisons and field typings properly
        old_without_impls = replace(
            old, params=tuple(old.params), implementations={}, defaults_for_locus={}, unitary=None
        )
        if new_without_impls != old_without_impls:
            raise ValueError(f"{gate_name} has already been registered with different properties. {old=}, {new=}")
        if new.unitary is None and old.unitary is not None:
            new = replace(new, unitary=old.unitary)

    new.implementations[impl_name] = impl_class
    if set_as_default and len(new.implementations) >= 1:
        new.set_default_implementation(impl_name)
    if not get_implementation_class(impl_class.__name__):
        expose_implementation(impl_class, overwrite)

    operations[gate_name] = new


_default_operations: QuantumOpTable = {
    op.name: op
    for op in [
        QuantumOp(
            "barrier",
            0,
            implementations={"": Barrier},
            symmetric=True,
        ),
        QuantumOp(
            "delay",
            0,
            ("duration",),
            implementations={"wait": Delay},
            symmetric=True,
        ),
        QuantumOp(
            "measure",
            0,
            ("key",),
            implementations={
                "constant": Measure_Constant,
                "constant_qnd": Measure_Constant,
            },
            factorizable=True,
        ),
        QuantumOp(
            "prx",
            1,
            ("angle", "phase"),
            implementations={
                "drag_gaussian": PRX_DRAGGaussian,
                "drag_crf": PRX_DRAGCosineRiseFall,
                "drag_crf_sx": PRX_DRAGCosineRiseFallSX,
                "drag_gaussian_sx": PRX_DRAGGaussianSX,
            },
            unitary=get_unitary_prx,
        ),
        QuantumOp(
            "u",
            1,
            ("theta", "phi", "lam"),
            implementations={"prx_u": UGate},
            unitary=get_unitary_u,
        ),
        QuantumOp("sx", 1, implementations={"prx_sx": SXGate}, unitary=lambda: get_unitary_prx(np.pi / 2, 0)),
        QuantumOp(
            "rz",
            1,
            ("angle",),
            implementations={"virtual": RZ_Virtual},
            unitary=get_unitary_rz,
        ),
        QuantumOp(
            "rz_physical",
            1,
            implementations={"ac_stark_crf": RZ_ACStarkShift_CosineRiseFall},
        ),
        QuantumOp(
            "cz",
            2,
            implementations={
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
            symmetric=True,
            unitary=lambda: np.diag([1.0, 1.0, 1.0, -1.0]),
        ),
        QuantumOp(
            "move",
            2,
            implementations={
                "tgss_crf": MOVE_TGSS_CRF,
                "crf_crf": MOVE_CRF_CRF,
            },
        ),
        QuantumOp(
            "cc_prx",
            1,
            ("angle", "phase", "feedback_qubit", "feedback_key"),
            implementations={
                "prx_composite": CCPRX_Composite,
                "prx_composite_drag_crf": CCPRX_Composite_DRAGCosineRiseFall,
                "prx_composite_drag_gaussian": CCPRX_Composite_DRAGGaussian,
            },
        ),
        QuantumOp(
            "reset",
            0,
            implementations={"reset_conditional": Reset_Conditional},
            symmetric=True,
            factorizable=True,
        ),
        QuantumOp(
            "reset_wait",
            0,
            implementations={"reset_wait": Reset_Wait},
            symmetric=True,
            factorizable=True,
        ),
    ]
}
"""Native quantum operations are hardcoded here. They can be overridden using the YAML config file."""
# TODO for now, the default ops are hardcoded here, but they could always be read from a config file.
