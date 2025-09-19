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

import copy
from dataclasses import replace

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
    FluxPulse_SmoothConstant_SmoothConstant,
    FluxPulseGate_CRF_CRF,
    FluxPulseGate_TGSS_CRF,
)
from iqm.pulse.gates.default_gates import _implementation_library, _quantum_ops_library
from iqm.pulse.gates.delay import Delay
from iqm.pulse.gates.flux_multiplexer import FluxMultiplexer_SampleLinear
from iqm.pulse.gates.measure import Measure_Constant, Measure_Constant_Qnd, Shelved_Measure_Constant
from iqm.pulse.gates.move import MOVE_CRF_CRF, MOVE_SLEPIAN_CRF, MOVE_TGSS_CRF
from iqm.pulse.gates.prx import (
    Constant_PRX_with_smooth_rise_fall,
    PRX_DRAGCosineRiseFall,
    PRX_DRAGCosineRiseFallSX,
    PRX_DRAGGaussian,
    PRX_DRAGGaussianSX,
    PRX_FastDrag,
    PRX_FastDragSX,
    PRX_HdDrag,
    PRX_HdDragSX,
    PRX_ModulatedDRAGCosineRiseFall,
    get_unitary_prx,
)
from iqm.pulse.gates.reset import Reset_Conditional, Reset_Wait
from iqm.pulse.gates.rz import (
    RZ_ACStarkShift_CosineRiseFall,
    RZ_ACStarkShift_smoothConstant,
    RZ_PRX_Composite,
    RZ_Virtual,
    get_unitary_rz,
)
from iqm.pulse.gates.sx import SXGate
from iqm.pulse.gates.u import UGate, get_unitary_u
from iqm.pulse.quantum_ops import QuantumOp, QuantumOpTable

_exposed_implementations: dict[str, type[GateImplementation]] = {
    cls.__name__: cls
    for cls in (
        Barrier,
        Constant_PRX_with_smooth_rise_fall,
        PRX_DRAGGaussian,
        PRX_DRAGCosineRiseFall,
        PRX_DRAGGaussianSX,
        PRX_DRAGCosineRiseFallSX,
        PRX_FastDragSX,
        PRX_FastDrag,
        PRX_HdDrag,
        PRX_HdDragSX,
        SXGate,
        UGate,
        RZ_PRX_Composite,
        RZ_Virtual,
        CZ_CRF_ACStarkCRF,
        CZ_Slepian_ACStarkCRF,
        CZ_GaussianSmoothedSquare,
        CZ_Slepian,
        CZ_Slepian_CRF,
        CZ_CRF,
        FluxPulse_SmoothConstant_SmoothConstant,
        CZ_TruncatedGaussianSmoothedSquare,
        FluxPulseGate_TGSS_CRF,
        FluxPulseGate_CRF_CRF,
        Measure_Constant,
        Measure_Constant_Qnd,
        Shelved_Measure_Constant,
        PRX_ModulatedDRAGCosineRiseFall,
        MOVE_CRF_CRF,
        MOVE_SLEPIAN_CRF,
        MOVE_TGSS_CRF,
        RZ_ACStarkShift_CosineRiseFall,
        RZ_ACStarkShift_smoothConstant,
        CCPRX_Composite,
        CCPRX_Composite_DRAGCosineRiseFall,
        CCPRX_Composite_DRAGGaussian,
        Reset_Conditional,
    )
}
"""These GateImplementations can be referred to in the configuration YAML."""


def get_implementation_class(class_name: str) -> type[GateImplementation] | None:
    """Get gate implementation class by class name."""
    return _exposed_implementations.get(class_name)


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


def _validate_implementation(
    op_name: str,
    impl_name: str,
    impl_class_name: str,
) -> None:
    """Check that canonical implementation names cannot be overridden.

    Args:
        op_name: Name of the operation.
        impl_name: Name of the new implementation.
        impl_class_name: Name of the GateImplementation class it maps to.

    Raises:
        ValueError: A canonical implementation name is being redefined.

    """
    default_implementations = _implementation_library.get(op_name, {})

    # check if the implementation name is canonical for this op
    if (impl_class := default_implementations.get(impl_name)) is not None:
        if impl_class_name != impl_class.__name__:
            raise ValueError(
                f"'{op_name}': '{impl_name}' is a reserved implementation name that refers to "
                f"'{impl_class.__name__}', and cannot be overridden. "
                "Consider renaming your implementation."
            )


def register_operation(
    operations: dict[str, QuantumOp],
    op: QuantumOp,
    overwrite: bool = False,
) -> None:
    """Register a new QuantumOp to the given operations table.

    Args:
        operations: Known operations, to which the new operation is added.
        op: Definition for the new operation.
        overwrite: If True, allows replacing an existing operation in ``operations``.

    Raises:
        ValueError: ``op.name`` exists in ``operations`` and ``overwrite==False``.
        ValueError: ``op.name`` is the name of a canonical operation in iqm-pulse.

    """
    if op.name in operations and not overwrite:
        raise ValueError(f"'{op.name}' already registered.")
    if op.name in _quantum_ops_library:
        raise ValueError(f"'{op.name}' conflicts with a canonical operation in iqm-pulse. Use a different name.")

    # make a deep copy since the dicts inside QuantumOp are mutable
    operations[op.name] = copy.deepcopy(op)


def register_implementation(
    operations: dict[str, QuantumOp],
    op_name: str,
    impl_name: str,
    impl_class: type[GateImplementation],
    *,
    set_as_default: bool = False,
    overwrite: bool = False,
) -> None:
    """Register a new GateImplementation.

    Args:
        operations: Known operations, to which the new implementation is added.
        op_name: Name of the operation under which the implementation is registered.
        impl_name: Name of the implementation to register.
        impl_class: Implementation class to register.
        set_as_default: Whether to set as the default implementation for ``op_name``.
        overwrite: If True, allows replacing an existing implementation.

    Raises:
        ValueError: ``op_name`` does not exist in ``operations``.
        ValueError: The implementation exists and ``overwrite==False``.


    """
    if (op := operations.get(op_name)) is None:
        raise ValueError(f"Operation '{op_name}' is not known, register it first.")

    # canonical implementation names must not be overridden
    _validate_implementation(op_name, impl_name, impl_class.__name__)

    # only overwrite existing implementations with permission
    if (old_class := op.implementations.get(impl_name)) is not None and not overwrite:
        if old_class is not impl_class:
            # cannot change an existing implementation name
            raise ValueError(f"'{op_name}' already has an implementation named '{impl_name}'.")
    else:
        # add the new implementation
        op.implementations[impl_name] = impl_class

    if set_as_default:
        op.set_default_implementation(impl_name)

    if get_implementation_class(impl_class.__name__) is None:
        expose_implementation(impl_class, overwrite=overwrite)
