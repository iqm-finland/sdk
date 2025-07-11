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
from iqm.pulse.gates.default_gates import _quantum_ops_library
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
    cls.__name__: cls  # type: ignore[misc]
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


def _compare_operations(op1: QuantumOp, op2: QuantumOp) -> bool:
    """Compares two QuantumOp instances. Operations are different only if certain parameters do not match.
    (everything else except the implementations information which the user is allowed to modify and
    :attr:`QuatumOp.unitary` the equality of which cannot be validated as it is a free-form function).

    Args:
        op1: First QuantumOp instance
        op2: Second QuantumOp instance

    Returns:
        True if the operations have identical parameters, False otherwise

    """
    IGNORED_FIELDS = {"implementations", "defaults_for_locus", "unitary"}
    op1_dict = vars(op1)
    op2_dict = vars(op2)
    return all(
        (op1_dict[field] == op2_dict[field] or (field == "params" and tuple(op1_dict[field]) == tuple(op2_dict[field])))
        for field in op1_dict
        if field not in IGNORED_FIELDS
    )


def _validate_operation(
    new_op: QuantumOp,
    gate_name: str,
    operations: QuantumOpTable,
    overwrite: bool = False,
) -> QuantumOp:
    """Validate new operation against existing operations and set unitary if needed.

    Args:
        new_op: Operation to validate
        gate_name: Name of the gate
        operations: Dictionary containing existing operations
        overwrite: Whether to allow overwriting existing operations

    Returns:
        Validated QuantumOp with unitary set if needed

    Raises:
        ValueError: If operation exists with different parameters and overwrite is False

    """
    if not overwrite and gate_name in operations:
        old = operations.get(gate_name)
        same = _compare_operations(new_op, old)  # type: ignore[arg-type]
        if not same:
            raise ValueError(f"{gate_name} already registered with different parameters")

    if gate_name in _quantum_ops_library:
        default = _quantum_ops_library.get(gate_name)
        same = _compare_operations(new_op, default)  # type: ignore[arg-type]
        if not same:
            raise ValueError(f"{gate_name} conflicts with a canonical operation in iqm-pulse")

    if new_op.unitary is None:
        if gate_name in operations and operations[gate_name].unitary is not None:
            unitary = operations[gate_name].unitary
        elif gate_name in _quantum_ops_library and _quantum_ops_library[gate_name].unitary is not None:
            unitary = default.unitary  # type: ignore[union-attr]
        else:
            unitary = None
        new_op = replace(new_op, unitary=unitary)

    return new_op


def _register_gate(
    operations: QuantumOpTable,
    gate_name: str,
    impl_class: type[GateImplementation],
    quantum_op_specs: QuantumOp | dict | None = None,
) -> QuantumOp:
    """Create the quantum operation for the new gate.

    Args:
        operations: Known operations, mapping gate names to QuantumOps
        gate_name: Name of the gate to register
        impl_class: The implementation class
        quantum_op_specs: The quantum operation specifications

    Returns:
        The instance of the new quantum operation

    """
    if isinstance(quantum_op_specs, QuantumOp):
        new_op = quantum_op_specs
    elif quantum_op_specs is None and gate_name in operations:
        new_op = operations[gate_name]
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
            new_kwargs["params"] = tuple(new_kwargs.get("params", ()))  # type: ignore[arg-type]

        new_op = QuantumOp(**new_kwargs)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]

    return new_op


def _add_implementation(
    operations: dict[str, QuantumOp],
    new_op: QuantumOp,
    impl_name: str,
    impl_class: type[GateImplementation],
    set_as_default: bool = False,
    overwrite: bool = False,
) -> None:
    """Register a new implementation for an existing quantum operation.

    Args:
        operations: Table of existing quantum operations
        new_op: The new quantum operation
        impl_name: The name for the implementation that is added
        impl_class: The GateImplementation class corresponding to the new implementation
        set_as_default: Whether to set as default implementation
        overwrite: If True, allows replacing existing implementation

    Returns:
        new_op: QuantumOp with the new implementation

    """
    new_op.implementations[impl_name] = impl_class
    _validate_implementation(operations, new_op, impl_name, impl_class)
    if set_as_default and len(new_op.implementations) >= 1:
        new_op.set_default_implementation(impl_name)
    if not get_implementation_class(impl_class.__name__):
        expose_implementation(impl_class, overwrite)

    return new_op  # type: ignore[return-value]


def _validate_implementation(
    operations: QuantumOpTable,
    new_op: QuantumOp,
    impl_name: str,
    impl_class: type[GateImplementation],
) -> None:
    """Validate new implementation against existing implementations.

    Args:
        operations: Table of quantum operations
        new_op: Operation whose implementation we want to validate
        impl_name: Name of the new implementation
        impl_class: The GateImplementation class corresponding to the new
        implementation

    Raises:
        ValueError: If there is already an implementation with the same name, but
        corresponds to a different implementation class.

    """
    if new_op.name in _quantum_ops_library:
        default_op = _quantum_ops_library[new_op.name]
        for default_impl_name, default_impl_cls in default_op.implementations.items():
            if impl_name == default_impl_name:
                default_impl_cls_name = (
                    default_impl_cls if isinstance(default_impl_cls, str) else default_impl_cls.__name__
                )
                if impl_class.__name__ != default_impl_cls_name:
                    raise ValueError(
                        f"The implementation '{default_impl_name}' already exists in default implementations with '{default_impl_cls.__name__}' as corresponding GateImplementation class."  # noqa: E501
                    )

    if new_op.name in operations:
        existing_op = operations[new_op.name]
        for existing_impl_name, existing_impl_cls in existing_op.implementations.items():
            if impl_name == existing_impl_name:
                existing_impl_cls_name = (
                    existing_impl_cls if isinstance(existing_impl_cls, str) else existing_impl_cls.__name__
                )  # noqa: E501
                if impl_class.__name__ != existing_impl_cls_name:
                    raise ValueError(
                        f"The implementation '{existing_impl_name}' already exists with '{existing_impl_cls.__name__}' as corresponding GateImplementation class."  # noqa: E501
                    )


def register_implementation(
    operations: dict[str, QuantumOp],
    gate_name: str,
    impl_name: str,
    impl_class: type[GateImplementation],
    set_as_default: bool = False,
    overwrite: bool = False,
    quantum_op_specs: QuantumOp | dict | None = None,
) -> None:
    """Register a new gate implementation, and a new gate (operation) if needed.

    Args:
        operations: Known operations, mapping gate names to QuantumOps
        gate_name: The gate name to register
        impl_name: The name for this implementation
        impl_class: The implementation class
        set_as_default: Whether to set as default implementation
        overwrite: If True, allows replacing existing operation/implementation
        quantum_op_specs: Specs for creating new quantum op if needed

    Raises:
        ValueError: If operation/implementation exists and overwrite=False

    """
    new_op = _register_gate(operations, gate_name, impl_class, quantum_op_specs)

    new_op = _validate_operation(new_op, gate_name, operations, overwrite)

    _add_implementation(operations, new_op, impl_name, impl_class, set_as_default, overwrite)

    operations[gate_name] = new_op
