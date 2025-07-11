#  ********************************************************************************
#
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
"""Quantum operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from itertools import permutations
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from iqm.pulse.base_utils import merge_dicts

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.gate_implementation import GateImplementation, OILCalibrationData, OpCalibrationDataTree


@dataclass(frozen=True)
class QuantumOp:
    """Describes a native quantum operation type.

    *Quantum operations* (or "ops" in short), are simple, abstract, self-contained actions one can
    execute on a station as parts of a quantum circuit. They include quantum gates, measurements,
    and resets. They must have an unambiguous definition in terms of their *intended* effect on the
    computational subspace of the quantum subsystems (qubits, qudits, qumodes...) they act on.
    They are implemented on the hardware using :class:`instruction schedules <.Schedule>`.

    A QuantumOp can also be a *metaoperation*, which (in an idealized picture) has no effect on the
    quantum state, but affects the scheduling of the other ops.
    :class:`Execution barriers <.Barrier>` are an example of a metaoperation.

    The ops can have any number of named parameters. For example, ``PRX`` is a two-parameter
    quantum gate family, whereas ``CZ`` is a single gate with no parameters.

    A *locus* (plural: loci) is a ``tuple[str, ...]`` (an ordered sequence) of CHAD component names
    an instance of a quantum operation acts on. The locus consists of those QPU components that store the
    quantum information the operation acts on. For example, a ``CZ`` gate implemented using a flux
    pulse on the coupler connecting the qubits *does not* include the coupler in its locus, since the
    coupler is simply an implementation detail.

    In a quantum circuit each operation type normally has several different loci. For example, you
    could have a ``PRX`` gate being used on qubits ``{('QB1',), ('QB2',), ('QB5',)}``, or a ``CZ``
    gate used on qubit pairs ``{('QB1', 'QB3'), ('QB3', 'QB5',), ('QB1', 'QB5',)}``.

    Each quantum operation can have any number of named *implementations*, each represented by a
    :class:`.GateImplementation` subclass. For example, we may have two implementations of the CZ gate,
    one with just a single flux pulse applied to the coupler, and another one with additional flux
    pulses applied to the qubits as well.

    * operation defines the abstract intention (what)
    * implementation defines the concrete method (how)
    * locus defines the target of the operation (where)

    The quantum operations are typically calibrated using specific calibration experiments that
    output the required calibration data. Each implementation of each operation can require
    its own, independent set of calibration data for each locus.
    """

    name: str
    """Unique name of the operation."""
    arity: int
    """Number of locus components the operation acts on.
    Each locus component corresponds to a quantum subsystem in the definition of the operation.
    The computational subspace always consists of the lowest two levels of the subsystem.
    Zero means the operation can be applied on any number of locus components."""
    params: tuple[str, ...] = ()
    """Names of required operation parameters, if any."""
    implementations: dict[str, type[GateImplementation]] = field(default_factory=dict)
    """Maps implementation names to :class:`.GateImplementation` classes that provide them.
    Each such class should describe the implementation in detail in its docstring.
    The implementations are in the dict in priority order, highest-priority (default) first. This global priority
    order can be overridden per locus by the attribute ``defaults_for_locus``."""
    symmetric: bool = False
    """True iff the effect of operation is symmetric in the quantum subsystems it acts on.
    Only meaningful if ``self.arity != 1``."""
    factorizable: bool = False
    """True iff the operation is always factorizable to independent single-subsystem operations, which
    is also how it is implemented, for example parallel single-qubit measurements.
    In this case the operation calibration data is for individual subsystems as well."""
    defaults_for_locus: dict[tuple[str, ...], str] = field(default_factory=dict)
    """Optionally define the implementation default individually per each locus. Maps the locus to the default
    gate implementation name. If a locus is not found in this dict (by default, the dict is empty), falls back to the
    global order defined in ``implementations``. The implementations must be first registered in ``implementations``."""
    unitary: Callable[..., np.ndarray] | None = None
    """Unitary matrix that represents the effect of this quantum operation in the computational basis, or ``None``
    if the quantum operation is not unitary or the exact unitary is not known.
    The Callable needs to take exactly the arguments given in :attr:`params`, for example if
    ``params=('angle','phase')``, the function must have signature ``f(angle:float, phase: float) -> np.ndarray``.
    For operations acting on more than 1 qubit, unitary should be given in the big-endian order, i.e. in the basis
    ``np.kron(first_qubit_basis_ket, second_qubit_basis_ket)``."""

    def __post_init__(self):
        for idx, impl_name in enumerate(self.implementations):
            impl_cls = self.implementations[impl_name]
            if impl_cls.symmetric and not self.symmetric:
                raise ValueError(f"{self.name}.{impl_name}: non-symmetric gate cannot have a symmetric implementation.")
            if idx == 0 and impl_cls.special_implementation:
                raise ValueError(
                    f"{self.name}.{impl_name}: a special implementation cannot be set as the default "
                    "(i.e. highest priority) implementation."
                )
        for impl_name in self.defaults_for_locus.values():
            if self.implementations[impl_name].special_implementation:
                raise ValueError(
                    f"{self.name}.{impl_name}: a special implementation cannot be set as a locus specific"
                    " default implementation."
                )

    def copy(self, **changes) -> QuantumOp:
        """Make a copy of ``self`` with the given changes applied to the contents."""
        return replace(self, **changes)

    @property
    def default_implementation(self) -> str:
        """Name of the default implementation (the global default)."""
        return next(iter(self.implementations))

    def set_default_implementation(self, default: str) -> None:
        """Sets the given implementation as the default.

        Args:
            default: name of the new default implementation

        Raises:
            ValueError: ``default`` is unknown or is a special implementation.

        """
        # breakpoint()
        if (impl := self.implementations.get(default)) is None:
            raise ValueError(f"Operation '{self.name}' has no implementation named '{default}'.")

        if impl.special_implementation:
            raise ValueError(f"{default} is a special implementation and cannot be set as the default.")

        # QuantumOp is immutable, the dict is not
        old = self.implementations.copy()
        self.implementations.clear()
        self.implementations[default] = impl
        self.implementations.update(old)

    def get_default_implementation_for_locus(self, locus: Iterable[str]) -> str:
        """Get the default (highest priority) implementation for the given locus.

        If no locus-specific priority is defined, returns the global default.

        Args:
            locus: tuple of component names defining the locus.

        Returns:
            The default implementation name.

        """
        if not self.defaults_for_locus:
            return self.default_implementation
        if not isinstance(locus, tuple):
            locus = tuple(locus)
        if self.arity > 1 and self.symmetric:
            loci = list(permutations(locus))
        else:
            loci = [locus]
        for permuted_locus in loci:
            if permuted_locus in self.defaults_for_locus:
                return self.defaults_for_locus[permuted_locus]
        return self.default_implementation

    def set_default_implementation_for_locus(self, default: str, locus: Iterable[str]) -> None:
        """Set the locus-specific default implementation.

        Args:
            default: name of the new default implementation for ``locus``.
            locus: tuple of component names defining the locus.

        Raises:
            ValueError: if there is no implementation defined with the name ``default`` or ``default`` is a special
                implementation.

        """
        if (impl := self.implementations.get(default)) is None:
            raise ValueError(f"Operation '{self.name}' has no implementation named '{default}'.")
        if impl.special_implementation:
            raise ValueError(f"{default} is a special implementation and cannot be set as a default.")
        if not isinstance(locus, tuple):
            locus = tuple(locus)
        self.defaults_for_locus[locus] = default


QuantumOpTable: TypeAlias = dict[str, QuantumOp]
"""Type for representing tables of known quantum operations, maps names of the ops to their definitions."""


def validate_op_calibration(calibration: OpCalibrationDataTree, ops: QuantumOpTable) -> None:
    """Validates quantum operation calibration data against the known quantum operations.

    NOTE: calibration data parameters that have a defined default value are not required to be in the calibration data.

    Args:
        calibration: quantum operation calibration data tree to validate
        ops: known quantum operations and their implementations

    Raises:
        ValueError: there is something wrong with the calibration data

    """
    for op_name, implementations in calibration.items():
        if (op := ops.get(op_name)) is None:
            raise ValueError(f"Unknown operation '{op_name}'. Known operations: {tuple(ops.keys())}")

        for impl_name, loci in implementations.items():
            if (impl := op.implementations.get(impl_name)) is None:
                raise ValueError(
                    f"Unknown implementation '{impl_name}' for quantum operation '{op_name}'. "
                    f"Known implementations: {tuple(op.implementations.keys())}"
                )

            default_cal_data = loci.get((), {})
            for locus, cal_data in loci.items():
                validate_locus_calibration(merge_dicts(default_cal_data, cal_data), impl, op, impl_name, locus)  # type: ignore[arg-type]


def validate_locus_calibration(
    cal_data: OILCalibrationData, impl: type[GateImplementation], op: QuantumOp, impl_name: str, locus: tuple[str, ...]
) -> None:
    """Validates calibration for a particular gate implementation at particular locus.

    Args:
        cal_data: Calibration data tree for the locus.
        impl: GateImplementation class that defines the required parameters.
        op: QuantumOp that `impl` implements.
        impl_name: name of the implementation, for error messages.
        locus: Locus of the operation

    Raises:
        ValueError: there is something wrong with the calibration data

    """
    if not locus:
        return  # default cal data for all loci
    # Remove registered gates from calibration so that validation has a chance to succeed for composite gates.
    # This validation is done anyway when constructing the registered gates themselves, so it is not necessary
    # to do it twice.
    cal_data_copy = cal_data.copy()
    if hasattr(impl, "registered_gates"):
        registered_gates = impl.registered_gates
        for gate in registered_gates:
            if gate in cal_data_copy:
                del cal_data_copy[gate]
    # since OILCalibrationData can have nested dicts, we do a recursive diff
    _diff_dicts(cal_data_copy, impl.parameters, [], op.name, impl_name, locus)

    n_components = len(locus)
    arity = op.arity
    if arity == 0:
        if n_components != 1:
            raise ValueError(
                f"{op.name}.{impl_name} at {locus}: for zero-arity operations, "
                "calibration data must be provided for single-component loci only"
            )
    elif n_components != arity:
        raise ValueError(f"{op.name}.{impl_name} at {locus}: locus must have {arity} component(s)")


def _diff_dicts(cal_data: dict[str, Any], impl_parameters: dict[str, Any], path, op_name, impl_name, locus) -> None:
    """Compare the calibration"""
    full = set(impl_parameters)
    defaults = {k: v.value for k, v in impl_parameters.items() if hasattr(v, "value")}
    have = {k for k, v in cal_data.items() if v is not None or (k in defaults and defaults[k] is None)}
    need = set(impl_parameters) - set(defaults)
    if need == {"*"}:
        return
    if diff := have - full:
        raise ValueError(f"Unknown calibration data for {op_name}.{impl_name} at {locus}: {'.'.join(path)} {diff}")
    if diff := need - have:
        raise ValueError(f"Missing calibration data for {op_name}.{impl_name} at {locus}: {'.'.join(path)} {diff}")
    for key, data in cal_data.items():
        required_value = impl_parameters[key]
        new_path = path + [key]
        if isinstance(required_value, dict):
            if isinstance(data, dict):
                _diff_dicts(data, required_value, new_path, op_name, impl_name, locus)
            else:
                raise ValueError(
                    f"Calibration data for {op_name}.{impl_name} at {locus}: '{'.'.join(new_path)}' should be a dict"
                )
        elif isinstance(data, dict):
            raise ValueError(
                f"Calibration data for {op_name}.{impl_name} at {locus}: '{'.'.join(new_path)}' should be a scalar"
            )
