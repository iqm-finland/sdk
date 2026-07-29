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
"""Tests for CompositeGate class."""

import numpy as np
import pytest

from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.gates import expose_implementation, register_operation
from iqm.pulse.quantum_ops import QuantumOp
from iqm.pulse.timebox import TimeBox


class CompositeHadamard(CompositeGate):
    """Hadamard gate."""

    registered_gates = ["prx"]
    default_implementations = {"prx": "special"}

    def __call__(self) -> TimeBox:
        return TimeBox.composite(
            [self.build("prx", self.locus).ry(np.pi / 2), self.build("prx", self.locus).rx(np.pi)]  # type: ignore
        )


expose_implementation(CompositeHadamard)


@pytest.mark.parametrize(
    "impl_request,impl_got,scale_i",
    [
        (None, "special", 0.2),  # None yields the class default implementation, different from global default
        ("special", "special", 0.2),
        ("drag_gaussian", "drag_gaussian", 0.1),
    ],
)
def test_build_uses_specific_calibration(schedule_builder_with_composite, impl_request, impl_got, scale_i):
    """Composite gate receives specific calibration (from yaml config)."""

    hadamard = schedule_builder_with_composite.get_implementation("hadamard", ("QB1",))
    assert hadamard.default_implementations["prx"] == "special"

    prx = hadamard.build("prx", ("QB1",), impl_name=impl_request)
    assert prx.name == impl_got
    assert prx.pulse.scale_i == scale_i
    assert prx.pulse.scale_q == -0.1214


@pytest.mark.parametrize(
    "impl_request,impl_got",
    [
        (None, "special"),  # None yields the class default implementation, different from global default
        ("special", "special"),
        ("drag_gaussian", "drag_gaussian"),
    ],
)
def test_build_uses_priority_calibration(schedule_builder_with_composite, impl_request, impl_got):
    """Composite gate receives specific calibration (from yaml config), priority calibration overrides it."""
    # add the implementation for hadamard
    op_table = schedule_builder_with_composite.op_table
    op_table["hadamard"] = op_table["hadamard"].copy(implementations={"composite": CompositeHadamard})

    locus = ("QB1",)
    hadamard = schedule_builder_with_composite.get_implementation(
        "hadamard",
        locus,
        priority_calibration={
            "prx": {impl_got: {locus: {"amplitude_i": 100.0}}},  # override
        },
    )
    assert hadamard.default_implementations["prx"] == "special"

    prx = hadamard.build("prx", locus, impl_name=impl_request)
    assert prx.name == impl_got
    assert prx.pulse.scale_i == 100.0
    assert prx.pulse.scale_q == -0.1214


@pytest.mark.parametrize(
    "locus,scales_i",
    [
        (("QB1",), (10.0,)),  # single-qubit locus with specific cal
        (("QB2",), (0.2,)),  # single-qubit locus with global cal
        (("QB1", "QB2"), (10.0, 0.2)),  # multiqubit locus
    ],
)
def test_build_uses_specific_calibration_factorizable(schedule_builder_with_composite, locus, scales_i):
    """Factorizable composite gates like reset handle the calibration in a more complicated way.

    Reset takes no cal data of its own, but can provide specific calibration for its members,
    of which measure is factorizable.
    """
    reset = schedule_builder_with_composite.get_implementation("reset", locus, impl_name="reset_conditional")

    # member gate measure is factorizable
    # we can explicitly build a single-qubit measure
    for q, scale_i in zip(locus, scales_i):
        measure = reset.build("measure", (q,), impl_name="constant")
        assert measure.name == "constant"
        assert not measure.sub_implementations
        assert measure._probe_instruction.scale_i == scale_i

    if len(locus) > 1:
        # or a multiqubit measure
        measure = reset.build("measure", locus, impl_name="constant")
        assert measure.name == "constant"
        assert measure.sub_implementations.keys() == set(locus)
        for q, scale_i in zip(locus, scales_i):
            m = measure.sub_implementations[q]
            assert m.name == "constant"
            assert not m.sub_implementations
            assert m._probe_instruction.scale_i == scale_i


@pytest.mark.parametrize(
    "locus,scales_i",
    [
        (("QB1",), (100.0,)),  # single-qubit locus with specific cal, overridden by priority cal
        (("QB2",), (100.0,)),  # single-qubit locus with global cal, overridden by priority cal
        (("QB1", "QB2"), (100.0, 0.2)),  # multiqubit locus
        (("QB2", "QB1"), (100.0, 10.0)),  # multiqubit locus (specific cal for QB1)
    ],
)
def test_build_uses_priority_calibration_factorizable(schedule_builder_with_composite, locus, scales_i):
    """Factorizable composite gates like reset handle the calibration in a more complicated way.

    Reset takes no cal data of its own, but can provide specific calibration for its members,
    of which measure is factorizable.
    Here we check that priority_calibration overrides both global and specific calibration.
    """
    changed_locus = locus[0:1]  # provide priority cal for the first qubit
    reset = schedule_builder_with_composite.get_implementation(
        "reset",
        locus,
        impl_name="reset_conditional",
        priority_calibration_factorizable={
            changed_locus: {"measure": {"constant": {changed_locus: {"amplitude_i": 100.0}}}},
        },
    )

    # member gate measure is factorizable
    # we can explicitly build a single-qubit measure
    for q, scale_i in zip(locus, scales_i):
        measure = reset.build("measure", (q,), impl_name="constant")
        assert measure.name == "constant"
        assert not measure.sub_implementations
        assert measure._probe_instruction.scale_i == scale_i

    # or a multiqubit measure
    if len(locus) > 1:
        measure = reset.build("measure", locus, impl_name="constant")
        assert measure.name == "constant"
        assert measure.sub_implementations.keys() == set(locus)
        for q, scale_i in zip(locus, scales_i):
            m = measure.sub_implementations[q]
            assert m.name == "constant"
            assert not m.sub_implementations
            assert m._probe_instruction.scale_i == scale_i


def test_factorizable_compositegates_must_have_factorizable_members(schedule_builder):
    """Factorizable CompositeGates may only have factorizable or arity-1 member ops."""

    class BadFactorizableComposite(CompositeGate):
        registered_gates = ["cz"]

    op_table = schedule_builder.op_table
    register_operation(
        op_table,
        QuantumOp(
            "test",
            arity=0,
            implementations={"bad": BadFactorizableComposite},
            factorizable=True,
        ),
    )

    with pytest.raises(
        ValueError, match="'test' is factorizable, but registered gate 'cz' is neither factorizable nor arity-1"
    ):
        schedule_builder.get_implementation("test", ("QB1",))


def test_unknown_registered_gates_not_accepted(schedule_builder_with_composite):
    """The registered gates must be known to the builder."""

    class BrokenComposite(CompositeGate):
        registered_gates = ["fake"]

    op_table = schedule_builder_with_composite.op_table
    register_operation(op_table, QuantumOp("test", implementations={"broken": BrokenComposite}))

    with pytest.raises(ValueError, match="Unknown registered gate 'fake'"):
        schedule_builder_with_composite.get_implementation("test", ("QB1",))
