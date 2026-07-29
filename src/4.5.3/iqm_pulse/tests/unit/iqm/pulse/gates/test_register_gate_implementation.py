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
import pytest

from iqm.pulse.gates import CZ_CRF, CZ_GaussianSmoothedSquare, RZ_Virtual, register_implementation, register_operation
from iqm.pulse.quantum_ops import QuantumOp


def test_changing_properties_of_existing_gate_is_not_allowed():
    op = QuantumOp("existing_gate", 1, ("param1",), implementations={"xxx": RZ_Virtual})
    ops: dict[str, QuantumOp] = {"existing_gate": op}

    with pytest.raises(ValueError, match="'existing_gate' already registered"):
        register_operation(
            ops,
            QuantumOp("existing_gate", implementations={"xxx": RZ_Virtual}),
        )


def test_changing_properties_of_default_gate_is_not_allowed():
    ops: dict[str, QuantumOp] = {}
    with pytest.raises(ValueError, match="'cz' conflicts with a canonical operation in iqm-pulse"):
        register_operation(
            ops,
            QuantumOp("cz", arity=3, symmetric=True, implementations={"xxx": CZ_CRF}),
        )


def test_register_operation_works():
    op = QuantumOp("new", arity=2, params=("a", "b"), symmetric=True, implementations={"xxx": CZ_CRF})
    ops: dict[str, QuantumOp] = {}
    register_operation(ops, op)

    assert ops["new"] == op
    assert ops["new"] is not op  # a copy was inserted
    assert ops["new"].implementations is not op.implementations


def test_cannot_register_implementation_without_quantum_op():
    with pytest.raises(ValueError, match="Operation 'new_gate' is not known, register it first."):
        register_implementation({}, "new_gate", "new_implementation", CZ_GaussianSmoothedSquare)


def test_can_insert_new_implementation_with_quantum_op():
    op = QuantumOp("some_gate", 2, ("param1",), symmetric=True, implementations={"old": CZ_CRF})
    ops: dict[str, QuantumOp] = {"some_gate": op}

    register_implementation(
        ops,
        "some_gate",
        "new",
        CZ_GaussianSmoothedSquare,
        set_as_default=True,
    )

    assert ops["some_gate"].name == "some_gate"
    assert ops["some_gate"].params == op.params
    assert ops["some_gate"].arity == op.arity
    assert len(ops["some_gate"].implementations) == 2
    assert ops["some_gate"].implementations["old"] == CZ_CRF
    assert ops["some_gate"].implementations["new"] == CZ_GaussianSmoothedSquare
    assert ops["some_gate"].default_implementation == "new"


def test_register_existing_implementation_with_different_class_fails():
    """Make sure user cannot change the class of an existing implementation name unless overwrite=True."""
    ops: dict[str, QuantumOp] = {
        "existing_gate": QuantumOp(
            "existing_gate", 2, implementations={"existing_impl": CZ_GaussianSmoothedSquare}, symmetric=True
        )
    }

    # overwrite works
    register_implementation(
        ops,
        "existing_gate",
        "existing_impl",
        CZ_CRF,
        overwrite=True,
    )
    assert ops["existing_gate"].implementations["existing_impl"] is CZ_CRF

    # try switching back without overwrite
    with pytest.raises(
        ValueError,
        match="'existing_gate' already has an implementation named 'existing_impl'",
    ):
        register_implementation(
            ops,
            "existing_gate",
            "existing_impl",
            CZ_GaussianSmoothedSquare,
        )


def test_register_canonical_implementation_with_different_class_fails():
    """Make sure user cannot change the class of a canonical implementation name."""
    ops: dict[str, QuantumOp] = {
        "cz": QuantumOp("cz", 2, implementations={"existing_impl": CZ_GaussianSmoothedSquare}, symmetric=True)
    }
    with pytest.raises(
        ValueError,
        match="'cz': 'tgss' is a reserved implementation name that refers to 'CZ_TruncatedGaussianSmoothedSquare'",
    ):
        register_implementation(
            ops,
            "cz",
            "tgss",
            CZ_GaussianSmoothedSquare,
        )
