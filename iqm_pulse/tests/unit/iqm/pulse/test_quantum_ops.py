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
from __future__ import annotations

from pathlib import Path

import pytest

from iqm.pulse.builder import load_config
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.quantum_ops import QuantumOp, validate_op_calibration


class FakeImplementation1(GateImplementation):
    pass


class FakeImplementation2(GateImplementation):
    pass


class FakeImplementation3(GateImplementation):
    pass


class FakeSpecialImplementation(GateImplementation):
    special_implementation = True


def test_symmetry_consistency():
    class SymmetricFakeImplementation(GateImplementation):
        symmetric = True

    with pytest.raises(ValueError, match="test.sym: non-symmetric gate cannot have a symmetric implementation"):
        _ = QuantumOp(
            "test",
            2,
            ("par",),
            implementations={"sym": SymmetricFakeImplementation},
            symmetric=False,
        )


def test_set_default_implementation():
    """Default implementation can be queried and changed."""
    op = QuantumOp(
        "test",
        2,
        implementations={
            "f1": FakeImplementation1,
            "f2": FakeImplementation2,
            "f3": FakeImplementation3,
        },
    )
    assert op.default_implementation == "f1"
    with pytest.raises(ValueError, match="Operation 'test' has no implementation named 'xxx'"):
        op.set_default_implementation("xxx")

    op.set_default_implementation("f1")
    assert op.default_implementation == "f1"
    assert list(op.implementations) == ["f1", "f2", "f3"]

    op.set_default_implementation("f2")
    assert op.default_implementation == "f2"
    # no changes in implementation order
    assert list(op.implementations) == ["f1", "f2", "f3"]
    assert op.implementations["f2"] is FakeImplementation2


def test_set_and_get_default_implementation_for_locus():
    op = QuantumOp(
        "test",
        2,
        implementations={
            "f1": FakeImplementation1,
            "f2": FakeImplementation2,
            "f3": FakeImplementation3,
        },
    )
    assert op.get_default_implementation_for_locus(["QB1", "QB2"]) == "f1"
    op.set_default_implementation_for_locus("f2", ["QB1", "QB2"])
    assert op.defaults_for_locus[("QB1", "QB2")] == "f2"
    assert op.get_default_implementation_for_locus(["QB1", "QB2"]) == "f2"

    with pytest.raises(ValueError, match="Operation 'test' has no implementation named 'f4'"):
        op.set_default_implementation_for_locus("f4", ["QB1", "QB2"])


def test_special_implementations():
    with pytest.raises(ValueError, match="test: a special implementation 'f1' cannot be set"):
        QuantumOp(
            "test",
            2,
            implementations={
                "f1": FakeSpecialImplementation,
                "f2": FakeImplementation2,
                "f3": FakeImplementation3,
            },
        )

    with pytest.raises(ValueError, match="test: a special implementation 'f2' cannot be set"):
        QuantumOp(
            "test",
            2,
            implementations={
                "f1": FakeImplementation1,
                "f2": FakeSpecialImplementation,
                "f3": FakeImplementation3,
            },
            defaults_for_locus={("QB1",): "f2"},
        )
    op = QuantumOp(
        "test",
        2,
        implementations={
            "f1": FakeImplementation1,
            "f2": FakeSpecialImplementation,
            "f3": FakeImplementation3,
        },
    )
    with pytest.raises(ValueError, match="test: a special implementation 'f2' cannot be set as a default"):
        op.set_default_implementation("f2")
    with pytest.raises(ValueError, match="test: a special implementation 'f2' cannot be set as a default"):
        op.set_default_implementation_for_locus("f2", ("QB1",))


@pytest.mark.parametrize(
    "filename, regexp",
    [
        (
            "unknown-op.yml",
            "Unknown operation 'unknown'. "
            + "Known operations: \\('barrier', 'delay', 'measure', 'measure_fidelity', 'prx', 'prx_12', 'u', "
            + "'sx', 'rz', 'rz_physical', 'cz', 'move', 'cc_prx', 'reset', 'reset_wait', 'lru', 'flux_multiplexer'\\)",
        ),
        (
            "unknown-implementation.yml",
            "Unknown implementation 'unknown' for quantum operation 'prx'. "
            + "Known implementations: \\('drag_gaussian',\\)",
        ),
        (
            "zero-arity-multiple-components.yml",
            "dummy.test at \\('QB1', 'QB2'\\): for zero-arity operations, "
            + "calibration data must be provided for single-component loci only",
        ),
        (
            "incorrect-arity.yml",
            "cz.crf at \\('QB1', 'QB2', 'QB5'\\): " + "locus must have 2 component\\(s\\)",
        ),
        (
            "unknown-calibration-data.yml",
            r"cz.crf at \('QB1', 'QB2'\): Unknown calibration data coupler.\{'unknown'\}",
        ),
        (
            "missing-calibration-data.yml",
            r"cz.crf at \('QB1', 'QB2'\): Missing calibration data coupler\.\{'rise_time'\}",
        ),
        (
            "should-be-dict.yml",
            r"cz.crf at \('QB1', 'QB2'\): Calibration data item 'coupler' should be a dict",
        ),
        (
            "should-be-scalar.yml",
            r"prx.drag_gaussian at \('QB1',\): Calibration data item 'full_width' should be a scalar",
        ),
    ],
)
def test_validate_op_calibration(filename, regexp):
    """Test various error cases in op calibration"""
    op_table, calib_data = load_config(str(Path(__file__).parents[3] / f"resources/validation/{filename}"))
    with pytest.raises(ValueError, match=regexp):
        validate_op_calibration(calib_data, op_table)


def test_zero_arity_single_component():
    """Test correct op calibration"""
    op_table, calib_data = load_config(str(Path(__file__).parents[3] / "resources/validation/zero-arity.yml"))
    assert validate_op_calibration(calib_data, op_table) is None
