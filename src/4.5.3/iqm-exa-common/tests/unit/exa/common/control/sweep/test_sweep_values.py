import json

import numpy as np
from pydantic import TypeAdapter
import pytest

from exa.common.control.sweep.sweep_values import SweepValues


@pytest.mark.parametrize(
    "sweep_values",
    [
        [1],
        [1, 2],
        [1.0],
        [1.0, 2.0],
        [1 + 1j],
        [1 + 1j, 2 + 2j],
        ["a"],
        ["a", "b"],
        [True],
        [True, False],
    ],
)
def test_sweep_values_roundtrip(sweep_values):
    type_adapter = TypeAdapter(SweepValues)
    sweep_values_json = type_adapter.dump_json(sweep_values.copy())
    sweep_values_roundtripped = type_adapter.validate_python(json.loads(sweep_values_json))
    assert sweep_values_roundtripped == sweep_values


@pytest.mark.parametrize(
    "sweep_values",
    [
        np.array([1]),
        np.array([1, 2]),
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.array([1 + 1j]),
        np.array([1 + 1j, 2 + 2j]),
        np.array(["a"]),
        np.array(["a", "b"]),
        np.array([True]),
        np.array([True, False]),
    ],
)
def test_sweep_values_accepts_ndarray(sweep_values):
    # ndarray should be accepted as an input, but it's still serialized to a list.
    # This is just for the user convenience.
    type_adapter = TypeAdapter(SweepValues)

    # Initialization
    validated_sweep_values = type_adapter.validate_python(sweep_values.copy())
    assert isinstance(validated_sweep_values, list)
    assert validated_sweep_values == sweep_values.tolist()

    # Roundtrip
    sweep_values_json = type_adapter.dump_json(sweep_values.copy())
    sweep_values_roundtripped = type_adapter.validate_python(json.loads(sweep_values_json))
    assert isinstance(sweep_values_roundtripped, list)
    assert sweep_values_roundtripped == sweep_values.tolist()


def test_cast_ndarray_to_list_can_be_serialized():
    type_adapter = TypeAdapter(SweepValues)
    # This is not the correct way to convert ndarray to a list, tolist() should be used instead.
    # This "list(np.arange(5))" would normally fail with an error:
    # "Unable to serialize unknown type: <class 'numpy.int64'>"
    # This unit test simply tests that our code can recover from this kind of user error without a failure.
    sweep_values = list(np.arange(5))

    # Roundtrip
    sweep_values_json = type_adapter.dump_json(sweep_values.copy())
    sweep_values_roundtripped = type_adapter.validate_python(json.loads(sweep_values_json))
    assert isinstance(sweep_values_roundtripped, list)
    assert sweep_values_roundtripped == sweep_values
