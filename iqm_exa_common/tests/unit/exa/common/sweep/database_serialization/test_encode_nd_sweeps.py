#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright (c) 2019-2022 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from collections import namedtuple

import pytest

from exa.common.data.parameter import DataType, Parameter, Sweep
from exa.common.sweep.database_serialization import decode_and_validate_sweeps, encode_nd_sweeps

SweepAndEncoding = namedtuple("SweepAndEncoding", ["sweep", "encoding"])


@pytest.fixture
def sweep_and_encoding():
    return SweepAndEncoding(
        Sweep(parameter=Parameter(name="param1"), data=[1, 2]),
        (
            '{"parameter": {"name": "param1", "label": "param1", "unit": "", "data_type": 1, "collection_type": 0, '
            '"element_indices": null}, "data": [1, 2]}'
        ),
    )


def test_nested_of_tuples_and_lists_is_serialized(sweep_and_encoding):
    sweep = sweep_and_encoding.sweep
    encoded = encode_nd_sweeps([sweep, sweep, (sweep, sweep)])
    encoding = "[" + sweep_and_encoding.encoding + ", " + sweep_and_encoding.encoding
    assert encoded.startswith(encoding)
    assert encoded.endswith(sweep_and_encoding.encoding + ", " + sweep_and_encoding.encoding + "]}]")


def test_complex_numbers_are_encoded_and_decoded_back():
    data_with_complex_numbers = [1 + 2j, 2 + 1j]
    sweep = Sweep(
        parameter=Parameter(name="param1", data_type=DataType.COMPLEX),
        data=data_with_complex_numbers,
    )
    encoded_with_complex_number = encode_nd_sweeps([sweep])
    assert "real" in encoded_with_complex_number
    assert "imag" in encoded_with_complex_number
    assert decode_and_validate_sweeps(encoded_with_complex_number)[0].data == data_with_complex_numbers
