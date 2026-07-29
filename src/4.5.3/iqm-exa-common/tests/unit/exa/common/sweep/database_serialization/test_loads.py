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

from exa.common.data.parameter import Parameter, Sweep
from exa.common.sweep.database_serialization import _loads, encode_nd_sweeps

SweepAndEncoding = namedtuple("SweepAndEncoding", ["sweep", "encoding"])


@pytest.fixture
def sweep_and_encoding():
    return SweepAndEncoding(
        Sweep(parameter=Parameter(name="param1"), data=[1, 2]),
        (
            '{"parameter": {"name": "param1", "label": "param1",'
            + ' "unit": "", "data_type": 1, "collection_type": 0}, "data": [1, 2]}'
        ),
    )


def test_complex_structure_composition_with_dumps_is_identity(sweep_and_encoding):
    sweep = sweep_and_encoding.sweep
    complex_structure = [(sweep,), [sweep, [sweep, [sweep, sweep]]], (sweep, sweep)]
    assert complex_structure == _loads(encode_nd_sweeps(complex_structure))
