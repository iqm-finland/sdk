#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import pytest

import exa.common.api.proto_serialization as protos
from exa.common.data.parameter import CollectionType, DataType, Parameter, Sweep


@pytest.mark.parametrize("minimal", [True, False])
def test_reversibility(minimal: bool):
    list_param = Parameter("dim2c", data_type=DataType.FLOAT, collection_type=CollectionType.LIST)
    element_parameter_0 = list_param.create_element_parameter_for(0)
    element_parameter_2 = list_param.create_element_parameter_for([2])
    nd_sweep = [
        (
            Sweep(parameter=Parameter("dim1a", data_type=DataType.INT), data=[1, 2, 3]),
            Sweep(parameter=Parameter("dim1b", data_type=DataType.FLOAT), data=[-0.1, -0.2, -0.3]),
        ),
        (
            Sweep(parameter=Parameter("dim2a", data_type=DataType.COMPLEX), data=[1j, 2j, 3j]),
            Sweep(parameter=Parameter("dim2b", data_type=DataType.BOOLEAN), data=[True, False, True]),
            Sweep(parameter=element_parameter_0, data=[0.1, 0.2]),
            Sweep(parameter=element_parameter_2, data=[0.01, 0.02]),
        ),
        (Sweep(parameter=Parameter("dim3", data_type=DataType.STRING), data=["a", "b"]),),
    ]
    packed = protos.nd_sweep.pack(nd_sweep, minimal=minimal)
    unpacked = protos.nd_sweep.unpack(packed)
    for original_parallel, unpacked_parallel in zip(nd_sweep, unpacked):
        for original_single, unpacked_single in zip(original_parallel, unpacked_parallel):
            assert unpacked_single.data == original_single.data
            assert unpacked_single.parameter.name == original_single.parameter.name
            if minimal:
                assert unpacked_single.parameter.data_type == DataType.ANYTHING
            else:
                assert unpacked_single.parameter == original_single.parameter
