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
from pathlib import Path

import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter, Sweep
from exa.common.sweep.util import NdSweep


@pytest.fixture
def parameter_mass() -> Parameter:
    return Parameter("mass", "Mass", "g", DataType.FLOAT, CollectionType.SCALAR)


@pytest.fixture
def parameter_length() -> Parameter:
    return Parameter("length", "Length", "m", DataType.COMPLEX, CollectionType.SCALAR)


@pytest.fixture
def parameter_volume() -> Parameter:
    return Parameter("volume", "Volume", "L", DataType.COMPLEX, CollectionType.SCALAR)


def load_return_parameters_json(filename: str) -> str:
    json_content_path = Path(__file__).parent / filename
    with open(json_content_path, "rt") as file:
        return "".join(file.readlines())


@pytest.fixture
def return_parameters5(parameter_mass, parameter_volume, parameter_length) -> tuple[dict[Parameter, NdSweep], str]:
    """Return parameters that were encoded with a ParallelSweep."""
    coord1 = Sweep(
        parameter=parameter_volume, data=[0.1 + 1.0j, 0.2 + 1.0j, 0.3 + 1.0j, 0.5 + 1.0j, 0.8 + 1.0j, 1.3 + 1.0j]
    )
    coord2 = Sweep(parameter=parameter_mass, data=[0.1, 0.2, 0.3, 0.5, 0.8, 1.3])
    json_content = load_return_parameters_json("return_parameters-5.json")
    return {parameter_length: [(coord1, coord2)]}, json_content


@pytest.fixture
def return_parameters6(parameter_mass, parameter_volume, parameter_length) -> tuple[dict[Parameter, NdSweep], str]:
    """Return parameters that were encoded with an NdSweep."""
    coord1 = Sweep(
        parameter=parameter_volume, data=[0.1 + 1.0j, 0.2 + 1.0j, 0.3 + 1.0j, 0.5 + 1.0j, 0.8 + 1.0j, 1.3 + 1.0j]
    )
    coord2 = Sweep(parameter=parameter_mass, data=[0.1, 0.2, 0.3, 0.5, 0.8, 1.3])
    json_content = load_return_parameters_json("return_parameters-6.json")
    return {parameter_length: [(coord1,), (coord2,)]}, json_content


@pytest.fixture
def return_parameters7(parameter_mass) -> tuple[dict[Parameter, None], str]:
    """Return parameters that were encoded with a None."""
    json_content = load_return_parameters_json("return_parameters-7.json")
    return {parameter_mass: None}, json_content


@pytest.fixture
def return_parameters8(parameter_mass) -> tuple[dict[Parameter, list], str]:
    """Return parameters that were encoded with length 1."""
    json_content = load_return_parameters_json("return_parameters-8.json")
    return {parameter_mass: []}, json_content
