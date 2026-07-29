#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from pydantic import ValidationError
import pytest

from exa.common.qcm_data.chad_model import CHAD, Components, ComputationalResonator, Coupler, Launcher, ProbeLine, Qubit


def test_chad_content_format_version_1_0(cheddar_data_1_0):
    chad = CHAD(**cheddar_data_1_0["data"])
    assert chad.mask_set_name == "M156"
    assert chad.variant == "A09"
    assert isinstance(chad.components, Components)
    assert all(isinstance(item, Qubit) for item in chad.components.qubits)
    assert all(isinstance(item, Coupler) for item in chad.components.couplers)
    assert all(isinstance(item, ProbeLine) for item in chad.components.probe_lines)
    assert all(isinstance(item, Launcher) for item in chad.components.launchers)
    assert chad.components.computational_resonators == ()  # Use default empty tuple for missing field


def test_chad_content_format_version_1_1(cheddar_data_1_1):
    chad = CHAD(**cheddar_data_1_1["data"])
    assert chad.mask_set_name == "M139"
    assert chad.variant == "N70"
    assert isinstance(chad.components, Components)
    assert all(isinstance(item, Qubit) for item in chad.components.qubits)
    assert all(isinstance(item, Coupler) for item in chad.components.couplers)
    assert all(isinstance(item, ProbeLine) for item in chad.components.probe_lines)
    assert all(isinstance(item, Launcher) for item in chad.components.launchers)
    assert all(isinstance(item, ComputationalResonator) for item in chad.components.computational_resonators)


def test_chad_model_aliasing(cheddar_data_1_1):
    data = cheddar_data_1_1["data"]
    # Original data is using unusual naming convention where list names are singular instead of plural
    # For example, "qubit" instead of "qubits" for a list of qubits
    assert "qubit" in data["content"]["components"]
    assert "tunable_coupler" in data["content"]["components"]
    assert "probe_line" in data["content"]["components"]
    assert "launcher" in data["content"]["components"]
    assert "computational_resonator" in data["content"]["components"]

    chad = CHAD(**data)

    # Alias list of components to plurals rather than singulars to follow normal naming conventions
    # In addition, alias "tunable_coupler" to simply "couplers" to follow the same naming convention as the code
    assert hasattr(chad.components, "qubits")
    assert hasattr(chad.components, "couplers")
    assert hasattr(chad.components, "probe_lines")
    assert hasattr(chad.components, "launchers")
    assert hasattr(chad.components, "computational_resonators")


def test_chad_model_flattening(cheddar_data_1_1):
    data = cheddar_data_1_1["data"]
    assert "content" in data

    chad = CHAD(**data)

    # Flatten CHAD data to remove unnecessary "content" layer, and use "components" directly instead
    assert not hasattr(chad, "content")
    assert hasattr(chad, "components")


def test_chad_model_can_handle_missing_components_keys_with_default_value(cheddar_data_1_1):
    # Remove components keys which should be accepted as missing and to be used empty tuple by default
    data = cheddar_data_1_1["data"]
    del data["content"]["components"]["qubit"]
    del data["content"]["components"]["tunable_coupler"]
    del data["content"]["components"]["probe_line"]
    del data["content"]["components"]["launcher"]
    del data["content"]["components"]["computational_resonator"]

    try:
        CHAD(**data)
    except ValidationError as error:
        pytest.fail(f"Unexpected ValidationError: {error}")


def test_chad_model_can_handle_missing_connections_with_default_value(cheddar_data_1_1):
    # Remove connections which should be accepted as missing and to be used empty list by default
    data = cheddar_data_1_1["data"]
    for item in data["content"]["components"]["qubit"]:
        del item["connections"]
    for item in data["content"]["components"]["tunable_coupler"]:
        del item["connections"]
    for item in data["content"]["components"]["probe_line"]:
        del item["connections"]
    for item in data["content"]["components"]["launcher"]:
        del item["connections"]
    for item in data["content"]["components"]["computational_resonator"]:
        del item["connections"]

    try:
        CHAD(**data)
    except ValidationError as error:
        pytest.fail(f"Unexpected ValidationError: {error}")


def test_chad_model_immutability(cheddar_data_1_1):
    data = cheddar_data_1_1["data"]
    chad = CHAD(**data)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        # noinspection Pydantic
        chad.mask_set_name = "M140"

    with pytest.raises(ValidationError, match="Instance is frozen"):
        chad.components.qubits[0].connections = []


def test_chad_model_components_sorted(chad):
    qubits = [component.name for component in chad.components.qubits]
    couplers = [component.name for component in chad.components.couplers]
    probe_lines = [component.name for component in chad.components.probe_lines]
    launchers = [component.name for component in chad.components.launchers]
    computational_resonators = [component.name for component in chad.components.computational_resonators]

    assert qubits == ["QB0", "QB1", "QB2", "QB3"]
    assert couplers == ["TC1", "TC2", "TC3"]
    assert probe_lines == ["PL"]
    assert launchers == [
        "DL-QB0",
        "DL-QB1",
        "DL-QB2",
        "DL-QB3",
        "FL-QB0",
        "FL-QB1",
        "FL-QB2",
        "FL-QB3",
        "FL-TC1",
        "FL-TC2",
        "FL-TC3",
        "PL-IN",
        "PL-OUT",
    ]
    assert computational_resonators == ["COMP_R"]


def test_chad_model_connections_sorted(chad):
    assert chad.components.qubits[0].connections == ("COMP_R", "DL-QB0", "FL-QB0", "PL")
    assert chad.components.qubits[1].connections == ("DL-QB1", "FL-QB1", "PL", "TC1")
    assert chad.components.couplers[0].connections == ("COMP_R", "FL-TC1", "QB1")
    assert chad.components.probe_lines[0].connections == ("PL-IN", "PL-OUT", "QB0", "QB1", "QB2", "QB3")
    assert chad.components.launchers[0].connections == ("QB0",)
    assert chad.components.computational_resonators[0].connections == ("QB0", "TC1", "TC2", "TC3")
