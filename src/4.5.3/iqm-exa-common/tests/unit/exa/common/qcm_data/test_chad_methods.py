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
import pytest

from exa.common.qcm_data.chad_model import Coupler, Qubit


def test_get_component(fake_chad):
    component_name = "QB1"
    component = fake_chad.get_component(component_name)
    assert isinstance(component, Qubit)  # Should return Qubit, not Component
    assert component.name == component_name
    assert component.connections == ("DL-QB1", "FL-QB1", "PL_RO-1", "TC-1-2")  # Sorted alphabetically

    component_name = "TC-1-2"
    component = fake_chad.get_component(component_name)
    assert isinstance(component, Coupler)  # Should return Coupler, not Component
    assert component.name == component_name
    assert component.connections == ("COMP_R", "FL-TC-1-2", "QB1")  # Sorted alphabetically


def test_qubit_names(fake_chad):
    assert fake_chad.qubit_names == ["QB1", "QB2", "QB3", "QB4", "QB12"]


def test_coupler_names(fake_chad):
    assert fake_chad.coupler_names == ["TC-1-2", "TC-2-3", "TC-2-4", "TC-2-5"]


def test_probe_line_names(fake_chad):
    assert fake_chad.probe_line_names == ["PL_RO-1", "PL_RO-2"]


def test_filter_qubit_components(fake_chad):
    components = ["QB1", "QB2"]
    assert fake_chad.filter_qubit_components(components) == ["QB1", "QB2"]

    components = ["QB1", "TC-1-2", "FL-QB1", "DL-QB1", "PL_RO-1", "COMP_R"]
    assert fake_chad.filter_qubit_components(components) == ["QB1"]


def test_filter_qubit_components_duplicate_component_names_raise_error(fake_chad):
    components = ["QB1", "QB1", "QB3"]
    with pytest.raises(ValueError, match="contain duplicates."):
        fake_chad.filter_qubit_components(components)


def test_filter_qubit_components_non_existing_component_names_raise_error(fake_chad):
    components = ["QB1", "QB3", "QB100"]
    with pytest.raises(ValueError, match="The provided component name 'QB100' doesn't exist in the CHAD."):
        fake_chad.filter_qubit_components(components)


def test_get_probe_line_names_for(fake_chad):
    # "QB1" should be connected only to "PL_RO-1"
    components = ["QB1"]
    probe_lines = fake_chad.get_probe_line_names_for(components)
    assert len(probe_lines) == 1
    assert probe_lines[0] == "PL_RO-1"

    # "QB2" should be connected only to "PL_RO-1"
    components = ["QB2"]
    probe_lines = fake_chad.get_probe_line_names_for(components)
    assert len(probe_lines) == 1
    assert probe_lines[0] == "PL_RO-1"

    # "QB3" should be connected only to "PL_RO-2"
    components = ["QB3"]
    probe_lines = fake_chad.get_probe_line_names_for(components)
    assert len(probe_lines) == 1
    assert probe_lines[0] == "PL_RO-2"

    # "QB4" should be connected only to "PL_RO-1"
    components = ["QB4"]
    probe_lines = fake_chad.get_probe_line_names_for(components)
    assert len(probe_lines) == 1
    assert probe_lines[0] == "PL_RO-1"

    # With both "QB1" and "QB3", both "PL_RO-1" and "PL_RO-2" should be returned
    components = ["QB1", "QB3"]
    probe_lines = fake_chad.get_probe_line_names_for(components)
    assert len(probe_lines) == 2
    assert "PL_RO-1" in probe_lines
    assert "PL_RO-2" in probe_lines


def test_get_probe_line_names_for_duplicate_component_names_raise_error(fake_chad):
    components = ["QB1", "QB1", "QB3"]
    with pytest.raises(ValueError, match="contain duplicates."):
        fake_chad.get_probe_line_names_for(components)


def test_get_probe_line_names_for_non_existing_component_names_raise_error(fake_chad):
    components = ["QB1", "QB3", "QB100"]
    with pytest.raises(ValueError, match="The provided component name 'QB100' doesn't exist in the CHAD."):
        fake_chad.get_probe_line_names_for(components)


def test_get_coupler_mapping_for_multiple_qubits(fake_chad):
    components = ["QB1", "QB2"]
    coupler_mapping = fake_chad.get_coupler_mapping_for(components)
    assert coupler_mapping == {
        "TC-1-2": ["QB1", "COMP_R"],
    }

    components = ["QB1", "QB2", "QB3", "QB4"]
    coupler_mapping = fake_chad.get_coupler_mapping_for(components)
    assert coupler_mapping == {
        "TC-1-2": ["QB1", "COMP_R"],
        "TC-2-3": ["QB3", "COMP_R"],
        "TC-2-4": ["QB2", "QB4"],
    }

    components = ["QB1", "QB4"]
    coupler_mapping = fake_chad.get_coupler_mapping_for(components)
    assert coupler_mapping == {
        "TC-1-2": ["QB1", "COMP_R"],
    }


def test_get_coupler_mapping_for_one_qubit_with_computational_resonator_connection(fake_chad):
    components = ["QB1"]
    coupler_mapping = fake_chad.get_coupler_mapping_for(components)
    assert coupler_mapping == {
        "TC-1-2": ["QB1", "COMP_R"],
    }


def test_get_coupler_mapping_for_one_qubit_without_computational_resonator_connection(fake_chad):
    components = ["QB2"]
    coupler_mapping = fake_chad.get_coupler_mapping_for(components)
    assert coupler_mapping == {}


def test_get_coupler_mapping_for_duplicate_component_names_raise_error(fake_chad):
    components = ["QB1", "QB1", "QB3"]
    with pytest.raises(ValueError, match="contain duplicates."):
        fake_chad.get_coupler_mapping_for(components)


def test_get_coupler_mapping_for_non_existing_component_names_raise_error(fake_chad):
    components = ["QB1", "QB3", "QB100"]
    with pytest.raises(ValueError, match="The provided component name 'QB100' doesn't exist in the CHAD."):
        fake_chad.get_coupler_mapping_for(components)


def test_get_probe_line_mapping_for(fake_chad):
    components = ["QB1", "QB2"]
    probe_line_mapping = fake_chad.get_probe_line_mapping_for(components)
    assert probe_line_mapping == {
        "PL_RO-1": ["QB1", "QB2"],
    }

    components = ["QB1", "QB2", "QB3", "QB4"]
    probe_line_mapping = fake_chad.get_probe_line_mapping_for(components)
    assert probe_line_mapping == {
        "PL_RO-1": ["QB1", "QB2", "QB4"],
        "PL_RO-2": ["QB3"],
    }

    components = ["QB1", "QB4"]
    probe_line_mapping = fake_chad.get_probe_line_mapping_for(components)
    assert probe_line_mapping == {
        "PL_RO-1": ["QB1", "QB4"],
    }


def test_get_common_coupler_for(fake_chad):
    # "QB1" and "COMP_R" are connected with "TC-1-2"
    components = ["QB1", "COMP_R"]
    coupler = fake_chad.get_common_coupler_for(components[0], components[1])
    assert coupler == "TC-1-2"

    # "QB2" and "QB4" are connected with "TC-2-4"
    components = ["QB2", "QB4"]
    coupler = fake_chad.get_common_coupler_for(components[0], components[1])
    assert coupler == "TC-2-4"

    # If no common coupler is found raises an error
    components = ["QB1", "QB4"]
    with pytest.raises(ValueError, match="No common coupler was found for"):
        fake_chad.get_common_coupler_for(components[0], components[1])


def test_group_components_per_default_operations(fake_chad):
    qubits = ["QB1", "QB2", "QB3", "QB4"]
    couplers = ["TC-1-2", "TC-2-3", "TC-2-4"]
    components = qubits + couplers
    grouped_qubits, grouped_couplers = fake_chad.group_components_per_default_operations(components)

    assert set(grouped_qubits["readout"]) == set(qubits)
    assert set(grouped_qubits["drive"]) == set(qubits)
    assert set(grouped_qubits["flux"]) == set(qubits)
    assert len(grouped_couplers["readout"]) == 0
    assert len(grouped_couplers["drive"]) == 0
    assert set(grouped_couplers["flux"]) == {"TC-1-2", "TC-2-3", "TC-2-4"}


def test_group_components_per_default_operations_duplicate_component_names_raise_error(fake_chad):
    components = ["QB1", "QB1", "QB3"]
    with pytest.raises(ValueError, match="contain duplicates."):
        fake_chad.group_components_per_default_operations(components)


def test_group_components_per_default_operations_non_existing_component_names_raise_error(fake_chad):
    components = ["QB1", "QB3", "QB100"]
    with pytest.raises(ValueError, match="The provided component name 'QB100' doesn't exist in the CHAD."):
        fake_chad.group_components_per_default_operations(components)
