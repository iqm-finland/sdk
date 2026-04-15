#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************

import pytest

from exa.common.qcm_data.chad_model import CHAD


@pytest.fixture
def chip_label():
    return "M139_W539_N70_G09"


@pytest.fixture
def cheddar_data_1_0():
    return {
        "data": {
            "mask_set_name": "M156",
            "variant": "A09",
            "content_format_version": "1.0",
            "content": {
                "id": {"variant": "A09", "mask_set_name": "M156"},
                "schema": "https://www.meetiqm.com/chip_architecture_definition_schema_v1.0.json",
                "components": {
                    "qubit": [
                        {"name": "QB1", "connections": ["FL-QB1", "DL-QB1", "TC-1-2", "PL_RO-1"]},
                        {"name": "QB2", "connections": ["FL-QB2", "DL-QB2", "TC-1-2", "TC-2-3", "PL_RO-1"]},
                        {"name": "QB3", "connections": ["FL-QB3", "DL-QB3", "TC-2-3", "PL_RO-1"]},
                    ],
                    "launcher": [
                        {"pin": "1", "name": "RO-1", "function": "probe_in", "connections": ["PL_RO-1"]},
                        {"pin": "2", "name": "RO-2", "function": "probe_out", "connections": ["PL_RO-1"]},
                        {"pin": "3", "name": "FL-QB1", "function": "flux", "connections": ["QB1"]},
                        {"pin": "4", "name": "FL-QB2", "function": "flux", "connections": ["QB2"]},
                        {"pin": "5", "name": "FL-QB3", "function": "flux", "connections": ["QB3"]},
                        {"pin": "6", "name": "FL-TC-1-2", "function": "flux", "connections": ["TC-1-2"]},
                        {"pin": "7", "name": "FL-TC-2-3", "function": "flux", "connections": ["TC-2-3"]},
                        {"pin": "8", "name": "DL-QB1", "function": "drive", "connections": ["QB1"]},
                        {"pin": "9", "name": "DL-QB2", "function": "drive", "connections": ["QB2"]},
                        {"pin": "10", "name": "DL-QB3", "function": "drive", "connections": ["QB3"]},
                    ],
                    "probe_line": [
                        {
                            "name": "PL_RO-1",
                            "connections": ["QB1", "QB2", "QB3", "RO-1", "RO-2"],
                        }
                    ],
                    "tunable_coupler": [
                        {"name": "TC-1-2", "connections": ["FL-TC-1-2", "QB1", "QB2"]},
                        {"name": "TC-2-3", "connections": ["FL-TC-2-3", "QB2", "QB3"]},
                    ],
                },
            },
        }
    }


@pytest.fixture
def cheddar_data_1_1():
    """(M139, N70) Star 7 chip CHAD with some components removed."""
    return {
        "data": {
            "mask_set_name": "M139",
            "variant": "N70",
            "content_format_version": "1.1",
            "content": {
                "id": {"variant": "N70", "mask_set_name": "M139"},
                "schema": "https://www.meetiqm.com/chip_architecture_definition_schema_v1.1.json",
                "components": {
                    "qubit": [
                        {"name": "QB3", "connections": ["FL-QB3", "DL-QB3", "TC3", "PL"]},
                        {"name": "QB1", "connections": ["FL-QB1", "DL-QB1", "TC1", "PL"]},
                        {"name": "QB0", "connections": ["FL-QB0", "DL-QB0", "COMP_R", "PL"]},
                        {
                            "name": "QB2",
                            "connections": ["FL-QB2", "DL-QB2", "TC2", "PL"],
                            "properties": {
                                "design": {
                                    "qubit_type": "",
                                    "kappa_rr": 10000000,
                                    "c_dl": 1.2e-16,
                                    "z_rs": 86.3,
                                    "etch_opposite_face": False,
                                }
                            },
                        },
                    ],
                    "launcher": [
                        {"pin": "1", "name": "FL-QB1", "function": "flux", "connections": ["QB1"]},
                        {"pin": "2", "name": "FL-TC1", "function": "flux", "connections": ["TC1"]},
                        {"pin": "3", "name": "FL-QB3", "function": "flux", "connections": ["QB3"]},
                        {"pin": "4", "name": "DL-QB3", "function": "drive", "connections": ["QB3"]},
                        {"pin": "5", "name": "FL-TC3", "function": "flux", "connections": ["TC3"]},
                        {"pin": "6", "name": "PL-IN", "function": "probe_in", "connections": ["PL"]},
                        {"pin": "13", "name": "PL-OUT", "function": "probe_out", "connections": ["PL"]},
                        {"pin": "17", "name": "FL-TC2", "function": "flux", "connections": ["TC2"]},
                        {"pin": "18", "name": "FL-QB2", "function": "flux", "connections": ["QB2"]},
                        {"pin": "20", "name": "DL-QB2", "function": "drive", "connections": ["QB2"]},
                        {"pin": "21", "name": "FL-QB0", "function": "flux", "connections": ["QB0"]},
                        {"pin": "22", "name": "DL-QB0", "function": "drive", "connections": ["QB0"]},
                        {"pin": "23", "name": "DL-QB1", "function": "drive", "connections": ["QB1"]},
                    ],
                    "probe_line": [
                        {
                            "name": "PL",
                            "connections": ["QB1", "QB0", "QB2", "PL-OUT", "QB3", "PL-IN"],
                        }
                    ],
                    "tunable_coupler": [
                        {"name": "TC3", "connections": ["FL-TC3", "QB3", "COMP_R"]},
                        {"name": "TC2", "connections": ["FL-TC2", "COMP_R", "QB2"]},
                        {"name": "TC1", "connections": ["FL-TC1", "QB1", "COMP_R"]},
                    ],
                    "computational_resonator": [{"name": "COMP_R", "connections": ["TC1", "TC2", "TC3", "QB0"]}],
                },
            },
        }
    }


@pytest.fixture
def cheddar_data_1_1_fake():
    """Handmade weird CHEDDAR used for testing CHAD parsing and methods.
    TODO connections are not symmetric as they should be."""
    return {
        "data": {
            "mask_set_name": "M139",
            "variant": "N70",
            "content_format_version": "1.1",
            "content": {
                "id": {"variant": "N70", "mask_set_name": "M139"},
                "schema": "https://www.meetiqm.com/chip_architecture_definition_schema_v1.0.json",
                "components": {
                    "qubit": [
                        {"name": "QB1", "connections": ["FL-QB1", "DL-QB1", "TC-1-2", "PL_RO-1"]},
                        {"name": "QB3", "connections": ["FL-QB3", "DL-QB3", "TC-2-3", "PL_RO-2"]},
                        {
                            "name": "QB2",
                            "connections": ["FL-QB2", "DL-QB2", "TC-1-2", "TC-2-3", "TC-2-4", "TC-2-5", "PL_RO-1"],
                        },
                        {"name": "QB12", "connections": ["PL_RO-1"]},
                        {
                            "name": "QB4",
                            "connections": ["FL-QB4", "DL-QB4", "TC-2-4", "PL_RO-1"],
                            "properties": {
                                "design": {
                                    "qubit_type": "",
                                    "kappa_rr": 10000000,
                                    "c_dl": 1.2e-16,
                                    "z_rs": 86.3,
                                    "etch_opposite_face": False,
                                }
                            },
                        },
                    ],
                    "launcher": [
                        {"pin": "1", "name": "FL-QB1", "function": "flux", "connections": ["QB1"]},
                        {"pin": "2", "name": "FL-QB2", "function": "flux", "connections": ["QB2"]},
                        {"pin": "3", "name": "FL-QB3", "function": "flux", "connections": ["QB3"]},
                        {"pin": "4", "name": "FL-QB4", "function": "flux", "connections": ["QB4"]},
                        {"pin": "6", "name": "FL-TC-1-2", "function": "flux", "connections": ["TC-1-2"]},
                        {"pin": "7", "name": "FL-TC-2-3", "function": "flux", "connections": ["TC-2-3"]},
                        {"pin": "8", "name": "FL-TC-2-4", "function": "flux", "connections": ["TC-2-4"]},
                        {"pin": "10", "name": "DL-QB1", "function": "drive", "connections": ["QB1"]},
                        {"pin": "11", "name": "DL-QB2", "function": "drive", "connections": ["QB2"]},
                        {"pin": "12", "name": "DL-QB3", "function": "drive", "connections": ["QB3"]},
                        {"pin": "13", "name": "DL-QB4", "function": "drive", "connections": ["QB4"]},
                        {"pin": "15", "name": "RO-1", "function": "probe_in", "connections": ["PL_RO-1"]},
                        {"pin": "16", "name": "RO-2", "function": "probe_out", "connections": ["PL_RO-1"]},
                    ],
                    "probe_line": [
                        {
                            "name": "PL_RO-1",
                            "connections": ["QB1", "QB12", "QB2", "QB4", "QB5", "RO-1", "RO-2", "TC-2-5"],
                        },
                        {"name": "PL_RO-2", "connections": ["QB3", "RO-1", "RO-2"]},
                    ],
                    "tunable_coupler": [
                        {"name": "TC-1-2", "connections": ["FL-TC-1-2", "QB1", "COMP_R"]},
                        {"name": "TC-2-4", "connections": ["FL-TC-2-4", "QB2", "QB4"]},
                        {"name": "TC-2-3", "connections": ["FL-TC-2-3", "QB3", "COMP_R"]},
                        {"name": "TC-2-5", "connections": ["PL_RO-1"]},
                    ],
                    "computational_resonator": [{"name": "COMP_R", "connections": ["TC-1-2", "TC-2-3"]}],
                },
            },
        }
    }


@pytest.fixture
def chad(cheddar_data_1_1):
    chad = CHAD(**cheddar_data_1_1["data"])
    return chad


@pytest.fixture
def cdr(cheddar_data_1_1):
    """Chip design record, as returned by station control (without the "data" layer)"""
    return cheddar_data_1_1["data"]


@pytest.fixture
def fake_chad(cheddar_data_1_1_fake):
    chad = CHAD(**cheddar_data_1_1_fake["data"])
    return chad


@pytest.fixture
def qdp_data_1_1():
    return {
        "data": [
            {
                "id": {"mask_set_name": "M1", "variant": "A"},
                "metadata": {"timestamp": "2024-03-27T16:02:50.227203+02:00", "format_version": "2.0"},
                "content": {
                    "id": {"mask_set_name": "M1", "variant": "A"},
                    "qubits": [],
                },
            }
        ]
    }
