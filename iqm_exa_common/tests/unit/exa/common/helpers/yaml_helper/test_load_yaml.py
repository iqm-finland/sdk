import logging
import os
from pathlib import Path

import pytest

from exa.common.helpers.yaml_helper import load_yaml

data = {
    "name": "name",
    "label": "label",
}


def test_load_yaml_works_with_path_parameter():
    path = Path(__file__).parent / "example.yml"
    yaml_data = load_yaml(path)
    assert yaml_data == data


def test_load_yaml_works_with_str_parameter(tmp_path):
    path = os.path.join(os.path.dirname(__file__), "example.yml")
    yaml_data = load_yaml(path)
    assert yaml_data == data


def test_load_yaml_file_does_not_exist_raises_error(tmp_path):
    path = Path(__file__).parent / "example_does_not_exist.yml"
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        load_yaml(path)


def test_load_yaml_file_invalid_raises_error(tmp_path):
    path = Path(__file__).parent / "example_invalid.yml"
    with pytest.raises(ValueError, match="Failed to load YAML file from"):
        load_yaml(path)


def test_load_yaml_file_non_safe_raises_error(tmp_path):
    path = Path(__file__).parent / "example_non_safe.yml"
    with pytest.raises(ValueError, match="Failed to load YAML file from"):
        load_yaml(path)


def test_load_yaml_logs_when_succeed(caplog):
    path = Path(__file__).parent / "example.yml"

    with caplog.at_level(logging.DEBUG):
        load_yaml(path)
    expected_message = "Loaded a YAML file from"
    assert any(expected_message in rec.message for rec in caplog.records)
