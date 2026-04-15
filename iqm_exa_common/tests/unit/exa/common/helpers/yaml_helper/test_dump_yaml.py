import logging
import os
from pathlib import Path

from exa.common.helpers.yaml_helper import dump_yaml

data = {
    "name": "name",
    "label": "label",
}


def test_dump_yaml_works_with_path_parameter(tmp_path):
    path = tmp_path / "example.yml"

    assert not path.exists()
    assert isinstance(path, Path)
    dump_yaml(data, path)
    assert path.exists()


def test_dump_yaml_works_with_str_parameter(tmp_path):
    path = tmp_path / "example.yml"
    path = path.absolute().as_posix()

    assert not os.path.isfile(path)
    assert isinstance(path, str)
    dump_yaml(data, path)
    assert os.path.isfile(path)


def test_dump_yaml_creates_missing_directories(tmp_path):
    directory = Path(tmp_path / "sub1" / "sub2")
    path = directory / "example.yml"

    assert not directory.exists()
    dump_yaml(data, path)
    assert directory.exists()


def test_dump_yaml_logs_when_succeed(tmp_path, caplog):
    path = tmp_path / "example.yml"

    with caplog.at_level(logging.DEBUG):
        dump_yaml(data, path)
    expected_message = "Saved a YAML file to"
    assert any(expected_message in rec.message for rec in caplog.records)
