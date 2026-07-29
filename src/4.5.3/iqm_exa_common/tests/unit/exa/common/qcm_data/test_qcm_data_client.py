#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from http import HTTPStatus
import json
from pathlib import Path
import tempfile

from mockito import contains, mock, when
from packaging.version import Version  # noqa: F401
import pytest
from pytest import raises
from requests.exceptions import InvalidSchema

from exa.common.errors.exa_error import ExaError
from exa.common.errors.station_control_errors import NotFoundError
from exa.common.qcm_data.qcm_data_client import (
    # noqa: F401
    # noqa: F401
    QCMDataClient,
)

pytestmark = pytest.mark.usefixtures("unstub")

QCM_DATA_API_URL = "https://test-qcm-rest.fi"


def contains_cheddar_endpoint():
    return contains("/cheddars/")


@pytest.fixture
def qcm_data_client(chip_label, cheddar_data_1_1):
    mock_response = mock(
        {
            "status_code": HTTPStatus.OK,
            "json": lambda: cheddar_data_1_1,
        }
    )

    qcm_data_client = QCMDataClient(root_url=QCM_DATA_API_URL)
    when(qcm_data_client.session).get(contains_cheddar_endpoint(), ...).thenReturn(mock_response)
    return qcm_data_client


def test_init_with_empty_url():
    with raises(ValueError, match="QCMDataClient 'root_url' cannot be empty, it must be a valid HTTP or file URL."):
        QCMDataClient(root_url="")


def test_get_chip_design_record_with_http_url(qcm_data_client, chip_label, cheddar_data_1_1):
    data = qcm_data_client.get_chip_design_record(chip_label)
    assert data == cheddar_data_1_1["data"]


def test_get_chip_design_record_with_file_url(chip_label, cheddar_data_1_1):
    temporary_directory = tempfile.TemporaryDirectory()
    root_dir = Path(temporary_directory.name)
    chip_design_record_dir = Path(root_dir, "cheddars")
    chip_design_record_dir.mkdir(parents=True)

    with open(Path(chip_design_record_dir, "M139_W539_N70_G09.json"), "w", encoding="utf-8") as chip_design_record_file:
        chip_design_record_file.write(json.dumps(cheddar_data_1_1))

    qcm_data_client = QCMDataClient(root_url=Path(root_dir).as_uri())

    data = qcm_data_client.get_chip_design_record(chip_label)
    assert data == cheddar_data_1_1["data"]

    try:
        temporary_directory.cleanup()
    except NotADirectoryError:
        pass


def test_get_chip_design_record_with_file_not_found(chip_label):
    temporary_directory = tempfile.TemporaryDirectory()
    root_dir = Path(temporary_directory.name)

    qcm_data_client = QCMDataClient(root_url=Path(root_dir).as_uri())

    with raises(NotFoundError, match="No such file or directory"):
        qcm_data_client.get_chip_design_record(chip_label)

    try:
        temporary_directory.cleanup()
    except NotADirectoryError:
        pass


def test_get_chip_design_record_with_invalid_schema(chip_label):
    qcm_data_client = QCMDataClient(root_url="xyz://invalid/chip_design_record/url")
    with raises(InvalidSchema, match="No connection adapters were found for 'xyz://"):
        qcm_data_client.get_chip_design_record(chip_label)


def test_get_chip_design_record_cache_is_functioning_correctly_when_same_root_url(qcm_data_client, chip_label):
    # Initially, cache must be empty
    cache_info = qcm_data_client._send_request.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 0
    assert cache_info.currsize == 0

    # First get should be a miss, and the result should be added to the cache
    qcm_data_client.get_chip_design_record(chip_label)
    cache_info = qcm_data_client._send_request.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1
    assert cache_info.currsize == 1

    # Second get should be a hit, and the result shouldn't be added to the cache again
    qcm_data_client.get_chip_design_record(chip_label)
    cache_info = qcm_data_client._send_request.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1
    assert cache_info.currsize == 1


def test_get_chip_design_record_cache_is_functioning_correctly_when_root_url_changed(qcm_data_client, chip_label):
    # First get should be a miss, and the result should be added to the cache
    qcm_data_client.root_url = QCM_DATA_API_URL
    qcm_data_client.get_chip_design_record(chip_label)
    cache_info = qcm_data_client._send_request.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1
    assert cache_info.currsize == 1

    # When root URL is changed, second get should be a miss since root URL should be a part of cached request as well
    qcm_data_client.root_url = "https://test-qcm-rest-2.fi"
    qcm_data_client.get_chip_design_record(chip_label)
    cache_info = qcm_data_client._send_request.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 2
    assert cache_info.currsize == 2

    # Going back to the original root URL, first get should be still in the cache, thus we should get a hit
    qcm_data_client.root_url = QCM_DATA_API_URL
    qcm_data_client.get_chip_design_record(chip_label)
    cache_info = qcm_data_client._send_request.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 2
    assert cache_info.currsize == 2


def test_uses_fallback_url(cheddar_data_1_1, chip_label):
    mock_response = mock({"status_code": HTTPStatus.OK, "json": lambda: cheddar_data_1_1})
    qcm_data_client = QCMDataClient(root_url="file:///no/such/dir", fallback_root_url=QCM_DATA_API_URL)
    when(qcm_data_client.session).get(contains(QCM_DATA_API_URL), ...).thenReturn(mock_response)
    when(qcm_data_client.session).get(contains("no/such/dir"), ...).thenRaise(NotFoundError("Not found"))

    assert qcm_data_client.get_chip_design_record(chip_label) == cheddar_data_1_1["data"]


@pytest.mark.parametrize("version", ["0.0", "0.9", "3.0"])
def test_validate_chip_design_record_content_format_version_invalid_versions(chip_label, cheddar_data_1_1, version):
    cheddar_data_1_1["data"]["content_format_version"] = version
    mock_response = mock(
        {
            "status_code": HTTPStatus.OK,
            "json": lambda: cheddar_data_1_1,
        }
    )

    qcm_data_client = QCMDataClient(root_url=QCM_DATA_API_URL)
    when(qcm_data_client.session).get(contains_cheddar_endpoint(), ...).thenReturn(mock_response)

    with raises(ExaError, match=f"CHAD content format version '{version}' is not supported."):
        qcm_data_client.get_chip_design_record(chip_label)


@pytest.mark.parametrize("version", ["1.0", "1.1", "2.0", "2.1", "2.99"])
def test_validate_chip_design_record_content_format_version_valid_versions(chip_label, cheddar_data_1_1, version):
    cheddar_data_1_1["data"]["content_format_version"] = version
    mock_response = mock(
        {
            "status_code": HTTPStatus.OK,
            "json": lambda: cheddar_data_1_1,
        }
    )

    qcm_data_client = QCMDataClient(root_url=QCM_DATA_API_URL)
    when(qcm_data_client.session).get(contains_cheddar_endpoint(), ...).thenReturn(mock_response)

    cda = qcm_data_client.get_chip_design_record(chip_label)
    assert cda


def test_validate_chip_design_record_mask_set_name(chip_label, cheddar_data_1_1):
    cheddar_data_1_1["data"]["mask_set_name"] = "M117"
    mock_response = mock(
        {
            "status_code": HTTPStatus.OK,
            "json": lambda: cheddar_data_1_1,
        }
    )

    qcm_data_client = QCMDataClient(root_url=QCM_DATA_API_URL)
    when(qcm_data_client.session).get(contains_cheddar_endpoint(), ...).thenReturn(mock_response)

    with raises(ExaError, match="CHAD mask set name 'M117' doesn't match the chip label mask set name 'M139'."):
        qcm_data_client.get_chip_design_record(chip_label)


def test_validate_chip_design_record_variant(chip_label, cheddar_data_1_1):
    cheddar_data_1_1["data"]["variant"] = "A11"
    mock_response = mock(
        {
            "status_code": HTTPStatus.OK,
            "json": lambda: cheddar_data_1_1,
        }
    )

    qcm_data_client = QCMDataClient(root_url=QCM_DATA_API_URL)
    when(qcm_data_client.session).get(contains_cheddar_endpoint(), ...).thenReturn(mock_response)

    with raises(ExaError, match="CHAD variant 'A11' doesn't match the chip label variant 'N70'."):
        qcm_data_client.get_chip_design_record(chip_label)
