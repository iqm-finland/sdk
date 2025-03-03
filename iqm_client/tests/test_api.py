# Copyright 2021-2023 IQM client developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the IQM client API."""
from mockito import ANY, when
import pytest
import requests

from iqm.iqm_client.api import APIConfig, APIEndpoint, APIVariant
from iqm.iqm_client.iqm_client import IQMClient
from tests.conftest import mock_supported_client_libraries_response


@pytest.fixture
def sample_api_config():
    return APIConfig(APIVariant.V1, "https://example.com")


def test_api_config_initialization(sample_api_config):
    """Test that APIConfig is initialized correctly"""
    assert sample_api_config.variant == APIVariant.V1
    assert sample_api_config.iqm_server_url == "https://example.com"
    assert isinstance(sample_api_config.urls, dict)


@pytest.mark.parametrize("variant", [None, APIVariant.V1, APIVariant.V2, APIVariant.RESONANCE_COCOS_V1])
def test_api_config_is_v1_by_default(variant):
    """Test that APIConfig is V1 by default for backward compatibility."""
    base_url = "https://example.com"
    when(requests).get(f"{base_url}/info/client-libraries", headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response()
    )
    iqm_client = IQMClient(base_url, api_variant=variant)
    assert iqm_client._api.variant == variant if variant is not None else APIVariant.V1


def test_api_config_get_api_urls_invalid_variant():
    """Test that _get_api_urls raises ValueError for invalid variant"""
    with pytest.raises(ValueError, match="Unsupported API variant"):
        APIConfig("INVALID", "https://example.com")._get_api_urls()


@pytest.mark.parametrize("variant", [APIVariant.V1, APIVariant.V2, APIVariant.RESONANCE_COCOS_V1])
def test_api_config_is_supported(variant):
    """Test that is_supported returns correct values"""
    api_config = APIConfig(variant, "https://example.com")
    expected_supported_endpoints = set(api_config.urls.keys())
    expected_not_supported_endpoints = {
        APIEndpoint(endpoint) for endpoint in (set(endpoint for endpoint in APIEndpoint) - expected_supported_endpoints)
    }

    for endpoint in expected_supported_endpoints:
        assert api_config.is_supported(endpoint) is True
    for endpoint in expected_not_supported_endpoints:
        assert api_config.is_supported(endpoint) is False


def test_api_config_url(sample_api_config):
    """Test that url method returns correct URLs"""
    assert sample_api_config.url(APIEndpoint.CONFIGURATION) == "https://example.com/configuration"
    assert sample_api_config.url(APIEndpoint.SUBMIT_JOB) == "https://example.com/jobs"
    assert sample_api_config.url(APIEndpoint.GET_JOB_RESULT, "123") == "https://example.com/jobs/123"


def test_api_config_url_unsupported_endpoint(sample_api_config):
    """Test that url method raises ValueError for unsupported endpoint"""
    with pytest.raises(ValueError, match="Unsupported API endpoint"):
        sample_api_config.url(APIEndpoint.GET_JOB_REQUEST_PARAMETERS)
