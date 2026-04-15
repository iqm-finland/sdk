# Copyright 2021-2025 IQM client developers
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
"""Tests for the IQMServerClient."""

from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
import pytest

from iqm.station_control.client.authentication import ClientConfigurationError


def test_cocos_fifo_url_is_parsed_correctly():
    result = _IQMServerClient.normalize_url("https://cocos.resonance.meetiqm.com/garnet", None)
    assert result == ("https://resonance.meetiqm.com", "garnet", False)


def test_cocos_timeslot_url_is_parsed_correctly():
    result = _IQMServerClient.normalize_url("https://cocos.resonance.meetiqm.com/garnet:timeslot", None)
    assert result == ("https://resonance.meetiqm.com", "garnet", True)


def test_server_url_is_parsed_correctly():
    result = _IQMServerClient.normalize_url("https://resonance.meetiqm.com", None)
    assert result == ("https://resonance.meetiqm.com", None, False)

    result = _IQMServerClient.normalize_url("https://resonance.meetiqm.com", "garnet")
    assert result == ("https://resonance.meetiqm.com", "garnet", False)

    result = _IQMServerClient.normalize_url("https://ixion.qc.iqm.fi", None)
    assert result == ("https://ixion.qc.iqm.fi", None, False)

    result = _IQMServerClient.normalize_url("http://localhost:49080", None)
    assert result == ("http://localhost:49080", None, False)


def test_scheme_must_be_http_or_https():
    with pytest.raises(
        ClientConfigurationError,
        match="The URL schema has to be http or https. Incorrect schema in URL: ws://ixion.qc.iqm.fi",
    ):
        _IQMServerClient.normalize_url("ws://ixion.qc.iqm.fi", None)


def test_url_must_contain_only_ascii_characters():
    with pytest.raises(ClientConfigurationError, match="Non-ASCII characters in URL: ws://⚛️.qc.iqm.fi"):
        _IQMServerClient.normalize_url("ws://⚛️.qc.iqm.fi", None)
