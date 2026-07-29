# Copyright 2022 IQM client developers
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
"""Tests for user authentication and token management in IQM client."""

from base64 import b64encode
import json
import re
import time

from mockito import expect, mock, verifyNoUnwantedInteractions, when
import pytest

from iqm.station_control.client.authentication import (
    ClientAuthenticationError,
    ClientConfigurationError,
    ExternalToken,
    TokenManager,
    TokenProviderInterface,
    TokensFileReader,
)
from iqm.station_control.client.station_control import StationControlClient

pytestmark = pytest.mark.usefixtures("unstub")


def make_token(token_type: str, lifetime: int) -> str:
    """Encode given token type and expire time as a token.

    Args:
        token_type: 'Bearer' for access tokens, 'Refresh' for refresh tokens
        lifetime: seconds from current time to token's expire time

    Returns:
        Encoded token
    """
    # TODO we don't use JWT tokens like this anymore, maybe remove this?
    empty = b64encode(b"{}").decode("utf-8")
    body = f'{{ "typ": "{token_type}", "exp": {int(time.time()) + lifetime} }}'
    body = b64encode(body.encode("utf-8")).decode("utf-8")
    return f"{empty}.{body}.{empty}"


class StationControlClientTest(StationControlClient):
    """Bypass the API version check for the tests."""

    def _check_api_versions(self) -> None:
        pass


def test_scc_init_token():
    """SCC can use a token given to it directly."""
    token = "super-secret-token"
    scc = StationControlClientTest("https://localhost/station", token=token)
    assert scc._auth_header_callback is not None
    assert scc._auth_header_callback() == f"Bearer {token}"


def test_scc_init_tokens_file(tmp_path):
    """SCC can use a token given to it in a file."""
    path = str(tmp_path / "tokens_file.json")
    token = make_token("Bearer", 300)
    with open(path, "w", encoding="utf-8") as tokens_file:
        tokens_file.write(json.dumps({"access_token": token}))

    scc = StationControlClientTest("https://localhost/station", tokens_file=path)
    assert scc._auth_header_callback is not None
    assert scc._auth_header_callback() == f"Bearer {token}"


def test_scc_init_get_token_callback():
    """SCC can use a token callback."""
    header = "Bearer super-secret-token"
    scc = StationControlClientTest("https://localhost/station", get_token_callback=lambda: header)
    assert scc._auth_header_callback is not None
    assert scc._auth_header_callback() == header


def test_external_token_provides_token():
    """Tests that ExternalToken provides the configured token"""
    token = make_token("Bearer", 300)

    tokenize_provider = ExternalToken(token)
    assert tokenize_provider.get_token() == token
    with pytest.raises(ClientAuthenticationError, match="Can not close externally managed auth session"):
        tokenize_provider.close()
    with pytest.raises(ClientAuthenticationError, match="No external token available"):
        tokenize_provider.get_token()


def test_tokens_file_reader_provides_token(tmp_path):
    """Tests that TokensFileReader provides the access token stored in the file"""
    path = str(tmp_path / "tokens_file.json")
    token = make_token("Bearer", 300)
    with open(path, "w", encoding="utf-8") as tokens_file:
        tokens_file.write(json.dumps({"access_token": token}))

    token_provider = TokensFileReader(path)
    assert token_provider.get_token() == token
    with pytest.raises(ClientAuthenticationError, match="Can not close externally managed auth session"):
        token_provider.close()
    with pytest.raises(ClientAuthenticationError, match="No tokens file available"):
        token_provider.get_token()


def test_tokens_file_reader_file_not_found(tmp_path):
    """Tests that TokensFileReader raises ClientAuthenticationError if the configured file is not found"""
    path = str(tmp_path / "tokens_file.json")
    token_provider = TokensFileReader(path)
    with pytest.raises(ClientAuthenticationError, match="Failed to read access token"):
        token_provider.get_token()


def test_tokens_file_reader_file_contains_invalid_data(tmp_path):
    """Tests that TokensFileReader raises ClientAuthenticationError if the configured file contains invalid data"""
    path = str(tmp_path / "tokens_file.json")
    with open(path, "w", encoding="utf-8") as tokens_file:
        tokens_file.write("some-invalid-data")

    token_provider = TokensFileReader(path)
    with pytest.raises(ClientAuthenticationError, match="Failed to read access token"):
        token_provider.get_token()
    with pytest.raises(ClientAuthenticationError, match="Can not close externally managed auth session"):
        token_provider.close()
    with pytest.raises(ClientAuthenticationError, match="No tokens file available"):
        token_provider.get_token()


def _patch_env(patcher, **patched):
    for key in ["IQM_TOKEN", "IQM_TOKENS_FILE"]:
        if patched.get(key):
            patcher(key, patched[key])
        else:
            patcher(key, "")


def test_token_manager_initialization_with_keyword_args(monkeypatch) -> None:
    """Test that TokenManager initializes correct token provider based on keyword arguments"""
    _patch_env(monkeypatch.setenv)

    token_manager = TokenManager()
    assert token_manager._token_provider is None

    token = make_token("Bearer", 300)
    token_manager = TokenManager(token=token)
    assert isinstance(token_manager._token_provider, ExternalToken)
    assert token_manager._token_provider._token == token

    path = "/some/path/to/tokens_file.json"
    token_manager = TokenManager(tokens_file=path)
    assert isinstance(token_manager._token_provider, TokensFileReader)
    assert token_manager._token_provider._path == path


def test_token_manager_initialization_with_environment_vars(monkeypatch):
    """Test that TokenManager initializes correct token provider based on environment variables"""
    _patch_env(monkeypatch.setenv)

    token_manager = TokenManager()
    assert token_manager._token_provider is None

    token = make_token("Bearer", 300)
    _patch_env(monkeypatch.setenv, **{"IQM_TOKEN": token})
    token_manager = TokenManager()
    assert isinstance(token_manager._token_provider, ExternalToken)
    assert token_manager._token_provider._token == token

    path = "/some/path/to/tokens_file.json"
    _patch_env(monkeypatch.setenv, **{"IQM_TOKENS_FILE": path})
    token_manager = TokenManager()
    assert isinstance(token_manager._token_provider, TokensFileReader)
    assert token_manager._token_provider._path == path


@pytest.mark.parametrize(
    "args,env",
    [
        # Mixed initialisation parameters and environment variables
        ({"tokens_file": "some-path"}, {"IQM_TOKEN": "some-token"}),
        ({"token": "some-token"}, {"IQM_TOKENS_FILE": "some-path"}),
    ],
)
def test_token_manager_mixed_source_of_parameters(args, env, monkeypatch):
    """Test that TokenManager raises ClientConfigurationError if the parameters are ambiguous"""
    _patch_env(monkeypatch.setenv, **env)
    error_message = "Authentication parameters given both as initialisation args and as environment variables"
    with pytest.raises(ClientConfigurationError, match=error_message):
        TokenManager(**args)


def test_token_manager_no_source_of_parameters_given():
    """Test that the token manager provides no tokens when no parameters have been given or set."""
    token_manager = TokenManager()
    assert token_manager._token_provider is None
    assert token_manager.get_auth_header_callback() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token": "aaa", "tokens_file": "bbb"},
        {"token": "aaa", "auth_header_callback": lambda: "ccc"},
        {"tokens_file": "bbb", "auth_header_callback": lambda: "ccc"},
        {"token": "aaa", "tokens_file": "bbb", "auth_header_callback": lambda: "ccc"},
    ],
)
def test_token_manager_multiple_parameters_given(kwargs):
    """Test that the token manager throws the correct error when multiple parameters have been given."""
    with pytest.raises(
        ClientConfigurationError,
        match=re.escape(f"No more than one authentication parameter may be given, received {list(kwargs)}"),
    ):
        _ = TokenManager(**kwargs)


def test_token_manager_provides_bearer_token(monkeypatch):
    """Test that TokenManager provides bearer token"""

    _patch_env(monkeypatch.setenv)
    expected_token = make_token("Bearer", 300)
    mock_provider = mock(TokenProviderInterface)
    when(mock_provider).get_token().thenReturn(expected_token)

    token_manager = TokenManager()

    # When authentication is not configured _get_bearer_token returns None
    assert token_manager._token_provider is None

    # An existing valid access token is returned instead of asking token_provider for a new one
    token_manager._token_provider = mock_provider
    existing_token = make_token("Bearer", 300)
    token_manager._access_token = existing_token
    assert token_manager._get_bearer_token() == f"Bearer {existing_token}"

    # Otherwise _get_bearer_token returns the token from the token_provider
    token_manager._access_token = None
    assert token_manager._get_bearer_token() == f"Bearer {expected_token}"

    verifyNoUnwantedInteractions()


def test_token_manager_close(monkeypatch):
    """Test that TokenManager closes the token provider"""

    _patch_env(monkeypatch.setenv)
    mock_provider = mock(TokenProviderInterface)
    expect(mock_provider, times=1).close().thenReturn(None)

    # When authentication is not configured there is nothing to close
    token_manager = TokenManager()
    assert not token_manager.close()

    # TokenManager calls `close()` of the token provider, sets token provider to None and returns True
    token_manager._token_provider = mock_provider
    assert token_manager.close()
    assert token_manager._token_provider is None

    verifyNoUnwantedInteractions()
