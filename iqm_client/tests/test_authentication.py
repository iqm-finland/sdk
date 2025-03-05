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
# pylint: disable=too-many-arguments
import json
from uuid import UUID, uuid4

from freezegun import freeze_time
from mockito import expect, mock, unstub, verifyNoUnwantedInteractions, when
from mockito.matchers import ANY
import pytest
import requests

from iqm.iqm_client import (
    REQUESTS_TIMEOUT,
    ClientAuthenticationError,
    ClientConfigurationError,
    IQMClient,
    RunResult,
    RunStatus,
)
from iqm.iqm_client.authentication import (
    AUTH_CLIENT_ID,
    AUTH_REALM,
    AUTH_REQUESTS_TIMEOUT,
    ExternalToken,
    TokenClient,
    TokenManager,
    TokenProviderInterface,
    TokensFileReader,
)
from tests.conftest import MockJsonResponse, make_token, mock_supported_client_libraries_response


@pytest.fixture()
def auth_server_url() -> str:
    """Authentication server base URL"""
    return 'https://example.com/auth'


@pytest.fixture()
def auth_realm() -> str:
    """Authentication realm for token client tests"""
    return 'test_realm'


@pytest.fixture()
def auth_username() -> str:
    """Username for getting token from an authentication server"""
    return 'some-user'


@pytest.fixture()
def auth_password() -> str:
    """Username for getting token from an authentication server"""
    return 'very-secret'


def _make_client_with_token(base_url) -> tuple[str, IQMClient]:
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response()
    )
    token = make_token('Bearer', 300)
    client = IQMClient(base_url, token=token)
    return token, client


def test_external_token_provides_token():
    """Tests that ExternalToken provides the configured token"""
    token = make_token('Bearer', 300)

    provider = ExternalToken(token)
    assert provider.get_token() == token
    with pytest.raises(ClientAuthenticationError, match='Can not close externally managed auth session'):
        provider.close()
    with pytest.raises(ClientAuthenticationError, match='No external token available'):
        provider.get_token()

    verifyNoUnwantedInteractions()
    unstub()


def test_tokens_file_reader_provides_token(tmp_path):
    """Tests that TokensFileReader provides the access token stored in the file"""
    path = str(tmp_path / 'tokens_file.json')
    token = make_token('Bearer', 300)
    with open(path, 'w', encoding='utf-8') as tokens_file:
        tokens_file.write(json.dumps({'access_token': token}))

    provider = TokensFileReader(path)
    assert provider.get_token() == token
    with pytest.raises(ClientAuthenticationError, match='Can not close externally managed auth session'):
        provider.close()
    with pytest.raises(ClientAuthenticationError, match='No tokens file available'):
        provider.get_token()

    verifyNoUnwantedInteractions()
    unstub()


def test_tokens_file_reader_file_not_found(tmp_path):
    """Tests that TokensFileReader raises ClientAuthenticationError if the configured file is not found"""
    path = str(tmp_path / 'tokens_file.json')
    provider = TokensFileReader(path)
    with pytest.raises(ClientAuthenticationError, match='Failed to read access token'):
        provider.get_token()

    verifyNoUnwantedInteractions()
    unstub()


def test_tokens_file_reader_file_contains_invalid_data(tmp_path):
    """Tests that TokensFileReader raises ClientAuthenticaitonError if the configured file contains invalid data"""
    path = str(tmp_path / 'tokens_file.json')
    with open(path, 'w', encoding='utf-8') as tokens_file:
        tokens_file.write('some-invalid-data')

    provider = TokensFileReader(path)
    with pytest.raises(ClientAuthenticationError, match='Failed to read access token'):
        provider.get_token()
    with pytest.raises(ClientAuthenticationError, match='Can not close externally managed auth session'):
        provider.close()
    with pytest.raises(ClientAuthenticationError, match='No tokens file available'):
        provider.get_token()

    verifyNoUnwantedInteractions()
    unstub()


def test_token_client_provides_token(auth_server_url, auth_realm, auth_username, auth_password):
    """Tests that TokenClient provides the access token acquired from an authentication server"""
    # pylint: disable=too-many-locals

    token_url = f'{auth_server_url}/realms/{auth_realm}/protocol/openid-connect/token'
    logout_url = f'{auth_server_url}/realms/{auth_realm}/protocol/openid-connect/logout'
    provider = TokenClient(auth_server_url, auth_realm, auth_username, auth_password)

    # login
    with freeze_time('2024-05-20 12:00:00'):
        login_data1 = {
            'client_id': AUTH_CLIENT_ID,
            'grant_type': 'password',
            'username': auth_username,
            'password': auth_password,
        }
        access_token1 = make_token('Bearer', 300)
        refresh_token1 = make_token('Refresh', 3000)

        expect(requests, times=1).post(token_url, data=login_data1, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
            MockJsonResponse(200, {'access_token': access_token1, 'refresh_token': refresh_token1})
        )

        assert provider.get_token() == access_token1

    # 1st refresh
    with freeze_time('2024-05-20 12:01:00'):
        refresh_data1 = {'client_id': AUTH_CLIENT_ID, 'grant_type': 'refresh_token', 'refresh_token': refresh_token1}
        access_token2 = make_token('Bearer', 300)
        refresh_token2 = make_token('Refresh', 3000)

        expect(requests, times=1).post(token_url, data=refresh_data1, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
            MockJsonResponse(200, {'access_token': access_token2, 'refresh_token': refresh_token2})
        )

        assert provider.get_token() == access_token2

    # 2nd refresh
    with freeze_time('2024-05-20 12:02:00'):
        refresh_data2 = {'client_id': AUTH_CLIENT_ID, 'grant_type': 'refresh_token', 'refresh_token': refresh_token2}
        access_token3 = make_token('Bearer', 300)
        refresh_token3 = make_token('Refresh', 3000)

        expect(requests, times=1).post(token_url, data=refresh_data2, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
            MockJsonResponse(200, {'access_token': access_token3, 'refresh_token': refresh_token3})
        )

        assert provider.get_token() == access_token3

    # logout
    with freeze_time('2024-05-20 12:03:00'):
        logout_data = {'client_id': AUTH_CLIENT_ID, 'refresh_token': refresh_token3}

        expect(requests, times=1).post(logout_url, data=logout_data, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
            MockJsonResponse(200, {})
        )

        provider.close()

    # new session
    with freeze_time('2024-05-20 12:04:00'):
        login_data2 = {
            'client_id': AUTH_CLIENT_ID,
            'grant_type': 'password',
            'username': auth_username,
            'password': auth_password,
        }
        access_token4 = make_token('Bearer', 300)
        refresh_token4 = make_token('Refresh', 3000)

        expect(requests, times=1).post(token_url, data=login_data2, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
            MockJsonResponse(200, {'access_token': access_token4, 'refresh_token': refresh_token4})
        )

        assert provider.get_token() == access_token4

    verifyNoUnwantedInteractions()
    unstub()


def test_token_client_login_fails_with_no_auth_server():
    """Tests that TokenClient raises ClientAuthenticationError if auth server has not been set"""

    provider = TokenClient('', '', '', '')
    provider._token_url = ''
    with pytest.raises(ClientConfigurationError, match='No auth server configured'):
        provider.get_token()

    verifyNoUnwantedInteractions()
    unstub()


def test_token_client_login_fails_with_wrong_password(auth_server_url, auth_realm, auth_username, auth_password):
    """Tests that TokenClient raises ClientAuthenticationError if server login fails"""

    token_url = f'{auth_server_url}/realms/{auth_realm}/protocol/openid-connect/token'
    login_data = {
        'client_id': AUTH_CLIENT_ID,
        'grant_type': 'password',
        'username': auth_username,
        'password': auth_password,
    }
    response_body = {'detail': 'invalid password'}

    # Prepare login
    expect(requests, times=1).post(token_url, data=login_data, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
        MockJsonResponse(401, response_body)
    )

    provider = TokenClient(auth_server_url, auth_realm, auth_username, auth_password)
    with pytest.raises(ClientAuthenticationError, match='Getting access token from auth server failed'):
        provider.get_token()

    verifyNoUnwantedInteractions()
    unstub()


def test_token_client_recovers_from_revoked_session(auth_server_url, auth_realm, auth_username, auth_password):
    """Tests that TokenClient starts a new session when getting new tokens using the refresh token fails."""

    refresh_token = make_token('Refresh', 3000)
    token_url = f'{auth_server_url}/realms/{auth_realm}/protocol/openid-connect/token'
    refresh_data = {
        'client_id': AUTH_CLIENT_ID,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
    }
    refresh_response_body = {'detail': 'session has been revoked'}

    # Prepare refresh request
    expect(requests, times=1).post(token_url, data=refresh_data, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
        MockJsonResponse(401, refresh_response_body)
    )

    login_data = {
        'client_id': AUTH_CLIENT_ID,
        'grant_type': 'password',
        'username': auth_username,
        'password': auth_password,
    }
    login_response_body = {
        'access_token': make_token('Bearer', 300),
        'refresh_token': make_token('Refresh', 3000),
    }

    # Prepare login request
    expect(requests, times=1).post(token_url, data=login_data, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
        MockJsonResponse(200, login_response_body)
    )

    provider = TokenClient(auth_server_url, auth_realm, auth_username, auth_password)
    provider._refresh_token = refresh_token

    assert provider.get_token() == login_response_body['access_token']

    verifyNoUnwantedInteractions()
    unstub()


def test_token_client_logout_fails(auth_server_url, auth_realm, auth_username, auth_password):
    """Tests that TokenClient raises ClientAuthenticationError if logout fails"""

    token_url = f'{auth_server_url}/realms/{auth_realm}/protocol/openid-connect/token'
    logout_url = f'{auth_server_url}/realms/{auth_realm}/protocol/openid-connect/logout'

    access_token1 = make_token('Bearer', 300)
    refresh_token1 = make_token('Refresh', 3000)

    login_data = {
        'client_id': AUTH_CLIENT_ID,
        'grant_type': 'password',
        'username': auth_username,
        'password': auth_password,
    }
    logout_data = {'client_id': AUTH_CLIENT_ID, 'refresh_token': refresh_token1}
    response_body = {'detail': 'unknown session'}

    # Prepare login
    expect(requests, times=1).post(token_url, data=login_data, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
        MockJsonResponse(200, {'access_token': access_token1, 'refresh_token': refresh_token1})
    )

    # Prepare logout
    expect(requests, times=1).post(logout_url, data=logout_data, timeout=AUTH_REQUESTS_TIMEOUT).thenReturn(
        MockJsonResponse(404, response_body)
    )

    provider = TokenClient(auth_server_url, auth_realm, auth_username, auth_password)
    assert provider.get_token() == access_token1
    with pytest.raises(ClientAuthenticationError, match='Logout failed'):
        provider.close()

    verifyNoUnwantedInteractions()
    unstub()


def _patch_env(patcher, **patched):
    for key in ['IQM_TOKEN', 'IQM_TOKENS_FILE', 'IQM_AUTH_SERVER', 'IQM_AUTH_USERNAME', 'IQM_AUTH_PASSWORD']:
        if patched.get(key):
            patcher(key, patched[key])
        else:
            patcher(key, '')


def test_token_manager_initialization_with_keyword_args(
    monkeypatch, auth_server_url, auth_username, auth_password
) -> None:
    """Test that TokenManager initializes correct token provider based on keyword arguments"""
    _patch_env(monkeypatch.setenv)

    token_manager = TokenManager()
    assert token_manager._token_provider is None

    token = make_token('Bearer', 300)
    token_manager = TokenManager(token=token)
    assert isinstance(token_manager._token_provider, ExternalToken)
    assert token_manager._token_provider._token == token

    path = '/some/path/to/tokens_file.json'
    token_manager = TokenManager(tokens_file=path)
    assert isinstance(token_manager._token_provider, TokensFileReader)
    assert token_manager._token_provider._path == path

    token_manager = TokenManager(auth_server_url=auth_server_url, username=auth_username, password=auth_password)
    assert isinstance(token_manager._token_provider, TokenClient)
    assert (
        token_manager._token_provider._token_url
        == f'{auth_server_url}/realms/{AUTH_REALM}/protocol/openid-connect/token'
    )
    assert (
        token_manager._token_provider._logout_url
        == f'{auth_server_url}/realms/{AUTH_REALM}/protocol/openid-connect/logout'
    )
    assert token_manager._token_provider._username == auth_username
    assert token_manager._token_provider._password == auth_password

    verifyNoUnwantedInteractions()
    unstub()


def test_token_manager_initialization_with_environment_vars(monkeypatch, auth_server_url, auth_username, auth_password):
    """Test that TokenManager initializes correct token provider based on environment variables"""
    _patch_env(monkeypatch.setenv)

    token_manager = TokenManager()
    assert token_manager._token_provider is None

    token = make_token('Bearer', 300)
    _patch_env(monkeypatch.setenv, **{'IQM_TOKEN': token})
    token_manager = TokenManager()
    assert isinstance(token_manager._token_provider, ExternalToken)
    assert token_manager._token_provider._token == token

    path = '/some/path/to/tokens_file.json'
    _patch_env(monkeypatch.setenv, **{'IQM_TOKENS_FILE': path})
    token_manager = TokenManager()
    assert isinstance(token_manager._token_provider, TokensFileReader)
    assert token_manager._token_provider._path == path

    _patch_env(
        monkeypatch.setenv,
        **{'IQM_AUTH_SERVER': auth_server_url, 'IQM_AUTH_USERNAME': auth_username, 'IQM_AUTH_PASSWORD': auth_password},
    )
    token_manager = TokenManager()
    assert isinstance(token_manager._token_provider, TokenClient)
    assert (
        token_manager._token_provider._token_url
        == f'{auth_server_url}/realms/{AUTH_REALM}/protocol/openid-connect/token'
    )
    assert (
        token_manager._token_provider._logout_url
        == f'{auth_server_url}/realms/{AUTH_REALM}/protocol/openid-connect/logout'
    )
    assert token_manager._token_provider._username == auth_username
    assert token_manager._token_provider._password == auth_password

    verifyNoUnwantedInteractions()
    unstub()


@pytest.mark.parametrize(
    'args,env',
    [
        # Token and some other parameter
        ({'token': 'some-token', 'tokens_file': 'some-path'}, {}),
        # Tokens file and some other parameter
        ({'tokens_file': 'some-path', 'username': 'some-user'}, {}),
        # Some auth server parameter is missing
        ({'auth_server_url': 'some-url', 'password': 'very-secret'}, {}),
        # Token and some other parameter as environment variables
        ({}, {'IQM_TOKEN': 'some-token', 'IQM_AUTH_USERNAME': 'some-path'}),
        # Tokens file and some other parameter
        ({}, {'IQM_TOKENS_FILE': 'some-path', 'IQM_AUTH_PASSWORD': 'very-secret'}),
        # Some auth server parameter is missing
        ({}, {'IQM_AUTH_USERNAME': 'some_user', 'IQM_AUTH_PASSWORD': 'very-secret'}),
    ],
)
def test_token_manager_invalid_combination_of_parameters(args, env, monkeypatch):
    """Test that TokenManager raises ClientConfigurationError if the parameters are ambiguous"""
    _patch_env(monkeypatch.setenv, **env)
    with pytest.raises(ClientConfigurationError, match='Invalid combination of authentication parameters specified'):
        TokenManager(**args)

    verifyNoUnwantedInteractions()
    unstub()


@pytest.mark.parametrize(
    'args,env',
    [
        # Mixed initialisation parameters and environment variables
        ({'tokens_file': 'some-path'}, {'IQM_TOKEN': 'some-token'}),
        # Mixed initialisation parameters and environment variables
        ({'auth_server_url': 'some-url'}, {'IQM_AUTH_USERNAME': 'some-user', 'IQM_AUTH_PASSWORD': 'very-secret'}),
    ],
)
def test_token_manager_mixed_source_of_parameters(args, env, monkeypatch):
    """Test that TokenManager raises ClientConfigurationError if the parameters are ambiguous"""
    _patch_env(monkeypatch.setenv, **env)
    error_message = 'Authentication parameters given both as initialisation args and as environment variables'
    with pytest.raises(ClientConfigurationError, match=error_message):
        TokenManager(**args)

    verifyNoUnwantedInteractions()
    unstub()


def test_token_manager_provides_bearer_token(monkeypatch):
    """Test that TokenManager provides bearer token"""

    _patch_env(monkeypatch.setenv)
    expected_token = make_token('Bearer', 300)
    mock_provider = mock(TokenProviderInterface)
    when(mock_provider).get_token().thenReturn(expected_token)

    token_manager = TokenManager()

    # When authentication is not configured get_bearer_token returns None
    assert token_manager._token_provider is None
    assert token_manager.get_bearer_token() is None

    # An existing valid access token is returned instead of asking token_provider for a new one
    token_manager._token_provider = mock_provider
    existing_token = make_token('Bearer', 300)
    token_manager._access_token = existing_token
    assert token_manager.get_bearer_token() == f'Bearer {existing_token}'

    # Otherwise get_bearer_token returns the token from the token_provider    token_manager._access_token = None
    token_manager._access_token = None
    assert token_manager.get_bearer_token() == f'Bearer {expected_token}'

    verifyNoUnwantedInteractions()
    unstub()


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
    unstub()


def test_submit_circuits_gets_token(
    monkeypatch, base_url, dynamic_architecture_url, jobs_url, sample_dynamic_architecture, sample_circuit
):
    """Test that submit_circuits gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, client = _make_client_with_token(base_url)

    when(requests).get(
        dynamic_architecture_url,
        headers={'User-Agent': client._signature, 'Authorization': f'Bearer {token}'},
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(MockJsonResponse(200, json_data=sample_dynamic_architecture.model_dump()))

    expect(requests, times=1).post(
        jobs_url,
        json=ANY,
        headers={'User-Agent': client._signature, 'Authorization': f'Bearer {token}', 'Expect': '100-Continue'},
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(MockJsonResponse(200, json_data={'id': str(uuid4())}))

    assert isinstance(client.submit_circuits(circuits=[sample_circuit], shots=10), UUID)

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_gets_token(monkeypatch, base_url, jobs_url, ready_job_result):
    """Test that get_run gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, client = _make_client_with_token(base_url)
    job_id = uuid4()

    expect(requests, times=1).get(
        f'{jobs_url}/{str(job_id)}',
        headers={'User-Agent': client._signature, 'Authorization': f'Bearer {token}'},
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(ready_job_result)

    assert isinstance(client.get_run(job_id), RunResult)

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_status_gets_token(monkeypatch, base_url, jobs_url, pending_compilation_status):
    """Test that get_run gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, client = _make_client_with_token(base_url)
    job_id = uuid4()

    expect(requests, times=1).get(
        f'{jobs_url}/{str(job_id)}/status',
        headers={'User-Agent': client._signature, 'Authorization': f'Bearer {token}'},
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(pending_compilation_status)

    assert isinstance(client.get_run_status(job_id), RunStatus)

    verifyNoUnwantedInteractions()
    unstub()


def test_abort_job_gets_token(monkeypatch, base_url, jobs_url):
    """Test that abort_job gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, client = _make_client_with_token(base_url)
    job_id = uuid4()

    expect(requests, times=1).post(
        f'{jobs_url}/{str(job_id)}/abort',
        headers={'User-Agent': client._signature, 'Authorization': f'Bearer {token}'},
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(MockJsonResponse(200, {}))

    client.abort_job(job_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_close_auth_session(monkeypatch, base_url):
    """Test that closing auth session closes TokenManager"""
    _patch_env(monkeypatch.setenv)
    _, client = _make_client_with_token(base_url)
    client._token_manager = mock(TokenManager)
    expect(client._token_manager, times=1).close().thenReturn(True)

    assert client.close_auth_session()

    verifyNoUnwantedInteractions()
    unstub()


def test_close_auth_session_when_client_destroyed(monkeypatch, base_url):
    """Test that deleting client closes TokenManager"""
    _patch_env(monkeypatch.setenv)
    _, client = _make_client_with_token(base_url)
    client._token_manager = mock(TokenManager)
    expect(client._token_manager, times=1).close().thenReturn(True)

    del client

    verifyNoUnwantedInteractions()
    unstub()
