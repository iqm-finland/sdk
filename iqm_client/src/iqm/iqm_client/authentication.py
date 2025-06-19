# Copyright 2024 IQM client developers
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
"""This module contains user authentication related classes and functions required by IQMClient."""

from abc import ABC, abstractmethod
from base64 import b64decode
import json
import os
import time
from typing import Any

from iqm.iqm_client.errors import ClientAuthenticationError, ClientConfigurationError
import requests

AUTH_CLIENT_ID = "iqm_client"
AUTH_REALM = "cortex"
AUTH_REQUESTS_TIMEOUT = float(os.environ.get("IQM_CLIENT_REQUESTS_TIMEOUT", 60.0))
REFRESH_MARGIN_SECONDS = 60


class TokenManager:
    """TokenManager manages the access token required for user authentication.

    Args:
        token: Long-lived IQM token in plain text format
        tokens_file: Path to a tokens file used for authentication
        auth_server_url: Base URL of the authentication server
        username: Username to log in to authentication server
        password: Password to log in to authentication server

    The parameters can also be read from the environment variables IQM_TOKEN, IQM_TOKENS_FILE,
    IQM_AUTH_SERVER, IQM_AUTH_USERNAME, IQM_AUTH_PASSWORD. Environment variables can not be
    mixed with initialisation arguments. All parameters must come from the same source.

    """

    @staticmethod
    def time_left_seconds(token: Any) -> int:
        """Check how much time is left until the token expires.

        Returns:
            Time left on token in seconds.

        """
        if not token or not isinstance(token, str):
            return 0
        parts = token.split(".", 2)
        if len(parts) != 3:
            return 0
        # Add padding to adjust body length to a multiple of 4 chars as required by base64 decoding
        try:
            body = parts[1] + ("=" * (-len(parts[1]) % 4))
            exp_time = int(json.loads(b64decode(body)).get("exp", "0"))
            return max(0, exp_time - int(time.time()))
        except (UnicodeDecodeError, json.decoder.JSONDecodeError, ValueError, TypeError):
            return 0

    def __init__(
        self,
        token: str | None = None,
        tokens_file: str | None = None,
        auth_server_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        def _format_names(variable_names: list[str]) -> str:
            """Format a list of variable names"""
            return ", ".join(f'"{name}"' for name in variable_names)

        auth_parameters: dict[str, str] = {}

        init_parameters = {
            "token": token,
            "tokens_file": tokens_file,
            "auth_server_url": auth_server_url,
            "username": username,
            "password": password,
        }
        init_params_given = [key for key, value in init_parameters.items() if value]

        env_variables = {
            "token": "IQM_TOKEN",
            "tokens_file": "IQM_TOKENS_FILE",
            "auth_server_url": "IQM_AUTH_SERVER",
            "username": "IQM_AUTH_USERNAME",
            "password": "IQM_AUTH_PASSWORD",
        }
        env_vars_given = [name for name in env_variables.values() if os.environ.get(name)]

        if init_params_given and env_vars_given:
            raise ClientConfigurationError(
                "Authentication parameters given both as initialisation args and as environment variables: "
                + f"initialisation args {_format_names(init_params_given)}, "
                + f"environment variables {_format_names(env_vars_given)}."
                + " Parameter sources must not be mixed."
            )

        if env_vars_given:
            auth_parameters = {key: value for key, name in env_variables.items() if (value := os.environ.get(name))}
        else:
            auth_parameters = {key: str(value) for key, value in init_parameters.items() if value}

        self._token_provider: TokenProviderInterface | None = None
        self._access_token: str | None = None

        if not auth_parameters:
            self._token_provider = None
        elif set(auth_parameters) == {"token"}:
            # This is not necessarily a JWT token
            self._token_provider = ExternalToken(auth_parameters["token"])
        elif set(auth_parameters) == {"tokens_file"}:
            self._token_provider = TokensFileReader(auth_parameters["tokens_file"])
        elif set(auth_parameters) == {"auth_server_url", "username", "password"}:
            self._token_provider = TokenClient(
                auth_parameters["auth_server_url"],
                AUTH_REALM,
                auth_parameters["username"],
                auth_parameters["password"],
            )
        else:
            raise ClientConfigurationError(
                f"""Invalid combination of authentication parameters specified: {list(auth_parameters)},
                Use either ['token'] or ['username', 'password'].""",
            )

    def get_bearer_token(self, retries: int = 1) -> str | None:
        """Returns a valid bearer token, or None if no user authentication has been configured.

        Raises:
            ClientAuthenticationError: getting the token failed

        """
        if self._token_provider is None:
            return None  # Authentication is not used

        # Use the existing access token if it is still valid
        if TokenManager.time_left_seconds(self._access_token) > REFRESH_MARGIN_SECONDS:
            return f"Bearer {self._access_token}"

        # Otherwise, get a new access token from token provider
        try:
            self._access_token = self._token_provider.get_token()
            return f"Bearer {self._access_token}"
        except ClientAuthenticationError:
            if retries < 1:
                raise

        # Try again
        return self.get_bearer_token(retries - 1)

    def close(self) -> bool:
        """Close the configured token provider.

        Returns:
            True if closing was successful

        Raises:
            ClientAuthenticationError: closing failed

        """
        if self._token_provider is None:
            return False

        self._token_provider.close()
        self._token_provider = None
        return True


class TokenProviderInterface(ABC):
    """Interface to token provider"""

    @abstractmethod
    def get_token(self) -> str:
        """Returns a valid access token.

        Raises:
            ClientAuthenticationError: acquiring the token failed

        """

    @abstractmethod
    def close(self) -> None:
        """Closes authentication session.

        Raises:
            ClientAuthenticationError: closing the session failed

        """


class ExternalToken(TokenProviderInterface):
    """Holds an external token"""

    def __init__(self, token: str):
        self._token: str | None = token

    def get_token(self) -> str:
        if self._token is None:
            raise ClientAuthenticationError("No external token available")
        return self._token

    def close(self) -> None:
        self._token = None
        raise ClientAuthenticationError("Can not close externally managed auth session")


class TokensFileReader(TokenProviderInterface):
    """Reads token from a file"""

    def __init__(self, tokens_file: str):
        self._path: str | None = tokens_file

    def get_token(self) -> str:
        try:
            if self._path is None:
                raise ClientAuthenticationError("No tokens file available")
            with open(self._path, encoding="utf-8") as file:
                raw_data = file.read()
            json_data = json.loads(raw_data)
            token = json_data.get("access_token")
            if TokenManager.time_left_seconds(token) <= 0:
                raise ClientAuthenticationError("Access token in file has expired or is not valid")
        except (FileNotFoundError, IsADirectoryError, json.decoder.JSONDecodeError) as e:
            raise ClientAuthenticationError(rf"Failed to read access token from file '{self._path}': {e}") from e
        return token

    def close(self) -> None:
        self._path = None
        raise ClientAuthenticationError("Can not close externally managed auth session")


class TokenClient(TokenProviderInterface):
    """Requests new token from an authentication server"""

    PASSWORD_GRANT_TYPE = "password"
    REFRESH_TOKEN_GRANT_TYPE = "refresh_token"

    def __init__(self, auth_server_url: str, realm: str, username: str, password: str):
        """Initialize the client"""
        self._token_url = f"{auth_server_url}/realms/{realm}/protocol/openid-connect/token"
        self._logout_url = f"{auth_server_url}/realms/{realm}/protocol/openid-connect/logout"
        self._username = username
        self._password = password
        self._refresh_token: str | None = None

    def _get_access_token_from_server(self, grant_type: str) -> str | None:
        """Get new access token from the server and update refresh token."""
        if grant_type == TokenClient.REFRESH_TOKEN_GRANT_TYPE:
            # Update tokens using existing refresh_token
            data = {
                "client_id": AUTH_CLIENT_ID,
                "grant_type": TokenClient.REFRESH_TOKEN_GRANT_TYPE,
                "refresh_token": str(self._refresh_token),
            }
        else:
            # There is no valid refresh token or refresh token has expired, start a new session
            data = {
                "client_id": AUTH_CLIENT_ID,
                "grant_type": TokenClient.PASSWORD_GRANT_TYPE,
                "username": self._username,
                "password": self._password,
            }

        # Request new tokens from the server
        access_token: str | None = None
        result = requests.post(self._token_url, data=data, timeout=AUTH_REQUESTS_TIMEOUT)
        if result.status_code == 200:
            tokens = result.json()
            self._refresh_token = tokens.get("refresh_token")
            if TokenManager.time_left_seconds(self._refresh_token) <= 0:
                self._refresh_token = None
            access_token = tokens.get("access_token")
            if TokenManager.time_left_seconds(access_token) <= 0:
                access_token = None
        return access_token

    def get_token(self) -> str:
        """Get new access token and refresh token from the server"""
        if not self._token_url:
            raise ClientConfigurationError("No auth server configured")

        access_token: str | None = None
        if TokenManager.time_left_seconds(self._refresh_token) > REFRESH_MARGIN_SECONDS:
            # There is a valid refresh token, try to update tokens using it
            access_token = self._get_access_token_from_server(TokenClient.REFRESH_TOKEN_GRANT_TYPE)
        if access_token is None:
            # Failed to get valid access token using refresh token, start a new session
            access_token = self._get_access_token_from_server(TokenClient.PASSWORD_GRANT_TYPE)
        if access_token is None:
            # Failed to get valid access token using username and password, raise an error
            raise ClientAuthenticationError("Getting access token from auth server failed")
        return str(access_token)  # access token can not be None here

    def close(self) -> None:
        """Close authentication session"""
        if not self._refresh_token:
            raise ClientAuthenticationError("No auth session active")

        data = {"client_id": AUTH_CLIENT_ID, "refresh_token": self._refresh_token}
        self._refresh_token = None

        result = requests.post(self._logout_url, data=data, timeout=AUTH_REQUESTS_TIMEOUT)
        if result.status_code not in [200, 204]:
            raise ClientAuthenticationError(f"Logout failed, {result.text}")
