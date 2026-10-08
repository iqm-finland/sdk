# Copyright 2026 IQM
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
"""Client implementation for Station Control service REST API."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from http import HTTPStatus
import importlib
from importlib.metadata import version
import json
import logging
import os
import platform
from typing import Any, TypeVar
from urllib.parse import urlparse
import warnings

from opentelemetry import propagate, trace
from packaging.version import Version, parse
from pydantic import BaseModel
import requests

from exa.common.errors.iqm_error import (
    InternalSystemError,
    InvalidOperationError,
)
from exa.common.qcm_data.qcm_data_client import QCMDataClient
from iqm.station_control.client.authentication import TokenManager
from iqm.station_control.client.list_models import (
    ListModel,
    ListWithMetaResponse,
)
from iqm.station_control.interface.errors import map_from_status_code_to_error
from iqm.station_control.interface.list_with_meta import ListWithMeta, Meta
from iqm.station_control.interface.pydantic_base import PydanticBase
from iqm.station_control.interface.serializable import Serializable

logger = logging.getLogger(__name__)
TypePydanticBase = TypeVar("TypePydanticBase", bound=PydanticBase)


class StationControlClient(Serializable):
    """Client implementation for station control service REST API.

    Args:
        station_control_url: Remote station control service URL.
        token: Long-lived authentication token in plain text format.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        get_token_callback: Callback function that returns an Authorization header containing a bearer token.
            If ``get_token_callback`` is given no other user authentication parameters should be given.
        client_signature: String that is added to the User-Agent header of requests
            sent to the server.

    """

    def __init__(
        self,
        station_control_url: str,
        *,
        token: str | None = None,
        tokens_file: str | None = None,
        get_token_callback: Callable[[], str] | None = None,
        client_signature: str | None = None,
    ):
        self.root_url = station_control_url

        # Enforce HTTPS on remote hosts
        parsed_url = urlparse(station_control_url)
        if parsed_url.scheme == "http":
            if parsed_url.hostname in ("localhost", "127.0.0.1"):
                # We purposefully allow HTTP without warnings for localhost/127.0.0.1.
                # This prevents console bloat for developers running local mock servers,
                # simulators, or containers, where traffic never leaves the machine.
                pass
            else:
                warnings.warn(
                    f"Insecure HTTP requests to remote hosts are heavily discouraged: '{station_control_url}'. "
                    "Please update your client configuration to use 'https://'.",
                    category=UserWarning,
                    stacklevel=2,
                )

        self._token = token
        self._tokens_file = tokens_file
        self._get_token_callback = get_token_callback
        tm = TokenManager(token, tokens_file, get_token_callback, use_env_vars=False)
        self._token_manager = tm
        self._auth_header_callback = tm.get_auth_header_callback()

        self._client_signature = client_signature
        self._signature = self._create_signature(client_signature)
        self._enable_opentelemetry = os.environ.get("JAEGER_OPENTELEMETRY_COLLECTOR_ENDPOINT", None) is not None
        # TODO SW-1387: Remove this when using v1 API, not needed
        self._check_api_versions()
        qcm_data_url = os.environ.get("CHIP_DESIGN_RECORD_FALLBACK_URL", None)
        self._qcm_data_client = QCMDataClient(qcm_data_url) if qcm_data_url else None

        # Timeout configurations (in seconds)
        self.timeout_secs: float = 10800.0  # 3 hours for absolute maximum polling time

    def serialize(self) -> dict[str, Any]:
        return {
            "station_control_url": self.root_url,
            "token": self._token,
            "tokens_file": self._tokens_file,
            "get_token_callback": self._get_token_callback
            if self._get_token_callback is None
            else f"{self._get_token_callback.__module__}.{self._get_token_callback.__qualname__}",
            "client_signature": self._client_signature,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> StationControlClient:
        if data["get_token_callback"] is None:
            get_token_callback = None
        else:
            mod, name = data["get_token_callback"].rsplit(".", 1)
            get_token_callback = getattr(importlib.import_module(mod), name)

        return cls(
            station_control_url=data["station_control_url"],
            token=data["token"],
            tokens_file=data["tokens_file"],
            get_token_callback=get_token_callback,
            client_signature=data["client_signature"],
        )

    @property
    def version(self) -> str:
        """Version of the Station Control API this client is using."""
        return "v1"

    @cache
    def get_about(self) -> dict:
        response = self.send_request(requests.get, "about")
        return response.json()

    def _check_api_versions(self):  # noqa: ANN202
        client_api_version = self._get_client_api_version()
        # Parse versions using standard packaging.version implementation.
        # For that purpose, we need to convert our custom " (local editable)" to follow packaging.version syntax.
        about = self.get_about()
        server_api_version = parse(
            about["software_versions"]["iqm-station-control-client"].replace(" (local editable)", "+local")
        )

        if client_api_version.local or server_api_version.local:
            logger.warning(
                "Client ('%s') and/or server ('%s') is using a local version of the station-control-client. "
                "Client and server compatibility cannot be guaranteed.",
                client_api_version,
                server_api_version,
            )
        elif client_api_version.major != server_api_version.major:
            logger.warning(
                "Client and server compatibility cannot be guaranteed. "
                "The installed station-control-client package version '%s' has a different major to "
                "station-control server version '%s'.",
                client_api_version,
                server_api_version,
            )
        elif client_api_version.minor > server_api_version.minor:
            logger.warning(
                "station-control-client version '%s' is newer minor version than '%s' used by the station control "
                "server, some new client features might not be supported.",
                client_api_version,
                server_api_version,
            )

    # TODO SW-1387: Remove this when using v1 API, not needed
    @staticmethod
    def _get_client_api_version() -> Version:
        return parse(version("iqm-station-control-client"))

    @staticmethod
    def deserialize_response(
        response: requests.Response,
        model_class: type[TypePydanticBase | ListModel],
        *,
        list_with_meta: bool = False,
    ) -> TypePydanticBase | ListWithMeta:
        # Use "model_validate_json(response.text)" instead of "model_validate(response.json())".
        # This validates the provided data as a JSON string or bytes object.
        # If your incoming data is a JSON payload, this is generally considered faster.
        if list_with_meta:
            list_with_meta_response: ListWithMetaResponse = ListWithMetaResponse.model_validate_json(response.text)
            meta = list_with_meta_response.meta or Meta()
            if meta and meta.errors:
                logger.warning("Errors in station control response:\n  - %s", "\n  - ".join(meta.errors))
            return ListWithMeta(model_class.model_validate(list_with_meta_response.items), meta=meta)
        model = model_class.model_validate_json(response.text)
        if isinstance(model, ListModel):
            return model.root
        return model

    @classmethod
    def _create_signature(cls, client_signature: str | None) -> str:
        signature = f"{platform.platform(terse=True)}"
        signature += f", python {platform.python_version()}"
        dist_pkg_name = "iqm-station-control-client"
        signature += f", {cls.__name__} {dist_pkg_name} {version(dist_pkg_name)}"
        if client_signature:
            signature += f", {client_signature}"
        return signature

    @staticmethod
    def clean_query_parameters(model: Any, **kwargs) -> dict[str, Any]:
        if issubclass(model, PydanticBase) and "invalid" in model.model_fields and "invalid" not in kwargs:
            # Get only valid items by default, "invalid=None" would return also invalid ones.
            # This default has to be set on the client side, server side uses default "None".
            kwargs["invalid"] = False
        return _remove_empty_values(kwargs)

    def serialize_query_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serialize query parameters, skipping None values and empty dictionaries."""
        return {key: self._serialize_query_param(value) for key, value in params.items() if value not in [None, {}]}

    @staticmethod
    def _serialize_query_param(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def send_request(
        self,
        http_method: Callable[..., requests.Response],
        url_path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: str | None = None,
        octets: bytes | None = None,
        timeout: int = 600,
    ) -> requests.Response:
        """Send an HTTP request.

        Parameters ``json_data`` and ``octets`` are mutually exclusive.
        The first non-None argument (in this order) will be used to construct the body of the request.

        Args:
            http_method: HTTP method to use for the request, any of requests.[post|get|put|head|delete|patch|options].
            url_path: URL for the request.
            headers: Additional HTTP headers for the request. Some may be overridden.
            params: HTTP query parameters to store in the query string of the request URL.
            json_data: JSON string to store in the body, may contain arbitrary Unicode characters.
            octets: Pre-serialized binary data to store in the body.
            timeout: Timeout for the request in seconds.

        Returns:
            Response to the request.

        Raises:
            IQMError: Request was not successful.

        """
        # Will raise an error if respectively an error response code is returned.
        # http_method should be any of requests.[post|get|put|head|delete|patch|options]

        is_mutating = http_method.__name__ in ("post", "put", "patch", "delete")
        request_kwargs = self._build_request_kwargs(
            is_mutating=is_mutating,
            headers=headers or {},
            params=params or {},
            json_data=json_data,
            octets=octets,
            timeout=timeout,
        )
        url = f"{self.root_url}/{url_path}"
        # TODO SW-1387: Use v1 API
        # url = f"{self.root_url}/{self.version}/{url_path}"
        response = http_method(url, **request_kwargs)

        if is_mutating and response.status_code in (
            HTTPStatus.MOVED_PERMANENTLY,
            HTTPStatus.FOUND,
            HTTPStatus.TEMPORARY_REDIRECT,
            HTTPStatus.PERMANENT_REDIRECT,
        ):
            new_location = response.headers.get("Location")
            method = http_method.__name__.upper()
            raise InvalidOperationError(
                f"The server requested a redirect to '{new_location}' during a {method} request. "
                "To prevent authentication drops or data loss, automatic redirects on state-changing requests"
                "are disabled. Please update your client's root URL to the redirected secure location."
            )

        if not response.ok:
            try:
                response_json = response.json()
                error_message = response_json.get("message")
            except (json.JSONDecodeError, KeyError):
                error_message = response.text

            try:
                error_class = map_from_status_code_to_error(response.status_code)  # type: ignore[arg-type]
            except KeyError:
                raise InternalSystemError(f"Unexpected response status code {response.status_code}: {error_message}")

            raise error_class(error_message)
        return response

    @staticmethod
    def serialize_model(model: BaseModel) -> str:
        """Serialize a Pydantic model into a JSON string.

        All Pydantic models should be serialized using this method, to keep the client behavior uniform.

        Args:
            model: Pydantic model to JSON-serialize.

        Returns:
            Corresponding JSON string, may contain arbitrary Unicode characters.

        """
        # Strings in model can contain non-latin-1 characters. Unlike json.dumps which encodes non-latin-1 chars
        # using the \uXXXX syntax, BaseModel.model_dump_json() keeps them in the produced JSON str.
        return model.model_dump_json()

    def _build_request_kwargs(
        self,
        *,
        is_mutating: bool,
        headers: dict[str, str],
        params: dict[str, Any],
        json_data: str | None = None,
        octets: bytes | None = None,
        timeout: int,
    ) -> dict[str, Any]:
        """Prepare the keyword arguments for an HTTP request."""
        if json_data is not None and octets is not None:
            raise ValueError("json_data and octets are mutually exclusive")
        # Add default headers
        headers["User-Agent"] = self._signature

        # json_data and octets are mutually exclusive
        data: bytes | None = None
        if json_data is not None:
            # Must be able to handle JSON strings with arbitrary unicode characters, so we use an explicit
            # encoding into bytes, and set the headers so the recipient can decode the request body correctly.
            data = json_data.encode("utf-8")
            headers["Content-Type"] = "application/json; charset=UTF-8"
        elif octets is not None:
            data = octets
            headers["Content-Type"] = "application/protobuf"

        if self._enable_opentelemetry:
            parent_span_context = trace.set_span_in_context(trace.get_current_span())
            propagate.inject(carrier=headers, context=parent_span_context)

        # If auth header callback exists, use it to add the header
        if self._auth_header_callback:
            headers["Authorization"] = self._auth_header_callback()

        # Strip Authorization header on insecure remote HTTP calls
        parsed_url = urlparse(self.root_url)
        is_insecure_remote = parsed_url.scheme == "http" and parsed_url.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",  # Handle IPv6 localhost
        )

        if is_insecure_remote and "Authorization" in headers:
            allow_insecure_auth = os.getenv("INSECURE_ALLOW_NON_HTTPS_AUTH", "false").lower() == "true"
            if not allow_insecure_auth:
                del headers["Authorization"]
                warnings.warn(
                    "Stripped the 'Authorization' header because you are connecting to a remote host "
                    "over unencrypted HTTP. To use authentication safely, please switch your root_url to HTTPS. "
                    "To override this, set INSECURE_ALLOW_NON_HTTPS_AUTH=true.",
                    category=UserWarning,
                    stacklevel=2,
                )

        kwargs = {
            "headers": headers,
            "params": params,
            "data": data,
            "timeout": timeout,
        }

        # Prevent dangerous silent redirects on mutating methods
        if is_mutating:
            kwargs["allow_redirects"] = False

        return _remove_empty_values(kwargs)


def _remove_empty_values(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the given dict without values that are None or {}."""
    return {key: value for key, value in kwargs.items() if value not in [None, {}]}
