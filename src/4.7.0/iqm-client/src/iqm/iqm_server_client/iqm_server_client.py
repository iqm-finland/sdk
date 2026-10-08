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
"""Shared HTTP client implementation for remote IQM Server communication.

This module provides the :class:`IQMServerClient`, which manages the underlying network
transport, session, and authentication. It is composed inside both IQMServerExecutor
and IQMServerStorage to prevent maintaining duplicate connections to the same server.
"""

from collections.abc import Callable
from http import HTTPStatus
from importlib.metadata import distributions, version
import json
import logging
import os
import platform
from typing import Any, TypeVar, cast
from urllib.parse import urlparse
import warnings

from iqm.iqm_server_client.models import ListQuantumComputersResponse, QuantumComputer
from opentelemetry import propagate, trace
from pydantic import BaseModel, TypeAdapter
import requests

from exa.common.errors.iqm_error import InvalidOperationError, IQMError, OperationTimeoutError, ValidationError
from exa.common.qcm_data.qcm_data_client import QCMDataClient
from iqm.station_control.client.authentication import ClientConfigurationError, TokenManager
from iqm.station_control.client.list_models import ListModel, ListWithMetaResponse
from iqm.station_control.interface.errors import IQMServerError, map_from_status_code_to_error
from iqm.station_control.interface.list_with_meta import ListWithMeta, Meta
from iqm.station_control.interface.models import CircuitMeasurementCounts, CircuitMeasurementResults
from iqm.station_control.interface.pydantic_base import PydanticBase
from iqm.station_control.interface.serializable import Serializable

logger = logging.getLogger(__name__)

TypePydanticBase = TypeVar("TypePydanticBase", bound=PydanticBase)
CircuitMeasurementResultsBatchAdapter = TypeAdapter(list[CircuitMeasurementResults])
CircuitCountsBatchAdapter = TypeAdapter(list[CircuitMeasurementCounts])

REQUESTS_TIMEOUT = float(os.environ.get("IQM_CLIENT_REQUESTS_TIMEOUT", "120"))
DEFAULT_TIMEOUT_SECONDS: float = 10800.0  # 3 hours


class IQMServerClient(Serializable):
    """Shared functionality for clients communicating with the same IQM server.

    Args:
        iqm_server_url: Remote IQM Server URL to connect to.
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one.
        token: Long-lived IQM token in plain text format.
        tokens_file: Path to a tokens file used for authentication.
        client_signature: String that is added to the User-Agent header of requests
            sent to the server.
        enable_opentelemetry: Iff True, enable Jaeger/OpenTelemetry tracing.
        timeout: Timeout for the request in seconds.

    """

    def __init__(
        self,
        iqm_server_url: str | None = None,
        *,
        quantum_computer: str | None = None,
        token: str | None = None,
        tokens_file: str | None = None,
        client_signature: str | None = None,
        enable_opentelemetry: bool = False,
        timeout: float = REQUESTS_TIMEOUT,
    ):
        self._iqm_server_url = iqm_server_url
        self._token = token
        self._tokens_file = tokens_file
        self._client_signature = client_signature

        iqm_server_url, quantum_computer = self._resolve_iqm_parameters(iqm_server_url, quantum_computer)
        root_url, quantum_computer = self._normalize_url(iqm_server_url, quantum_computer)
        self.root_url = root_url

        tm = TokenManager(token, tokens_file)
        self._token_manager = tm
        self._auth_header_callback = tm.get_auth_header_callback()

        self._signature = self._create_signature(client_signature)

        self._enable_opentelemetry = enable_opentelemetry
        self._timeout = timeout
        # Resolving the quantum computer below sends a request via send_request, which reads
        # self._proxy_path: it must already exist (as None, a non-station-proxied request) at that point.
        self._proxy_path: str | None = None
        resolved_quantum_computer, quantum_computer_count = self._resolve_quantum_computer(quantum_computer)
        self._quantum_computer = resolved_quantum_computer.alias
        self._proxy_path = self._resolve_proxy_path(resolved_quantum_computer, quantum_computer_count)
        # TODO remove: temporary QCM data client until iqm-server can serve CHEDDARs for multiple stations
        qcm_data_url = os.environ.get("CHIP_DESIGN_RECORD_FALLBACK_URL", None)
        self._qcm_data_client = QCMDataClient(qcm_data_url) if qcm_data_url else None

    def serialize(self) -> dict[str, Any]:  # noqa: D102
        return {
            "iqm_server_url": self._iqm_server_url,
            "quantum_computer": self._quantum_computer,
            "token": self._token,
            "tokens_file": self._tokens_file,
            "client_signature": self._client_signature,
            "enable_opentelemetry": self._enable_opentelemetry,
            "timeout": self._timeout,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "IQMServerClient":  # noqa: D102
        return cls(
            iqm_server_url=data["iqm_server_url"],
            quantum_computer=data["quantum_computer"],
            token=data["token"],
            tokens_file=data["tokens_file"],
            client_signature=data["client_signature"],
            enable_opentelemetry=data["enable_opentelemetry"],
            timeout=data["timeout"],
        )

    @property
    def api_version(self) -> str:
        """API version of the IQM Server API this client is using."""
        return "v1"

    @property
    def quantum_computer(self) -> str:
        """Human-readable alias of the quantum computer this client connects to."""
        return self._quantum_computer

    @property
    def proxy_path(self) -> str:
        """Resolved URL path prefix for station-proxied requests to this quantum computer.

        Raises:
            ClientConfigurationError: Station-proxied requests are not available for this quantum
                computer, e.g. due to missing permissions, or the station proxy feature not being
                enabled on the server.

        """
        if self._proxy_path is None:
            raise ClientConfigurationError(
                f"No station proxy path is available for quantum computer '{self._quantum_computer}'. "
                "This is usually caused by missing permissions, or the station proxy feature not "
                "being enabled for this quantum computer on the server."
            )
        return self._proxy_path

    def get_about(self) -> dict:
        """Get information about IQM Server."""
        about = self.send_request(requests.get, "about").json()
        about_station = self.send_request(
            requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/about"
        ).json()
        about["about_station"] = {
            "version": about_station["version"],
            "software_versions": {
                key: value for key, value in about_station["software_versions"].items() if key.startswith("iqm-")
            },
        }
        return about

    def get_health(self) -> dict[str, Any]:
        """Get the status of the IQM Server."""
        response = self.send_request(requests.get, f"quantum-computers/{self._quantum_computer}/health")
        return response.json()

    def _get_quantum_computers(self) -> list[QuantumComputer]:
        """Get the quantum computers of the server."""
        response = self.send_request(requests.get, "quantum-computers")

        # cast tells Mypy: "Trust me, this is a ListQuantumComputersResponse"
        model = cast(ListQuantumComputersResponse, self.deserialize_response(response, ListQuantumComputersResponse))
        return model.quantum_computers

    def _debug_info(self) -> dict[str, Any]:
        """Information about the client and server versions, and the platform.

        .. note:: The output of this method is for internal use only, and may change without notice.
        """

        def mask_env_var(name: str) -> int | None:
            """Mask the sensitive information in an env var, returning just its length."""
            temp = os.environ.get(name)
            return None if temp is None else len(temp)

        # locally installed packages
        local_dist_pkgs = distributions()

        def pkg_filter(name: str) -> bool:
            """Filter out the packages we are interested in."""
            return name.startswith("iqm-") or name in {"cirq", "qiskit", "qiskit-aer", "qrisp"}

        info: dict[str, Any] = {
            "platform.platform": platform.platform(),
            "platform.version": platform.version(),
            "platform.python_version": platform.python_version(),
            "root_url": self.root_url,
            "quantum_computer": self._quantum_computer,
            "len(IQM_TOKEN)": mask_env_var("IQM_TOKEN"),
            "len(IQM_TOKENS_FILE)": mask_env_var("IQM_TOKENS_FILE"),
            "token_provider": type(self._token_manager._token_provider),
            "auth_header_callback": self._token_manager._auth_header_callback,
            "local packages": {dist.name: dist.version for dist in local_dist_pkgs if pkg_filter(dist.name)},
        }
        try:
            info["about"] = self.get_about()
        except IQMError as exc:
            info["server error"] = str(exc)
        return info

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

    def send_request(
        self,
        http_method: Callable[..., requests.Response],
        url_path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: str | None = None,
        protobuf_data: bytes | None = None,
    ) -> requests.Response:
        """Send an HTTP request.

        Parameters ``json_str`` and ``octets`` are mutually exclusive.
        The first non-None argument (in this order) will be used to construct the body of the request.

        Args:
            http_method: HTTP method to use for the request, any of requests.[post|get|put|head|delete|patch|options].
            url_path: URL for the request.
            headers: Additional HTTP headers for the request. Some may be overridden.
            params: HTTP query parameters to store in the query string of the request URL.
            json_data: JSON string to store in the body, may contain arbitrary Unicode characters.
            protobuf_data: Pre-serialized protobuf binary data to store in the body.

        Returns:
            Response to the request.

        Raises:
            IQMError: Request was not successful.

        """
        method_name = http_method.__name__.upper()
        is_mutating = method_name in ("POST", "PUT", "PATCH", "DELETE")
        request_kwargs = self.prepare_request_kwargs(
            is_mutating=is_mutating,
            headers=headers or {},
            params=params or {},
            json_data=json_data,
            protobuf_data=protobuf_data,
            timeout=self._timeout,
        )

        # This is a temporary path resolution to enable calling the direct station-control proxy for
        # some calls, and the native IQM server API for others. This selection can be cleaned up once
        # The proxy endpoints are no longer in use.
        is_station_proxy_request = self._proxy_path is not None and url_path.startswith(self._proxy_path)
        api_path = "" if is_station_proxy_request else f"api/{self.api_version}/"
        url = f"{self.root_url}/{api_path}{url_path}"

        try:
            response = http_method(url, **request_kwargs)
        except requests.exceptions.Timeout as exc:
            # This handles the client-side 120s timer running out
            raise OperationTimeoutError(f"Client-side timeout reached ({self._timeout}s).") from exc

        # Handle 3xx Status Codes Explicitly for blocked redirects
        if is_mutating and response.status_code in (
            HTTPStatus.MOVED_PERMANENTLY,
            HTTPStatus.FOUND,
            HTTPStatus.TEMPORARY_REDIRECT,
            HTTPStatus.PERMANENT_REDIRECT,
        ):
            new_location = response.headers.get("Location")
            raise InvalidOperationError(
                f"The server requested a redirect to '{new_location}' during a {method_name} request. "
                "Automatic redirects on state-changing requests are disabled."
            )

        if not response.ok:
            try:
                iqm_server_error = IQMServerError.model_validate(response.json())
                iqm_server_error.raise_exception(response.status_code)
            except (json.JSONDecodeError, ValidationError):
                # Fallback for non-JSON responses or malformed error models
                error_class = map_from_status_code_to_error(response.status_code)
                raise error_class(response.text)

        return response

    @staticmethod
    def clean_query_parameters(model: Any, **kwargs) -> dict[str, Any]:
        """Sanitize query parameters, defaulting 'invalid' to False if supported by the model."""
        if issubclass(model, PydanticBase) and "invalid" in model.model_fields and "invalid" not in kwargs:
            # Get only valid items by default, "invalid=None" would return also invalid ones.
            # This default has to be set on the client side, server side uses default "None".
            kwargs["invalid"] = False
        return IQMServerClient._remove_empty_values(kwargs)

    def prepare_request_kwargs(
        self,
        *,
        is_mutating: bool,
        headers: dict[str, str],
        params: dict[str, Any],
        json_data: str | None = None,
        protobuf_data: bytes | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        """Return the keyword arguments for a :mod:`requests` HTTP method."""
        if json_data is not None and protobuf_data is not None:
            raise ValidationError("json_data and protobuf_data are mutually exclusive")

        # Add default headers
        _headers = self._default_headers()
        _headers.update(headers)

        # Strip Authorization header on insecure remote HTTP calls
        parsed_url = urlparse(self.root_url)
        is_insecure_remote = parsed_url.scheme == "http" and parsed_url.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",  # Handle IPv6 localhost
        )

        if is_insecure_remote and "Authorization" in _headers:
            allow_insecure_auth = os.getenv("INSECURE_ALLOW_NON_HTTPS_AUTH", "false").lower() == "true"
            if not allow_insecure_auth:
                del _headers["Authorization"]
                warnings.warn(
                    "Stripped the 'Authorization' header because you are connecting to a remote host "
                    "over unencrypted HTTP. To use authentication safely, please switch your root_url to HTTPS. "
                    "To override this, set INSECURE_ALLOW_NON_HTTPS_AUTH=true.",
                    category=UserWarning,
                    stacklevel=2,
                )

        # "json_str" and "protobuf_data" are mutually exclusive
        data: bytes | None = None
        if json_data is not None:
            # Must be able to handle JSON strings with arbitrary Unicode characters,
            # so we use an explicit encoding into bytes,
            # and set the headers so the recipient can decode the request body correctly.
            data = json_data.encode("utf-8")
            _headers["Content-Type"] = "application/json; charset=UTF-8"
        elif protobuf_data is not None:
            data = protobuf_data
            _headers["Content-Type"] = "application/protobuf"

        if "Accept" in headers:
            _headers["Accept"] = headers["Accept"]

        if self._enable_opentelemetry:
            parent_span_context = trace.set_span_in_context(trace.get_current_span())
            propagate.inject(carrier=_headers, context=parent_span_context)

        kwargs = {
            "headers": _headers,
            "params": params,
            "data": data,
            "timeout": timeout,
        }

        # Prevent dangerous silent redirects on mutating methods
        if is_mutating:
            kwargs["allow_redirects"] = False

        return self._remove_empty_values(kwargs)

    @staticmethod
    def deserialize_response(
        response: requests.Response,
        model_class: type[TypePydanticBase | ListModel],
        *,
        list_with_meta: bool = False,
    ) -> TypePydanticBase | ListWithMeta:
        """Parse HTTP response text directly into Pydantic models.

        Logs metadata errors as warnings if list_with_meta is enabled.
        """
        # Use "model_validate_json(response.text)" instead of "model_validate(response.json())".
        # This validates the provided data as a JSON string or bytes object.
        # If your incoming data is a JSON payload, this is generally considered faster.
        if list_with_meta:
            list_with_meta_response: ListWithMetaResponse = ListWithMetaResponse.model_validate_json(response.text)
            meta = list_with_meta_response.meta or Meta()
            if meta and meta.errors:
                logger.warning("Errors in response:\n  - %s", "\n  - ".join(meta.errors))
            return ListWithMeta(model_class.model_validate(list_with_meta_response.items), meta=meta)
        model = model_class.model_validate_json(response.text)
        if isinstance(model, ListModel):
            return model.root
        return model

    def serialize_query_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serialize query parameters, skipping None values and empty dictionaries."""
        return {key: self._serialize_query_param(value) for key, value in params.items() if value not in [None, {}]}

    @staticmethod
    def _resolve_iqm_parameters(
        iqm_server_url: str | None = None, quantum_computer: str | None = None
    ) -> tuple[str, str | None]:
        """Resolves URL and QC, prioritizing explicit arguments over environment variables."""
        init_parameters = {
            "iqm_server_url": iqm_server_url,
            "quantum_computer": quantum_computer,
        }
        env_variables = {
            "iqm_server_url": "IQM_SERVER_URL",
            "quantum_computer": "IQM_QUANTUM_COMPUTER",
        }

        for param_name, env_var in env_variables.items():
            # Only fall back to the environment variable if the explicit argument is None
            if init_parameters[param_name] is None and (env_var_value := os.environ.get(env_var)) is not None:
                init_parameters[param_name] = env_var_value

        if init_parameters["iqm_server_url"] is None:
            raise ValueError("IQM Server URL must be provided.")

        return init_parameters["iqm_server_url"], init_parameters["quantum_computer"]

    @staticmethod
    def _normalize_url(iqm_server_url: str, quantum_computer: str | None) -> tuple[str, str | None]:
        """Validate the connection details, provide some backwards compatibility."""
        # Security measure: mitigate UTF-8 read order control character
        # exploits by allowing only ASCII urls
        if not iqm_server_url.isascii():
            raise ClientConfigurationError(f"Non-ASCII characters in URL: {iqm_server_url}")
        try:
            url = urlparse(iqm_server_url)
        except Exception as e:
            raise ClientConfigurationError(f"Invalid URL: {iqm_server_url}") from e

        if url.scheme not in {"http", "https"}:
            raise ClientConfigurationError(
                f"The URL schema has to be http or https. Incorrect schema in URL: {iqm_server_url}"
            )

        if url.scheme == "http":
            if url.hostname in ("localhost", "127.0.0.1"):
                # We purposefully allow HTTP without warnings for localhost/127.0.0.1.
                # This prevents console bloat for developers running local mock servers,
                # simulators, or containers, where traffic never leaves the machine.
                pass
            else:
                raise ClientConfigurationError(
                    f"Insecure HTTP requests to remote hosts are not allowed: '{iqm_server_url}'. "
                    "Please update your client configuration to use 'https://'."
                )

        hostname = url.hostname or ""
        path_segments = (url.path or "").split("/")

        quantum_computer_from_path = path_segments[-1].removesuffix(":timeslot")
        if quantum_computer is not None and quantum_computer_from_path:
            raise ClientConfigurationError(
                "The IQM Server URL must not contain quantum computer name when initializing client with "
                + "explicit quantum computer name. To fix this error, use server base url."
            )
        if quantum_computer_from_path:
            raise ClientConfigurationError(
                "The IQM Server URL must not contain the quantum computer name when initializing client. "
                "Use the server base URL, and choose the quantum computer using the "
                "`quantum_computer` parameter instead."
            )
        # Same for timeslots: the timeslot / FIFO queue selection had to be embedded into URL, whereas in this
        # new implementation, the explicit timeslot usage is preferred upon the actual job submission. Fixing
        # compatibility but giving a warning about the deprecated usage.
        use_timeslot_default = path_segments[-1].endswith(":timeslot")
        if use_timeslot_default:
            raise ClientConfigurationError(
                "Quantum computer timeslot URLs are obsolete. Individual jobs can be submitted to timeslots "
                + "by setting the `IQMClient.submit_circuits` parameter `use_timeslot` to True. See the IQM Server web "
                + "dashboard or https://docs.iqm.tech/iqm-client/ for more detailed instructions."
            )

        if hostname.startswith("cocos."):
            raise ClientConfigurationError(
                "Resonance CoCoS API is obsolete. Use https://resonance.iqm.tech. See the Resonance "
                + "documentation or https://docs.iqm.tech/iqm-client/ for more detailed instructions."
            )

        # Use hostname without "cocos" subdomain and quantum computer name
        port_suffix = f":{url.port}" if url.port else ""
        netloc = "/".join([hostname] + path_segments[:-1]).rstrip("/")
        base_url = f"{url.scheme}://{netloc}{port_suffix}"

        return base_url, quantum_computer

    def _default_headers(self) -> dict[str, str]:
        """Return the default headers for an HTTP request to IQM Server."""
        headers = {
            "User-Agent": self._signature,
            "Accept": "application/json",
        }
        # If auth header callback exists, use it to add the header
        if self._auth_header_callback:
            headers["Authorization"] = self._auth_header_callback()
        return headers

    @staticmethod
    def _remove_empty_values(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the given dict without values that are None or {}."""
        return {key: value for key, value in kwargs.items() if value not in [None, {}]}

    @staticmethod
    def _serialize_query_param(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @classmethod
    def _create_signature(cls, client_signature: str | None) -> str:
        """Prepare the User-Agent header sent to the server."""
        signature = f"{platform.platform(terse=True)}"
        signature += f", python {platform.python_version()}"
        dist_pkg_name = "iqm-client"
        signature += f", {cls.__name__} {dist_pkg_name} {version(dist_pkg_name)}"
        if client_signature:
            signature += f", {client_signature}"
        return signature

    def _resolve_quantum_computer(self, user_defined_quantum_computer: str | None) -> tuple[QuantumComputer, int]:
        """Resolve the quantum computer this client connects to.

        Returns:
            The resolved quantum computer, and the total number of quantum computers the server
            reported.

        """
        quantum_computers = self._get_quantum_computers()
        aliases = ", ".join(qc.alias for qc in quantum_computers)
        if user_defined_quantum_computer is None:
            if len(quantum_computers) == 1:
                return quantum_computers[0], 1
            raise ClientConfigurationError(f"Quantum computer not selected. Available quantum computers are: {aliases}")

        qc = next((qc for qc in quantum_computers if qc.alias == user_defined_quantum_computer), None)
        if qc is None:
            raise ClientConfigurationError(
                f'Quantum computer "{user_defined_quantum_computer}" does not exist. '
                + f"Available quantum computers are: {aliases}"
            )
        return qc, len(quantum_computers)

    @staticmethod
    def _resolve_proxy_path(quantum_computer: QuantumComputer, quantum_computer_count: int) -> str | None:
        """Resolve the station-proxy path prefix for a quantum computer, handling compatibility.

        Returns:
            The path prefix to use for station-proxied requests, or None if such requests are not
            available for this quantum computer.

        """
        if "proxy_path" not in quantum_computer.model_fields_set:
            # Server predates the proxy_path field: keep the legacy hardcoded path. A server with only
            # one quantum computer is probably (but not guaranteed to be) a single-station deployment,
            # which doesnt include QC name in the proxy path. Note that this is a best effort guess that
            # will break if a multi-station deployment happens to have only one QC.
            # this fallback can be removed once no server instances predating the addition of proxy_path
            # are running, or when the proxy endpoints are deprecated.
            if quantum_computer_count == 1:
                return "station"
            return f"station/{quantum_computer.alias}"
        return quantum_computer.proxy_path
