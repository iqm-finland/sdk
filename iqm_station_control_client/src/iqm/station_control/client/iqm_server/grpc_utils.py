# Copyright 2025 IQM
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
"""Internal utility functions used by IqmServerClient."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
import uuid

from google.protobuf import json_format, struct_pb2, timestamp_pb2
import grpc
from grpc import Compression
from pydantic import HttpUrl

from iqm.station_control.client.iqm_server import proto
from iqm.station_control.client.iqm_server.error import IqmServerError
from iqm.station_control.interface.models.type_aliases import StrUUID


class ClientCallDetails(grpc.ClientCallDetails):
    def __init__(self, details):
        self.method = details.method
        self.metadata = list(details.metadata or [])
        self.timeout = details.timeout
        self.credentials = details.credentials
        self.wait_for_ready = details.wait_for_ready
        self.compression = details.compression


class ApiTokenAuth(grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor):
    def __init__(self, get_token_callback: Callable[[], str]):
        self.get_token_callback = get_token_callback

    def _add_auth_header(self, client_call_details) -> ClientCallDetails:
        details = ClientCallDetails(client_call_details)
        details.metadata.append(("authorization", self.get_token_callback()))
        return details

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._add_auth_header(client_call_details), request)

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._add_auth_header(client_call_details), request)


@dataclass(frozen=True, kw_only=True)
class ConnectionParameters:
    server_address: str
    is_secure: bool
    quantum_computer: str
    use_timeslot: bool


def parse_connection_params(qc_url: str) -> ConnectionParameters:
    # Security measure: mitigate UTF-8 read order control character
    # exploits by allowing only ASCII urls
    if not qc_url.isascii():
        raise ValueError("Invalid quantum computer URL")

    # IQM Server QC urls are now form "https://cocos.<server_base_url>/<qc_name>[:timeslot]"
    # In the future, "cocos." subdomain will be dropped. The parsing logic should work with
    # the both url formats
    url = HttpUrl(qc_url)
    qc_name = (url.path or "").split("/")[-1].removesuffix(":timeslot")
    use_timeslot = qc_url.endswith(":timeslot")
    if not qc_name:
        raise ValueError("Invalid quantum computer URL: device name is missing")

    is_secure = url.scheme == "https"
    hostname = (url.host or "").removeprefix("cocos.")
    port = url.port or (443 if is_secure else 80)

    return ConnectionParameters(
        server_address=f"{hostname}:{port}",
        is_secure=is_secure,
        quantum_computer=qc_name,
        use_timeslot=use_timeslot,
    )


def create_channel(
    connection_params: ConnectionParameters,
    get_token_callback: Callable[[], str] | None = None,
    enable_compression: bool = True,
) -> grpc.Channel:
    compression = Compression.Gzip if enable_compression else None
    options = [
        # Let's try to parametrize this at least when we're merging station-control-client and iqm-client
        ("grpc.keepalive_time_ms", 5000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_timeout_ms", 1000),
    ]
    address = connection_params.server_address
    channel = (
        grpc.secure_channel(
            address, credentials=grpc.ssl_channel_credentials(), options=options, compression=compression
        )
        if connection_params.is_secure
        else grpc.insecure_channel(address, options=options, compression=compression)
    )
    if get_token_callback is not None:
        channel = grpc.intercept_channel(channel, ApiTokenAuth(get_token_callback))
    return channel


def to_proto_uuid(value: StrUUID) -> proto.Uuid:
    if isinstance(value, str):
        value = uuid.UUID(value)
    return proto.Uuid(raw=value.bytes)


def from_proto_uuid(value: proto.Uuid) -> uuid.UUID:
    if value.WhichOneof("data") == "str":
        return uuid.UUID(hex=value.str)
    return uuid.UUID(bytes=value.raw)


def to_datetime(timestamp: timestamp_pb2.Timestamp) -> datetime:
    return timestamp.ToDatetime()


def load_all(chunks: Iterable[proto.DataChunk]) -> bytes:
    result = bytearray()
    for chunk in chunks:
        result.extend(chunk.data)
    return bytes(result)


def extract_error(error: grpc.RpcError, title: str | None = None) -> IqmServerError:
    message = error.details()
    status_code = str(error.code().name)
    metadata = {k: v for k, v in list(error.initial_metadata()) + list(error.trailing_metadata())}
    error_code = str(metadata.get("error_code")) if "error_code" in metadata else None
    details = None
    if details_bin := metadata.get("grpc-status-details-bin"):
        value_proto = struct_pb2.Value()
        value_proto.ParseFromString(details_bin)
        details = json_format.MessageToJson(value_proto)
    return IqmServerError(
        message=f"{title}: {message}" if title else message,
        status_code=status_code,
        error_code=error_code,
        details=details,  # type: ignore[arg-type]
    )
