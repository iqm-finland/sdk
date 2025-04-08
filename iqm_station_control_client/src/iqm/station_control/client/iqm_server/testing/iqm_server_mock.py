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
"""Internal testing utilities for IqmServerClient"""

from collections.abc import Iterator
from datetime import datetime
import uuid

from google.protobuf import timestamp_pb2
import grpc

from iqm.station_control.client.iqm_server import proto
from iqm.station_control.client.iqm_server.grpc_utils import from_proto_uuid


class IqmServerMockBase(proto.QuantumComputersServicer, proto.CalibrationsServicer, proto.JobsServicer):
    """Base class for IQM server mocks. Only meant for testing IQM library packages, do *not*
    use outside of tests!
    """

    @staticmethod
    def proto_uuid(base: uuid.UUID | None = None) -> proto.Uuid:
        """Helper function for generating protobuf UUIDs"""
        return proto.Uuid(raw=(base or uuid.uuid4()).bytes)

    @staticmethod
    def parse_uuid(value: proto.Uuid) -> uuid.UUID:
        """Helper function for generating protobuf UUIDs"""
        return from_proto_uuid(value)

    @staticmethod
    def proto_timestamp(base: datetime | None = None) -> timestamp_pb2.Timestamp:
        """Helper function for generating protobuf timestamps"""
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(base or datetime.now())
        return timestamp

    def channel(self) -> grpc.Channel:
        """Gets a `grpc.Channel` that connects to this mock server instance. Can be used to initialize
        a new `IqmServerClient` that uses this mock server instance as a backend for the
        invoked GRPC calls.
        """
        return _MockChannel(self)

    @staticmethod
    def chunk_stream(data: bytes) -> Iterator[proto.DataChunk]:
        """A utility function for converting a binary data blob into a`(stream DataChunk)`."""
        yield proto.DataChunk(data=data)


class _MockChannel(grpc.Channel):
    def __init__(self, mock: IqmServerMockBase):
        self._mock = mock

    def subscribe(self, callback, try_to_connect=False):
        pass

    def unsubscribe(self, callback):
        pass

    def unary_unary(self, method, *args, **kwargs):
        return self._create_callable(method)

    def unary_stream(self, method, *args, **kwargs):
        return self._create_callable(method)

    def stream_unary(self, method, *args, **kwargs):
        return self._create_callable(method)

    def stream_stream(self, method, *args, **kwargs):
        return self._create_callable(method)

    def close(self):
        pass

    def _create_callable(self, fq_method: str):
        _, fn_name = fq_method.lstrip("/").split("/")
        f = getattr(self._mock, fn_name)

        def callable(request):
            return f(request, _MockContext())

        return callable


class _MockContext:
    def set_code(self, code):
        pass

    def set_details(self, details):
        pass
