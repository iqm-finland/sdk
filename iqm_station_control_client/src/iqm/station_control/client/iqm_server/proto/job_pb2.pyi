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
from . import common_pb2 as _common_pb2
from . import uuid_pb2 as _uuid_pb2
from . import qc_pb2 as _qc_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CIRCUIT: _ClassVar[JobType]
    PULSE: _ClassVar[JobType]

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    IN_QUEUE: _ClassVar[JobStatus]
    EXECUTING: _ClassVar[JobStatus]
    COMPLETED: _ClassVar[JobStatus]
    CANCELLED: _ClassVar[JobStatus]
    FAILED: _ClassVar[JobStatus]
    INTERRUPTED: _ClassVar[JobStatus]
CIRCUIT: JobType
PULSE: JobType
IN_QUEUE: JobStatus
EXECUTING: JobStatus
COMPLETED: JobStatus
CANCELLED: JobStatus
FAILED: JobStatus
INTERRUPTED: JobStatus

class JobV1(_message.Message):
    __slots__ = ("id", "type", "quantum_computer", "input", "status", "queue_position", "error", "created_at", "updated_at", "execution_started_at", "execution_ended_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    QUANTUM_COMPUTER_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_POSITION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.Uuid
    type: JobType
    quantum_computer: _qc_pb2.QuantumComputerV1
    input: JobInputSummaryV1
    status: JobStatus
    queue_position: int
    error: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    execution_started_at: _timestamp_pb2.Timestamp
    execution_ended_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[_Union[_uuid_pb2.Uuid, _Mapping]] = ..., type: _Optional[_Union[JobType, str]] = ..., quantum_computer: _Optional[_Union[_qc_pb2.QuantumComputerV1, _Mapping]] = ..., input: _Optional[_Union[JobInputSummaryV1, _Mapping]] = ..., status: _Optional[_Union[JobStatus, str]] = ..., queue_position: _Optional[int] = ..., error: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., execution_started_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., execution_ended_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class JobInputSummaryV1(_message.Message):
    __slots__ = ("job_type", "shots", "circuits")
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    SHOTS_FIELD_NUMBER: _ClassVar[int]
    CIRCUITS_FIELD_NUMBER: _ClassVar[int]
    job_type: JobType
    shots: int
    circuits: int
    def __init__(self, job_type: _Optional[_Union[JobType, str]] = ..., shots: _Optional[int] = ..., circuits: _Optional[int] = ...) -> None: ...

class JobEventV1(_message.Message):
    __slots__ = ("keepalive", "update")
    KEEPALIVE_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    keepalive: _common_pb2.Keepalive
    update: JobV1
    def __init__(self, keepalive: _Optional[_Union[_common_pb2.Keepalive, _Mapping]] = ..., update: _Optional[_Union[JobV1, _Mapping]] = ...) -> None: ...

class JobLookupV1(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.Uuid
    def __init__(self, id: _Optional[_Union[_uuid_pb2.Uuid, _Mapping]] = ...) -> None: ...

class SubmitJobRequestV1(_message.Message):
    __slots__ = ("qc_id", "type", "payload", "use_timeslot")
    QC_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    USE_TIMESLOT_FIELD_NUMBER: _ClassVar[int]
    qc_id: _uuid_pb2.Uuid
    type: JobType
    payload: bytes
    use_timeslot: bool
    def __init__(self, qc_id: _Optional[_Union[_uuid_pb2.Uuid, _Mapping]] = ..., type: _Optional[_Union[JobType, str]] = ..., payload: _Optional[bytes] = ..., use_timeslot: bool = ...) -> None: ...
