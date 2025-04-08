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
"""StationControlClient implementation for IQM Server"""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
import dataclasses
from io import BytesIO
import json
import logging
from time import sleep
from typing import Any, TypeVar, cast
import uuid

import grpc
from iqm.models.channel_properties import ChannelProperties

from exa.common.data.setting_node import SettingNode
from exa.common.data.value import ObservationValue, validate_value
from iqm.station_control.client.iqm_server import proto
from iqm.station_control.client.iqm_server.error import IqmServerError
from iqm.station_control.client.iqm_server.grpc_utils import (
    create_channel,
    extract_error,
    from_proto_uuid,
    load_all,
    parse_connection_params,
    to_datetime,
    to_proto_uuid,
)
from iqm.station_control.client.iqm_server.meta_class import IqmServerClientMeta
from iqm.station_control.client.list_models import DutFieldDataList, DutList
from iqm.station_control.client.serializers import deserialize_sweep_results, serialize_sweep_task_request
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.client.serializers.task_serializers import deserialize_sweep_task_request
from iqm.station_control.client.station_control import StationControlClient
from iqm.station_control.interface.models import (
    DutData,
    DutFieldData,
    Statuses,
    SweepData,
    SweepDefinition,
    SweepResults,
    SweepStatus,
)
from iqm.station_control.interface.models.sweep import SweepBase

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IqmServerClient(StationControlClient, metaclass=IqmServerClientMeta):
    def __init__(
        self,
        root_url: str,
        get_token_callback: Callable[[], str] | None = None,
        grpc_channel: grpc.Channel | None = None,
    ):
        self.root_url = root_url
        self._connection_params = parse_connection_params(root_url)
        self._cached_resources = {}
        self._latest_submitted_sweep = None
        self._channel = grpc_channel or create_channel(self._connection_params, get_token_callback)
        self._current_qc = resolve_current_qc(self._channel, self._connection_params.quantum_computer)

    def __del__(self):
        try:
            self._channel.close()
        except Exception as e:
            logger.error("Failed to close GRPC channel gracefully", exc_info=e)

    def get_about(self) -> dict:
        return self._get_resource("about", parse_json)

    def get_configuration(self) -> dict:
        return self._get_resource("configuration", parse_json)

    def get_settings(self) -> SettingNode:
        return self._get_resource("settings", deserialize_setting_node).copy()

    def get_chip_design_record(self, dut_label: str) -> dict:
        return self._get_resource(f"chip-design-records/{dut_label}", parse_json)

    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        return self._get_resource("channel-properties", unpack_channel_properties)

    def get_duts(self) -> list[DutData]:
        return self._get_resource("duts", lambda data: DutList.model_validate(parse_json(data)))

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        return self._get_resource(
            f"dut-fields/{dut_label}", lambda data: DutFieldDataList.model_validate(parse_json(data))
        )

    def sweep(self, sweep_definition: SweepDefinition) -> dict:
        with wrap_error("Job submission failed"):
            jobs = proto.JobsStub(self._channel)
            job: proto.JobV1 = jobs.SubmitJobV1(
                proto.SubmitJobRequestV1(
                    qc_id=self._current_qc.id,
                    type=proto.JobType.PULSE,
                    payload=serialize_sweep_task_request(sweep_definition, queue_name="sweeps"),
                    use_timeslot=self._connection_params.use_timeslot,
                )
            )
            # Optimization: we know that in most of the cases the submitted sweep is queried
            # right after submitting it so we can cache reference to the submitted sweep here
            # to avoid extra request to the server
            job_id = from_proto_uuid(job.id)
            self._latest_submitted_sweep = dataclasses.replace(sweep_definition, sweep_id=job_id)
            return {
                "sweep_id": str(job_id),
                "task_id": str(job_id),
            }

    def get_sweep(self, sweep_id: uuid.UUID) -> SweepData:
        with wrap_error("Job loading failed"):
            jobs = proto.JobsStub(self._channel)
            job_lookup = proto.JobLookupV1(id=to_proto_uuid(sweep_id))
            job: proto.JobV1 = jobs.GetJobV1(job_lookup)
            # IQM server job does not include any details about the sweep properties so we need to
            # construct the resulting sweep data using the payload (= input sweep) and metadata
            # from the IQM server job
            sweep = self._get_cached_sweep(sweep_id) or payload_to_sweep(load_all(jobs.GetJobPayloadV1(job_lookup)))
            return SweepData(
                created_timestamp=to_datetime(job.created_at),
                modified_timestamp=to_datetime(job.updated_at),
                begin_timestamp=to_datetime(job.execution_started_at) if job.HasField("execution_started_at") else None,
                end_timestamp=to_datetime(job.execution_ended_at) if job.HasField("execution_ended_at") else None,
                sweep_status=to_sweep_status(job.status),
                # Sweep definition is a subclass of SweepBase so we can just copy all SweepBase fields
                # from the input sweep to the sweep data
                **{f.name: getattr(sweep, f.name) for f in dataclasses.fields(SweepBase)},
            )

    def get_sweep_results(self, sweep_id: uuid.UUID) -> SweepResults:
        with wrap_error("Job result loading failed"):
            jobs = proto.JobsStub(self._channel)
            data_chunks = jobs.GetJobResultsV1(proto.JobLookupV1(id=to_proto_uuid(sweep_id)))
            return deserialize_sweep_results(load_all(data_chunks))

    def revoke_sweep(self, sweep_id: uuid.UUID) -> None:
        with wrap_error("Job cancellation failed"):
            jobs = proto.JobsStub(self._channel)
            jobs.CancelJobV1(proto.JobLookupV1(id=to_proto_uuid(sweep_id)))

    def get_task(self, task_id: uuid.UUID) -> dict:
        with wrap_error("Job loading failed"):
            jobs = proto.JobsStub(self._channel)
            job: proto.JobV1 = jobs.GetJobV1(proto.JobLookupV1(id=to_proto_uuid(task_id)))
            return {
                # It would be nice to have these typed somewhere...
                "task_id": str(from_proto_uuid(job.id)),
                "task_status": to_task_status(job.status),
                "task_result": {"message": ""},
                "task_error": job.error if job.HasField("error") else "",
                "position": job.queue_position if job.HasField("queue_position") else None,
                "is_position_capped": False,
            }

    def _wait_task_completion(
        self,
        task_id: str,
        update_progress_callback: Callable[[Statuses], None] | None,
    ) -> bool:
        with wrap_error("Job subscription failed"):
            try:
                notify = update_progress_callback or (lambda _: None)
                job_id = uuid.UUID(task_id)
                initial_queue_position = None
                status = None
                # SubscribeToJobV1 runs until job reaches its final status (completed, failed, interrupted)
                job_events = subscribe_to_job_events(self._channel, job_id)
                for job in job_events:
                    status = job.status
                    if status == proto.JobStatus.IN_QUEUE:
                        if initial_queue_position is None:
                            initial_queue_position = job.queue_position
                        queue_progress = initial_queue_position - job.queue_position
                        notify([("Progress in queue", queue_progress, initial_queue_position)])
                # In case of success, mark progress bar to 100% (looks nicer)
                if initial_queue_position is not None and status == proto.JobStatus.COMPLETED:
                    notify([("Progress in queue", initial_queue_position, initial_queue_position)])
                return False
            except KeyboardInterrupt:
                return True

    def get_calibration_set_values(self, calibration_set_id: uuid.UUID) -> dict[str, ObservationValue]:
        with wrap_error("Calibration set loading failed"):
            calibrations = proto.CalibrationsStub(self._channel)
            data_chunks = calibrations.GetFullCalibrationDataV1(
                proto.CalibrationLookupV1(
                    id=to_proto_uuid(calibration_set_id),
                )
            )
            _, cal_set_values = parse_calibration_set(load_all(data_chunks))
            return cal_set_values

    def get_latest_calibration_set_id(self, dut_label: str) -> uuid.UUID:
        with wrap_error("Calibration set metadata loading failed"):
            calibrations = proto.CalibrationsStub(self._channel)
            metadata: proto.CalibrationMetadataV1 = calibrations.GetLatestQuantumComputerCalibrationV1(
                proto.LatestQuantumComputerCalibrationLookupV1(
                    qc_id=self._current_qc.id,
                )
            )
            if metadata.dut_label != dut_label:
                raise ValueError(f"No calibration set for dut_label = {dut_label}")
            return from_proto_uuid(metadata.id)

    def _get_cached_sweep(self, sweep_id: uuid.UUID) -> SweepDefinition | None:
        latest_submitted = self._latest_submitted_sweep
        if latest_submitted and latest_submitted.sweep_id == sweep_id:
            return latest_submitted
        return None

    def _get_resource(self, resource_name: str, deserialize: Callable[[bytes], T]) -> T:
        with wrap_error(f"Failed to load QC resource '{resource_name}'"):
            if (cached := self._cached_resources.get(resource_name)) is not None:
                return cached
            qcs = proto.QuantumComputersStub(self._channel)
            data_chunks = qcs.GetQuantumComputerResourceV1(
                proto.QuantumComputerResourceLookupV1(
                    qc_id=self._current_qc.id,
                    resource_name=resource_name,
                )
            )
            resource = deserialize(load_all(data_chunks))
            self._cached_resources[resource_name] = resource
            return resource


def resolve_current_qc(channel: grpc.Channel, alias: str) -> proto.QuantumComputerV1:
    qcs = proto.QuantumComputersStub(channel)
    qc_list: proto.QuantumComputersListV1 = qcs.ListQuantumComputersV1(proto.ListQuantumComputerFiltersV1())
    for qc in qc_list.items:
        if qc.alias == alias:
            return qc
    raise ValueError(f"Quantum computer '{alias}' does not exist")


def subscribe_to_job_events(channel: grpc.Channel, job_id: uuid.UUID) -> Iterable[proto.JobV1]:
    jobs = proto.JobsStub(channel)
    attempts = 1
    while True:
        try:
            events = jobs.SubscribeToJobV1(proto.JobLookupV1(id=to_proto_uuid(job_id)))
            for event in events:
                job_event = cast(proto.JobEventV1, event)
                if job_event.HasField("update"):
                    yield job_event.update
            return
        except grpc.RpcError as e:
            # Server may cancel subscription due to e.g. restarts, in which case we can just retry after some waiting
            error = extract_error(e)
            if error.error_code == "server_cancel" and attempts <= 10:
                attempts += 1
                sleep(5)
                continue
            raise e


def parse_calibration_set(cal_set_data: bytes) -> tuple[uuid.UUID, dict[str, ObservationValue]]:
    # IQM server calibration sets are in cocos calibration set JSON format, we can get
    # both id and observations from it
    cal_set = parse_json(cal_set_data)
    cal_set_id = cal_set["calibration_set_id"]
    observations = cal_set.get("observations", {})
    cal_set_values = {k: validate_value(v["value"]) for k, v in observations.items()}
    return cal_set_id, cal_set_values


def payload_to_sweep(job_payload: bytes) -> SweepDefinition:
    sweep, _ = deserialize_sweep_task_request(job_payload)
    return sweep


def to_sweep_status(job_status: proto.JobStatus) -> SweepStatus:
    match job_status:
        case proto.JobStatus.IN_QUEUE:
            return SweepStatus.PENDING
        case proto.JobStatus.EXECUTING:
            return SweepStatus.PROGRESS
        case proto.JobStatus.FAILED:
            return SweepStatus.FAILURE
        case proto.JobStatus.COMPLETED:
            return SweepStatus.SUCCESS
        case proto.JobStatus.INTERRUPTED:
            return SweepStatus.INTERRUPTED
        case proto.JobStatus.CANCELLED:
            return SweepStatus.REVOKED
    raise ValueError(f"Unknown job status: '{job_status}'")


def to_task_status(job_status: proto.JobStatus) -> str:
    match job_status:
        case proto.JobStatus.IN_QUEUE:
            return "PENDING"
        case proto.JobStatus.EXECUTING:
            return "STARTED"
        case proto.JobStatus.FAILED | proto.JobStatus.INTERRUPTED | proto.JobStatus.CANCELLED:
            return "FAILURE"
        case proto.JobStatus.COMPLETED:
            return "SUCCESS"
    raise ValueError(f"Unknown job status: '{job_status}'")


def parse_json(data: bytes) -> Any:
    return json.load(BytesIO(data))


@contextmanager
def wrap_error(title: str):
    try:
        yield
    except grpc.RpcError as e:
        raise extract_error(e, title) from e
    except Exception as e:
        raise IqmServerError(message=f"{title}: {e}", status_code=str(grpc.StatusCode.INTERNAL.name)) from e
