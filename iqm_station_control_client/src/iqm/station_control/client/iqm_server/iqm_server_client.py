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
"""Client implementation for IQM Server."""

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
import dataclasses
from io import BytesIO
import json
import logging
from time import sleep
from typing import Any, TypeVar, cast
import uuid
from uuid import UUID

import grpc
from iqm.models.channel_properties import ChannelProperties
import requests

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
from iqm.station_control.client.list_models import DutFieldDataList, DutList
from iqm.station_control.client.serializers import deserialize_sweep_results, serialize_sweep_job_request
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.client.serializers.task_serializers import deserialize_sweep_job_request
from iqm.station_control.client.station_control import _StationControlClientBase
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    DutData,
    DutFieldData,
    DynamicQuantumArchitecture,
    JobData,
    JobExecutorStatus,
    JobResult,
    ObservationData,
    ObservationDefinition,
    ObservationLite,
    ObservationSetData,
    ObservationSetDefinition,
    ObservationSetUpdate,
    ObservationUpdate,
    QualityMetrics,
    RunData,
    RunDefinition,
    RunLite,
    SequenceMetadataData,
    SequenceMetadataDefinition,
    SequenceResultData,
    SequenceResultDefinition,
    StaticQuantumArchitecture,
    Statuses,
    SweepData,
    SweepDefinition,
    SweepResults,
)
from iqm.station_control.interface.models.jobs import JobError
from iqm.station_control.interface.models.sweep import SweepBase
from iqm.station_control.interface.models.type_aliases import GetObservationsMode, SoftwareVersionSet, StrUUID

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IqmServerClient(_StationControlClientBase):
    def __init__(
        self,
        root_url: str,
        *,
        get_token_callback: Callable[[], str] | None = None,
        client_signature: str | None = None,
        grpc_channel: grpc.Channel | None = None,
    ):
        super().__init__(root_url, get_token_callback=get_token_callback, client_signature=client_signature)
        self._connection_params = parse_connection_params(root_url)
        self._cached_resources: dict[str, Any] = {}
        self._latest_submitted_sweep = None
        self._channel = grpc_channel or create_channel(self._connection_params, self._get_token_callback)
        self._current_qc = resolve_current_qc(self._channel, self._connection_params.quantum_computer)

    def __del__(self):
        try:
            self._channel.close()
        except Exception:
            pass

    def get_about(self) -> dict:
        return self._get_resource("about", parse_json)

    def get_health(self) -> dict:
        raise NotImplementedError

    def get_configuration(self) -> dict:
        return self._get_resource("configuration", parse_json)

    def get_exa_configuration(self) -> str:
        raise NotImplementedError

    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        raise NotImplementedError

    def get_settings(self) -> SettingNode:
        return self._get_resource("settings", deserialize_setting_node).copy()

    def get_chip_design_record(self, dut_label: str) -> dict:
        return self._get_resource(f"chip-design-records/{dut_label}", parse_json)

    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        return self._get_resource("channel-properties", unpack_channel_properties)

    def sweep(self, sweep_definition: SweepDefinition) -> dict:
        with wrap_error("Job submission failed"):
            jobs = proto.JobsStub(self._channel)
            job: proto.JobV1 = jobs.SubmitJobV1(
                proto.SubmitJobRequestV1(
                    qc_id=self._current_qc.id,
                    type=proto.JobType.PULSE,
                    payload=serialize_sweep_job_request(sweep_definition, queue_name="sweeps"),
                    use_timeslot=self._connection_params.use_timeslot,
                )
            )
            # Optimization: we know that in most of the cases the submitted sweep is queried
            # right after submitting it so we can cache reference to the submitted sweep here
            # to avoid extra request to the server
            job_id = from_proto_uuid(job.id)
            self._latest_submitted_sweep = dataclasses.replace(sweep_definition, sweep_id=job_id)  # type: ignore[assignment]
            return {
                "job_id": str(job_id),
            }

    def get_sweep(self, sweep_id: UUID) -> SweepData:
        with wrap_error("Job loading failed"):
            if isinstance(sweep_id, str):
                sweep_id = uuid.UUID(sweep_id)

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
                job_status=to_job_status(job.status),
                # Sweep definition is a subclass of SweepBase so we can just copy all SweepBase fields
                # from the input sweep to the sweep data
                **{f.name: getattr(sweep, f.name) for f in dataclasses.fields(SweepBase)},
            )

    def delete_sweep(self, sweep_id: UUID) -> None:
        raise NotImplementedError

    def get_sweep_results(self, sweep_id: UUID) -> SweepResults:
        with wrap_error("Job result loading failed"):
            jobs = proto.JobsStub(self._channel)
            data_chunks = jobs.GetJobResultsV1(proto.JobLookupV1(id=to_proto_uuid(sweep_id)))
            return deserialize_sweep_results(load_all(data_chunks))

    def run(
        self,
        run_definition: RunDefinition,
        update_progress_callback: Callable[[Statuses], None] | None = None,
        wait_job_completion: bool = True,
    ) -> bool:
        raise NotImplementedError

    def get_run(self, run_id: UUID) -> RunData:
        raise NotImplementedError

    def query_runs(self, **kwargs) -> ListWithMeta[RunLite]:  # type: ignore[type-arg]
        raise NotImplementedError

    def create_observations(
        self, observation_definitions: Sequence[ObservationDefinition]
    ) -> ListWithMeta[ObservationData]:  # type: ignore[type-arg]
        raise NotImplementedError

    def get_observations(
        self,
        *,
        mode: GetObservationsMode,
        dut_label: str | None = None,
        dut_field: str | None = None,
        tags: list[str] | None = None,
        invalid: bool | None = False,
        run_ids: list[UUID] | None = None,
        sequence_ids: list[UUID] | None = None,
        limit: int | None = None,
    ) -> list[ObservationData]:
        raise NotImplementedError

    def query_observations(self, **kwargs) -> ListWithMeta[ObservationData]:  # type: ignore[type-arg]
        raise NotImplementedError

    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        raise NotImplementedError

    def query_observation_sets(self, **kwargs) -> ListWithMeta[ObservationSetData]:  # type: ignore[type-arg]
        raise NotImplementedError

    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        raise NotImplementedError

    def get_observation_set(self, observation_set_id: UUID) -> ObservationSetData:
        raise NotImplementedError

    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        raise NotImplementedError

    def finalize_observation_set(self, observation_set_id: UUID) -> None:
        raise NotImplementedError

    def get_observation_set_observations(self, observation_set_id: UUID) -> list[ObservationLite]:
        raise NotImplementedError

    def get_default_calibration_set(self) -> ObservationSetData:
        raise NotImplementedError

    def get_default_calibration_set_observations(self) -> list[ObservationLite]:
        raise NotImplementedError

    def get_dynamic_quantum_architecture(self, calibration_set_id: UUID) -> DynamicQuantumArchitecture:
        response = self._send_request(requests.get, f"api/v1/calibration/{calibration_set_id}/gates")
        return DynamicQuantumArchitecture.model_validate_json(response.text)

    def get_default_dynamic_quantum_architecture(self) -> DynamicQuantumArchitecture:
        raise NotImplementedError

    def get_default_calibration_set_quality_metrics(self) -> QualityMetrics:
        raise NotImplementedError

    def get_calibration_set_quality_metrics(self, calibration_set_id: UUID) -> QualityMetrics:
        raise NotImplementedError

    def get_duts(self) -> list[DutData]:
        return self._get_resource("duts", lambda data: DutList.model_validate(parse_json(data)))  # type: ignore[arg-type,return-value]

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        return self._get_resource(
            f"dut-fields/{dut_label}",
            lambda data: DutFieldDataList.model_validate(parse_json(data)),  # type: ignore[arg-type,return-value]
        )

    def query_sequence_metadatas(self, **kwargs) -> ListWithMeta[SequenceMetadataData]:  # type: ignore[type-arg]
        raise NotImplementedError

    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        raise NotImplementedError

    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        raise NotImplementedError

    def get_sequence_result(self, sequence_id: UUID) -> SequenceResultData:
        raise NotImplementedError

    def get_static_quantum_architecture(self, dut_label: str) -> StaticQuantumArchitecture:
        response = self._send_request(requests.get, "api/v1/quantum-architecture")
        return StaticQuantumArchitecture.model_validate_json(response.text)

    def get_job(self, job_id: StrUUID) -> JobData:
        with wrap_error("Job loading failed"):
            jobs = proto.JobsStub(self._channel)
            job: proto.JobV1 = jobs.GetJobV1(proto.JobLookupV1(id=to_proto_uuid(job_id)))
            return JobData(
                job_id=from_proto_uuid(job.id),
                job_status=job.status,  # type: ignore[arg-type]
                job_result=JobResult(
                    job_id=from_proto_uuid(job.id),
                    parallel_sweep_progress=[],
                    interrupted=False,
                ),
                job_error=JobError(full_error_log=job.error, user_error_message=job.error)
                if job.HasField("error")
                else None,
                position=job.queue_position if job.HasField("queue_position") else None,
            )

    def abort_job(self, job_id: StrUUID) -> None:
        with wrap_error("Job cancellation failed"):
            jobs = proto.JobsStub(self._channel)
            jobs.CancelJobV1(proto.JobLookupV1(id=to_proto_uuid(job_id)))

    def get_calibration_set_values(self, calibration_set_id: StrUUID) -> dict[str, ObservationValue]:
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

    def _wait_job_completion(
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
    sweep, _ = deserialize_sweep_job_request(job_payload)
    return sweep


def to_job_status(job_status: proto.JobStatus) -> JobExecutorStatus:
    match job_status:
        case proto.JobStatus.IN_QUEUE:
            return JobExecutorStatus.PENDING_EXECUTION
        case proto.JobStatus.EXECUTING:
            return JobExecutorStatus.EXECUTION_STARTED
        case proto.JobStatus.FAILED:
            return JobExecutorStatus.FAILED
        case proto.JobStatus.COMPLETED:
            return JobExecutorStatus.READY
        case proto.JobStatus.INTERRUPTED:
            return JobExecutorStatus.ABORTED
        case proto.JobStatus.CANCELLED:
            return JobExecutorStatus.ABORTED
    raise ValueError(f"Unknown job status: '{job_status}'")


def to_string_job_status(job_status: proto.JobStatus) -> str:
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
