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
"""Implementation of the ExecutorInterface for communication with a Station Control.

This module provides the :class:`StationControlExecutor`, which enables quantum job
execution and hardware metadata retrieval by interfacing with the Station Control's
REST API.
"""

from dataclasses import fields
from functools import cache, lru_cache
import json
from typing import Any, TypeVar, cast, overload
import uuid

from iqm.models.channel_properties import ChannelProperties
import requests
import ruamel.yaml

from exa.common.data.setting_node import SettingNode
from exa.common.errors.iqm_error import ConflictError
from iqm.station_control.client.job_trackers import CircuitJobTracker, RunJobTracker
from iqm.station_control.client.list_models import DutList
from iqm.station_control.client.serializers import deserialize_sweep_results, serialize_run_job_request
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.client.station_control_client import StationControlClient
from iqm.station_control.interface.errors import IQMServerError
from iqm.station_control.interface.executor_interface import ExecutorInterface, JobTracker
from iqm.station_control.interface.models import (
    CircuitJobDefinition,
    CircuitMeasurementCounts,
    CircuitMeasurementResults,
    RunDefinition,
    SoftwareVersionSet,
    StaticQuantumArchitecture,
    StrUUID,
    SweepResults,
)
from iqm.station_control.interface.models.jobs import JobData as JobDataLegacy
from iqm.station_control.interface.models.jobs import JobExecutorStatus
from iqm.station_control.interface.models.jobs_new import (
    JobData,
    JobDefinition,
    JobExecution,
    JobStatus,
    JobType,
    ProgressInfo,
)

T = TypeVar("T")


def cast_dataclass(source_instance: Any, target_class: type[T], **extra_kwargs) -> T:
    """Copies shared fields from source_instance to a new target_class instance."""
    target_fields = {f.name for f in fields(target_class)}  # type: ignore[arg-type]
    # Handle standard instance dictionaries or pre-extracted dict data
    source_dict = source_instance.__dict__ if hasattr(source_instance, "__dict__") else source_instance

    clean_kwargs = {k: v for k, v in source_dict.items() if k in target_fields}
    clean_kwargs.update(extra_kwargs)
    return target_class(**clean_kwargs)


# ruff: noqa: D102
class StationControlExecutor(ExecutorInterface):
    """Concrete executor that communicates with a remote Station Control to manage hardware execution and results."""

    def __init__(self, client: StationControlClient):
        self.client = client

    @cache
    def get_experiment_configuration(self) -> dict[str, Any]:
        response = self.client.send_request(requests.get, "exa/configuration")
        yaml = ruamel.yaml.YAML(typ="safe", pure=True)
        raw_yaml_data: dict = yaml.load(response.content.decode("utf-8")) or {}
        # FIX: Unpack the 'experiment' block if it exists!
        experiment_configuration_dict = raw_yaml_data.get("experiment", raw_yaml_data)
        return experiment_configuration_dict

    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        # FIXME: We don't have information if the object was created or fetched. Thus, server always responds 200 (OK).
        json_data = json.dumps(software_version_set)
        response = self.client.send_request(requests.post, "software-version-sets", json_data=json_data)
        return int(response.content)

    @cache
    def get_dut_label(self) -> str:
        response = self.client.send_request(requests.get, "duts")
        duts = self.client.deserialize_response(response, DutList)
        if len(duts) != 1:
            raise ConflictError(f"There must be exactly one DUT available, got {len(duts)} instead.")
        return duts[0].label

    @cache
    def get_chip_design_record(self) -> dict[str, Any]:
        dut_label = self.get_dut_label()
        response = self.client.send_request(requests.get, f"chip-design-records/{dut_label}")
        return response.json()

    @cache
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        headers = {"accept": "application/protobuf"}
        response = self.client.send_request(requests.get, "channel-properties", headers=headers)
        decoded_dict = unpack_channel_properties(response.content)
        return decoded_dict

    @cache
    def get_static_quantum_architecture(self) -> StaticQuantumArchitecture:
        response = self.client.send_request(requests.get, f"static-quantum-architectures/{self.get_dut_label()}")
        return self.client.deserialize_response(response, StaticQuantumArchitecture)  # type: ignore[return-value]

    def get_settings(self) -> SettingNode:
        return self._get_cached_settings().model_copy(deep=True)

    @cache
    def _get_cached_settings(self) -> SettingNode:
        response = self.client.send_request(requests.get, "settings")
        return deserialize_setting_node(response.content)

    @overload
    def submit_job(self, job_definition: RunDefinition, *, use_timeslot: bool = False) -> RunJobTracker: ...

    @overload
    def submit_job(self, job_definition: CircuitJobDefinition, *, use_timeslot: bool = False) -> CircuitJobTracker: ...

    def submit_job(self, job_definition: JobDefinition, *, use_timeslot: bool = False) -> JobTracker:
        if isinstance(job_definition, RunDefinition):
            # Only assign new UUIDs if one hasn't been set yet!
            if not job_definition.run_id:
                job_definition.run_id = uuid.uuid4()

            if not job_definition.sweep_definition.sweep_id:
                job_definition.sweep_definition.sweep_id = uuid.uuid4()

            data = serialize_run_job_request(job_definition, queue_name="sweeps")
            response = self.client.send_request(requests.post, "runs", octets=data)
            job_data = self.get_job(response.json()["job_id"])
            tracker_cls: type[JobTracker] = RunJobTracker
        else:
            raise ValueError(f"Unsupported job definition type: {type(job_definition)}")

        return tracker_cls(job_data=job_data, _payload=job_definition, _executor=self)

    def get_job(self, job_id: StrUUID) -> JobData:
        response = self.client.send_request(requests.get, f"jobs/{job_id}")
        # convert JobDataLegacy into the current JobData
        job_data_legacy = cast(JobDataLegacy, self.client.deserialize_response(response, JobDataLegacy))
        status = self._convert_status(job_data_legacy.job_status)
        execution = None
        if job_data_legacy.job_result.parallel_sweep_progress:
            execution = JobExecution(
                progress={
                    label: ProgressInfo(value=value, max_value=max_value)
                    for label, value, max_value in job_data_legacy.job_result.parallel_sweep_progress
                }
            )
        job_error = job_data_legacy.job_error
        return JobData(
            id=job_data_legacy.job_id,
            type="run",
            status=status,
            execution=execution,
            errors=[]
            if job_error is None
            else [IQMServerError(message=job_error.user_error_message, source="iqm-station-control")],
            queue_position=job_data_legacy.position if status == JobStatus.WAITING else None,
        )

    @staticmethod
    def _convert_status(status: JobExecutorStatus) -> JobStatus:
        match status:
            case JobExecutorStatus.RECEIVED:
                return JobStatus.WAITING
            case JobExecutorStatus.READY:
                return JobStatus.COMPLETED
            case JobExecutorStatus.FAILED:
                return JobStatus.FAILED
            case JobExecutorStatus.ABORTED:
                return JobStatus.CANCELLED
            case _:
                return JobStatus.PROCESSING

    @lru_cache(maxsize=1024)
    def get_job_payload(self, job_id: StrUUID, job_type: JobType) -> JobDefinition:
        raise NotImplementedError(
            "StationControlExecutor doesn't support 'get_job_payload', "
            "use 'submit_job' to create 'JobTracker' with the correct payload."
        )

    def get_artifact_sweep_results(self, job_id: StrUUID) -> SweepResults:
        # "job_id" maps to the server's expected "sweep_id" parameter.
        response = self.client.send_request(requests.get, f"sweeps/{job_id}/results")
        return deserialize_sweep_results(response.content)

    def get_artifact_measurements(self, job_id: StrUUID) -> list[CircuitMeasurementResults]:
        raise NotImplementedError("StationControlExecutor doesn't support 'get_artifact_measurements'.")

    def get_artifact_measurement_counts(self, job_id: StrUUID) -> list[CircuitMeasurementCounts]:
        raise NotImplementedError("StationControlExecutor doesn't support 'get_artifact_measurement_counts'.")

    def cancel_job(self, job_id: StrUUID) -> None:
        self.client.send_request(requests.post, f"jobs/{job_id}/abort")

    def delete_job(self, job_id: StrUUID) -> None:
        self.client.send_request(requests.delete, f"jobs/{job_id}")

    def serialize(self) -> dict[str, Any]:
        return {
            "client": self.client.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "StationControlExecutor":
        return cls(
            client=StationControlClient.deserialize(data["client"]),
        )
