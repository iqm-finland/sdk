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
"""Implementation of the ExecutorInterface for communication with an IQM Server.

This module provides the :class:`IQMServerExecutor`, which enables quantum job
execution and hardware metadata retrieval by interfacing with the IQM Server's
REST API.
"""

from functools import cache, lru_cache
from typing import Any, overload
import uuid

from iqm.data_definitions.station_control.v2.run_definition_pb2 import RunDefinition as RunDefinitionProto
from iqm.iqm_server_client.iqm_server_client import (
    CircuitCountsBatchAdapter,
    CircuitMeasurementResultsBatchAdapter,
    IQMServerClient,
)
from iqm.models.channel_properties import ChannelProperties
import requests
import ruamel.yaml

from exa.common.data.setting_node import SettingNode
from iqm.station_control.client.job_trackers import CircuitJobTracker, RunJobTracker
from iqm.station_control.client.list_models import StaticQuantumArchitectureList
from iqm.station_control.client.serializers import (
    deserialize_run_definition,
    deserialize_sweep_results,
    serialize_run_definition,
)
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
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
from iqm.station_control.interface.models.jobs_new import JobData, JobDefinition, JobType


# ruff: noqa: D102
class IQMServerExecutor(ExecutorInterface):
    """Concrete executor that communicates with a remote IQM Server to manage hardware execution and results."""

    def __init__(self, client: IQMServerClient):
        self.client = client

    @cache
    def get_experiment_configuration(self) -> dict[str, Any]:
        response = self.client.send_request(requests.get, f"{self.client.proxy_path}/exa/configuration")
        yaml = ruamel.yaml.YAML(typ="safe", pure=True)
        raw_yaml_data: dict = yaml.load(response.content.decode("utf-8")) or {}
        # FIX: Unpack the 'experiment' block if it exists!
        experiment_configuration_dict = raw_yaml_data.get("experiment", raw_yaml_data)
        return experiment_configuration_dict

    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        # This is meaningless to try and save to SC over proxy since there will be no job to link to.
        # For the moment, Exa can work with a mock response here until IQM Server can implement a way
        # to save these.
        return 0

    @cache
    def get_dut_label(self) -> str:
        return self.get_chip_design_record()["cheddar"]["chip_label"]

    @cache
    def get_chip_design_record(self) -> dict[str, Any]:
        response = self.client.send_request(
            requests.get, f"quantum-computers/{self.client.quantum_computer}/artifacts/chip-design-records"
        )
        # IQMServer returns a list of one item
        return response.json()[0]

    @cache
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        headers = {"Accept": "application/protobuf"}
        response = self.client.send_request(
            requests.get,
            f"quantum-computers/{self.client.quantum_computer}/artifacts/channel-properties",
            headers=headers,
        )
        decoded_dict = unpack_channel_properties(response.content)
        return decoded_dict

    @cache
    def get_static_quantum_architecture(self) -> StaticQuantumArchitecture:
        response = self.client.send_request(
            requests.get, f"quantum-computers/{self.client.quantum_computer}/artifacts/static-quantum-architectures"
        )
        # IQMServer returns a list of one item
        return self.client.deserialize_response(response, StaticQuantumArchitectureList)[0]

    def get_settings(self) -> SettingNode:
        return self._get_cached_settings().model_copy(deep=True)

    @cache
    def _get_cached_settings(self) -> SettingNode:
        headers = {"Accept": "application/protobuf"}
        response = self.client.send_request(
            requests.get, f"quantum-computers/{self.client.quantum_computer}/artifacts/settings", headers=headers
        )
        return deserialize_setting_node(response.content)

    @overload
    def submit_job(self, job_definition: RunDefinition, *, use_timeslot: bool = False) -> RunJobTracker: ...

    @overload
    def submit_job(self, job_definition: CircuitJobDefinition, *, use_timeslot: bool = False) -> CircuitJobTracker: ...

    def submit_job(self, job_definition: JobDefinition, *, use_timeslot: bool = False) -> JobTracker:
        if isinstance(job_definition, RunDefinition):
            # Create random run and sweep IDs for the job
            job_definition.run_id = uuid.uuid4()
            job_definition.sweep_definition.sweep_id = uuid.uuid4()
            job_data = self._submit_job(
                job_type="run",
                protobuf_data=serialize_run_definition(job_definition).SerializeToString(),
                use_timeslot=use_timeslot,
            )
            tracker_cls: type[JobTracker] = RunJobTracker
        elif isinstance(job_definition, CircuitJobDefinition):
            job_data = self._submit_job(
                job_type="circuit",
                json_data=self.client.serialize_model(job_definition),
                use_timeslot=use_timeslot,
            )
            tracker_cls = CircuitJobTracker
        else:
            raise ValueError(f"Unsupported job definition type: {type(job_definition)}")

        return tracker_cls(job_data=job_data, _payload=job_definition, _executor=self)

    def _submit_job(
        self,
        *,
        job_type: JobType,
        json_data: str | None = None,
        protobuf_data: bytes | None = None,
        use_timeslot: bool = False,
    ) -> JobData:
        params = self.client.serialize_query_params({"use_timeslot": use_timeslot})
        response = self.client.send_request(
            requests.post,
            f"jobs/{self.client.quantum_computer}/{job_type}",
            params=params,
            json_data=json_data,
            protobuf_data=protobuf_data,
        )
        return self.client.deserialize_response(response, JobData)  # type: ignore[return-value]

    def get_job(self, job_id: StrUUID) -> JobData:
        response = self.client.send_request(requests.get, f"jobs/{job_id}")
        return self.client.deserialize_response(response, JobData)  # type: ignore[return-value]

    @lru_cache(maxsize=1024)
    def get_job_payload(self, job_id: StrUUID, job_type: JobType) -> JobDefinition:
        response = self.client.send_request(requests.get, f"jobs/{job_id}/payload")

        if job_type == "run":
            run_definition_proto = RunDefinitionProto()
            run_definition_proto.ParseFromString(response.content)
            return deserialize_run_definition(run_definition_proto)

        if job_type == "circuit":
            return self.client.deserialize_response(response, CircuitJobDefinition)  # type: ignore[return-value]

        raise ValueError(f"Unsupported job type for payload deserialization: {job_type}")

    def get_artifact_sweep_results(self, job_id: StrUUID) -> SweepResults:
        response = self.client.send_request(requests.get, f"jobs/{job_id}/artifacts/sweep_results")
        return deserialize_sweep_results(response.content)

    def get_artifact_measurements(self, job_id: StrUUID) -> list[CircuitMeasurementResults]:
        response = self.client.send_request(requests.get, f"jobs/{job_id}/artifacts/measurements")
        return CircuitMeasurementResultsBatchAdapter.validate_json(response.text)

    def get_artifact_measurement_counts(self, job_id: StrUUID) -> list[CircuitMeasurementCounts]:
        response = self.client.send_request(requests.get, f"jobs/{job_id}/artifacts/measurement_counts")
        return CircuitCountsBatchAdapter.validate_json(response.text)

    def cancel_job(self, job_id: StrUUID) -> None:
        self.client.send_request(requests.post, f"jobs/{job_id}/cancel")

    def delete_job(self, job_id: StrUUID) -> None:
        self.client.send_request(requests.delete, f"jobs/{job_id}")

    def serialize(self) -> dict[str, Any]:
        return {
            "client": self.client.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "IQMServerExecutor":
        return cls(
            client=IQMServerClient.deserialize(data["client"]),
        )
