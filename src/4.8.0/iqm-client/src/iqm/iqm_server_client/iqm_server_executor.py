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

from collections.abc import Iterable, Iterator
from functools import cache, lru_cache
from typing import Any, overload
import uuid

from iqm.data_definitions.station_control.v2.run_definition_pb2 import RunDefinition as RunDefinitionProto
from iqm.models.channel_properties import ChannelProperties
from pydantic import TypeAdapter
import requests
import ruamel.yaml

from exa.common.data import SettingNode
from iqm.station_control.client.job_trackers import CircuitJobTracker, RunJobTracker
from iqm.station_control.client.list_models import IterationDataList, JobDataList, StaticQuantumArchitectureList
from iqm.station_control.client.serializers import (
    deserialize_run_definition,
    deserialize_sweep_results,
    serialize_run_definition,
)
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.interface.executor_interface import ExecutorInterface, JobTracker
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    CircuitJobDefinition,
    CircuitMeasurementCounts,
    CircuitMeasurementResults,
    IterationData,
    IterationDefinition,
    IterationStatus,
    RunDefinition,
    SoftwareVersionSet,
    StaticQuantumArchitecture,
    StrUUID,
    SweepResults,
)
from iqm.station_control.interface.models.jobs_new import JobData, JobDefinition, JobStatus, JobType
from iqm.station_control.interface.pagination import paginate

from .iqm_server_client import IQMServerClient

_CircuitMeasurementResultsBatchAdapter = TypeAdapter(list[CircuitMeasurementResults])
_CircuitCountsBatchAdapter = TypeAdapter(list[CircuitMeasurementCounts])


# ruff: noqa: D102
class IQMServerExecutor(ExecutorInterface):
    """Concrete executor that communicates with a remote IQM Server to manage hardware execution and results."""

    def __init__(self, client: IQMServerClient):
        self.client = client

    def close(self) -> None:
        """Close the HTTP session of this executor's client, releasing its pooled connections."""
        self.client.close()

    @cache
    def get_experiment_configuration(self) -> dict[str, Any]:
        response = self.client.send_request(
            requests.get, f"quantum-computers/{self.client.quantum_computer}/artifacts/exa-configuration"
        )

        json_data = response.json()

        # yaml wrapped in json
        json_data = response.json()
        yaml_string = json_data["yaml"]

        yaml = ruamel.yaml.YAML(typ="safe", pure=True)
        experiment_config: dict = yaml.load(yaml_string) or {}
        # actual experiment config may or may not be under the "experiment" key
        return experiment_config.get("experiment", experiment_config)

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
    def submit_job(
        self, job_definition: RunDefinition, *, use_timeslot: bool = False, _iteration_id: StrUUID | None = None
    ) -> RunJobTracker: ...

    @overload
    def submit_job(
        self, job_definition: CircuitJobDefinition, *, use_timeslot: bool = False, _iteration_id: StrUUID | None = None
    ) -> CircuitJobTracker: ...

    def submit_job(
        self, job_definition: JobDefinition, *, use_timeslot: bool = False, _iteration_id: StrUUID | None = None
    ) -> JobTracker:
        if isinstance(job_definition, RunDefinition):
            # Create random run and sweep IDs for the job
            job_definition.run_id = uuid.uuid4()
            job_definition.sweep_definition.sweep_id = uuid.uuid4()
            job_data = self._submit_job(
                job_type="run",
                protobuf_data=serialize_run_definition(job_definition).SerializeToString(),
                use_timeslot=use_timeslot,
                iteration_id=_iteration_id,
            )
            tracker_cls: type[JobTracker] = RunJobTracker
        elif isinstance(job_definition, CircuitJobDefinition):
            job_data = self._submit_job(
                job_type="circuit",
                json_data=self.client.serialize_model(job_definition),
                use_timeslot=use_timeslot,
                iteration_id=_iteration_id,
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
        iteration_id: StrUUID | None = None,
    ) -> JobData:
        """Submit the given job for execution on the server."""
        if use_timeslot and iteration_id is not None:
            raise ValueError("The `use_timeslot` and `iteration_id` parameters are mutually exclusive.")
        params = self.client.serialize_query_params(
            {
                "use_timeslot": use_timeslot,
                "iteration_id": iteration_id,
            }
        )
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
        return _CircuitMeasurementResultsBatchAdapter.validate_json(response.text)

    def get_artifact_measurement_counts(self, job_id: StrUUID) -> list[CircuitMeasurementCounts]:
        response = self.client.send_request(requests.get, f"jobs/{job_id}/artifacts/measurement_counts")
        return _CircuitCountsBatchAdapter.validate_json(response.text)

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

    # Iteration methods. They are not a part of ExecutorInterface yet.

    def request_iteration(
        self,
        *,
        max_duration_seconds: int,
        idle_timeout_seconds: int | None = None,
    ) -> IterationData:
        """Request a new iteration from the server.

        * The iteration is created with status :attr:`.IterationStatus.WAITING`: the request has been
          accepted and queued, but the iteration is not active yet.
        * It transitions to :attr:`.IterationStatus.ACTIVE` on its own once the server starts it;
          poll :meth:`get_iteration` to observe that transition.
          Once active, the quantum computer will only execute jobs submitted into this iteration's queue.
        * While the iteration status is ``WAITING`` or ``ACTIVE`` it remains open.
          The iteration ends either automatically (maximum duration or idle timeout reached), or explicitly
          via :meth:`end_iteration`.

        Args:
            max_duration_seconds: Maximum duration of the iteration (in seconds).
                The iteration ends automatically once it has been active this long.
            idle_timeout_seconds: Maximum time (in seconds) the iteration queue may be empty.
                The iteration ends automatically once this limit has been passed.
                None means use the server default.

        Returns:
            Iteration status information.

        """
        model = IterationDefinition(
            qc=self.client.quantum_computer,
            max_duration_seconds=max_duration_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        response = self.client.send_request(
            requests.post,
            "iterations",
            json_data=self.client.serialize_model(model),
        )
        return self.client.deserialize_response(response, IterationData)  # type: ignore[return-value]

    def get_iterations(
        self,
        *,
        status: Iterable[IterationStatus] | None = None,
    ) -> Iterator[IterationData]:
        """Get the current data for iterations you own on the quantum computer the interface represents.

        IQM Server retains the data for finished iterations for some finite amount of time,
        after which they are purged.

        Args:
            status: Only include iterations currently in one of these statuses. None means all
                (including iterations already in a terminal state).

        Returns:
            Current data for the queried iterations.

        """
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = list(status)
        params["qc"] = self.client.quantum_computer

        def fetch_page(page_limit: int, offset: int) -> ListWithMeta[IterationData]:
            params["page_size"] = page_limit
            params["page"] = offset // page_limit + 1
            response = self.client.send_request(requests.get, "iterations", params=params)
            return self.client.deserialize_response(response, IterationDataList, list_with_meta=True)

        return paginate(fetch_page)

    def get_iteration(self, iteration_id: StrUUID) -> IterationData:
        """Get the current data for an iteration, including its status.

        Commonly used to poll the server (e.g. every few seconds) to monitor an iteration's status, for
        example to detect its transition from :attr:`.IterationStatus.WAITING` to
        :attr:`.IterationStatus.ACTIVE`, or to a terminal status (see
        :meth:`.IterationStatus.terminal_statuses`).

        Args:
            iteration_id: ID of the iteration to query.

        Returns:
            Current data for iteration ``iteration_id``.

        """
        response = self.client.send_request(requests.get, f"iterations/{iteration_id}")
        return self.client.deserialize_response(response, IterationData)  # type: ignore[return-value]

    def get_iteration_jobs(
        self,
        iteration_id: StrUUID,
        *,
        status: Iterable[JobStatus] | None = None,
    ) -> Iterator[JobData]:
        """Get all the jobs submitted to the given iteration.

        Args:
            iteration_id: ID of the iteration to query.
            status: Only include jobs currently in one of these statuses. None means all.

        Returns:
            Data for all jobs submitted to iteration ``iteration_id``.

        """
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = list(status)

        def fetch_page(page_limit: int, offset: int) -> ListWithMeta[JobData]:
            params["page_size"] = page_limit
            params["page"] = offset // page_limit + 1
            response = self.client.send_request(requests.get, f"iterations/{iteration_id}/jobs", params=params)
            return self.client.deserialize_response(response, JobDataList, list_with_meta=True)

        return paginate(fetch_page)

    def end_iteration(self, iteration_id: StrUUID) -> IterationData:
        """End an iteration which is not yet in a terminal state.

        * If the iteration is :attr:`.IterationStatus.WAITING` it transitions to
          :attr:`.IterationStatus.CANCELLED`: it never becomes active.
        * If the iteration is :attr:`.IterationStatus.ACTIVE`, any jobs waiting in its job queue are
          cancelled. Any currently executing job(s) are allowed to finish, after which the iteration
          transitions to :attr:`.IterationStatus.COMPLETED`.
        * If the iteration is already in a terminal state (see :meth:`.IterationStatus.terminal_statuses`)
          this method has no effect (it is idempotent).

        Args:
            iteration_id: ID of the iteration to end.

        Returns:
            Current status information for ``iteration_id``.

        """
        response = self.client.send_request(requests.post, f"iterations/{iteration_id}/end")
        return self.client.deserialize_response(response, IterationData)  # type: ignore[return-value]
