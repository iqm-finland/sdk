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
"""Client implementation for station control service REST API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from importlib.metadata import version
import json
import logging
import os
import platform
from time import sleep
from typing import Any, TypeVar
import uuid

from iqm.models.channel_properties import ChannelProperties
from opentelemetry import propagate, trace
from packaging.version import Version, parse
from pydantic import BaseModel
import requests

from exa.common.data.setting_node import SettingNode
from exa.common.errors.station_control_errors import (
    InternalServerError,
    NotFoundError,
    StationControlError,
    map_from_status_code_to_error,
)
from exa.common.qcm_data.qcm_data_client import QCMDataClient
from iqm.station_control.client.list_models import (
    DutFieldDataList,
    DutList,
    ListModel,
    ObservationDataList,
    ObservationDefinitionList,
    ObservationLiteList,
    ObservationSetDataList,
    ObservationUpdateList,
    ResponseWithMeta,
    RunLiteList,
    SequenceMetadataDataList,
)
from iqm.station_control.client.serializers import (
    deserialize_run_data,
    deserialize_sweep_results,
    serialize_run_job_request,
    serialize_sweep_job_request,
)
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.client.serializers.sweep_serializers import deserialize_sweep_data
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    DutData,
    DutFieldData,
    DynamicQuantumArchitecture,
    GetObservationsMode,
    JobData,
    JobExecutorStatus,
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
    SoftwareVersionSet,
    StaticQuantumArchitecture,
    Statuses,
    SweepData,
    SweepDefinition,
    SweepResults,
)
from iqm.station_control.interface.models.type_aliases import StrUUID
from iqm.station_control.interface.pydantic_base import PydanticBase
from iqm.station_control.interface.station_control import StationControlInterface

logger = logging.getLogger(__name__)
TypePydanticBase = TypeVar("TypePydanticBase", bound=PydanticBase)


class _StationControlClientBase(StationControlInterface):
    """Shared functionality for StationControlClient and IqmServerClient.

    Args:
        root_url: Remote server URL.
        get_token_callback: A callback function that returns a token
            which will be passed in Authorization header in all requests.
        client_signature: String that is added to the User-Agent header of requests
            sent to the server.
        enable_opentelemetry: Iff True, enable Jaeger/OpenTelemetry tracing.

    """

    def __init__(
        self,
        root_url: str,
        *,
        get_token_callback: Callable[[], str] | None = None,
        client_signature: str | None = None,
        enable_opentelemetry: bool = False,
    ):
        self.root_url = root_url
        self._get_token_callback = get_token_callback
        self._signature = self._create_signature(client_signature)
        self._enable_opentelemetry = enable_opentelemetry

    @classmethod
    def _create_signature(cls, client_signature: str | None) -> str:
        signature = f"{platform.platform(terse=True)}"
        signature += f", python {platform.python_version()}"
        dist_pkg_name = "iqm-station-control-client"
        signature += f", {cls.__name__} {dist_pkg_name} {version(dist_pkg_name)}"
        if client_signature:
            signature += f", {client_signature}"
        return signature

    def _send_request(
        self,
        http_method: Callable[..., requests.Response],
        url_path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_str: str | None = None,
        octets: bytes | None = None,
        timeout: int = 600,
    ) -> requests.Response:
        """Send an HTTP request.

        Parameters ``json_str`` and ``octets`` are mutually exclusive.
        The first non-None argument (in this order) will be used to construct the body of the request.

        Args:
            http_method: HTTP method to use for the request, any of requests.[post|get|put|head|delete|patch|options].
            url_path: URL for the request.
            headers: Additional HTTP headers for the request. Some may be overridden.
            params: HTTP query parameters to store in the query string of the request URL.
            json_str: JSON string to store in the body, may contain arbitrary Unicode characters.
            octets: Pre-serialized binary data to store in the body.
            timeout: Timeout for the request in seconds.

        Returns:
            Response to the request.

        Raises:
            StationControlError: Request was not successful.

        """
        # Will raise an error if respectively an error response code is returned.
        # http_method should be any of requests.[post|get|put|head|delete|patch|options]

        request_kwargs = self._build_request_kwargs(
            headers=headers or {}, params=params or {}, json_str=json_str, octets=octets, timeout=timeout
        )
        url = f"{self.root_url}/{url_path}"
        # TODO SW-1387: Use v1 API
        # url = f"{self.root_url}/{self.version}/{url_path}"
        response = http_method(url, **request_kwargs)
        if not response.ok:
            try:
                response_json = response.json()
                error_message = response_json.get("message") or response_json["detail"]
            except (json.JSONDecodeError, KeyError):
                error_message = response.text

            try:
                error_class = map_from_status_code_to_error(response.status_code)  # type: ignore[arg-type]
            except KeyError:
                raise RuntimeError(f"Unexpected response status code {response.status_code}: {error_message}")

            raise error_class(error_message)
        return response

    def _build_request_kwargs(
        self,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        json_str: str | None = None,
        octets: bytes | None = None,
        timeout: int,
    ) -> dict[str, Any]:
        """Prepare the keyword arguments for an HTTP request."""
        # add default headers
        headers["User-Agent"] = self._signature

        # json_str and octets are mutually exclusive
        data: bytes | None = None
        if json_str is not None:
            # Must be able to handle JSON strings with arbitrary unicode characters, so we use an explicit
            # encoding into bytes, and set the headers so the recipient can decode the request body correctly.
            data = json_str.encode("utf-8")
            headers["Content-Type"] = "application/json; charset=UTF-8"
        elif octets is not None:
            data = octets
            headers["Content-Type"] = "application/octet-stream"

        if self._enable_opentelemetry:
            parent_span_context = trace.set_span_in_context(trace.get_current_span())
            propagate.inject(carrier=headers, context=parent_span_context)

        # If token callback exists, use it to retrieve the token and add it to the headers
        if self._get_token_callback:
            headers["Authorization"] = self._get_token_callback()

        kwargs = {
            "headers": headers,
            "params": params,
            "data": data,
            "timeout": timeout,
        }
        return _remove_empty_values(kwargs)


class StationControlClient(_StationControlClientBase):
    """Client implementation for station control service REST API.

    Args:
        root_url: Remote station control service URL.
        get_token_callback: A callback function that returns a token (str)
            which will be passed in Authorization header in all requests.
        client_signature: String that is added to the User-Agent header of requests
            sent to the server.

    """

    def __init__(
        self, root_url: str, *, get_token_callback: Callable[[], str] | None = None, client_signature: str | None = None
    ):
        super().__init__(
            root_url,
            get_token_callback=get_token_callback,
            client_signature=client_signature,  # type: ignore[arg-type]
            enable_opentelemetry=os.environ.get("JAEGER_OPENTELEMETRY_COLLECTOR_ENDPOINT", None) is not None,
        )
        # TODO SW-1387: Remove this when using v1 API, not needed
        self._check_api_versions()
        qcm_data_url = os.environ.get("CHIP_DESIGN_RECORD_FALLBACK_URL", None)
        self._qcm_data_client = QCMDataClient(qcm_data_url) if qcm_data_url else None

    @property
    def version(self) -> str:
        """Return the version of the station control API this client is using."""
        return "v1"

    @cache
    def get_about(self) -> dict:
        response = self._send_request(requests.get, "about")
        return response.json()

    def get_health(self) -> dict:
        response = self._send_request(requests.get, "health")
        return response.json()

    @cache
    def get_configuration(self) -> dict:
        response = self._send_request(requests.get, "configuration")
        return response.json()

    @cache
    def get_exa_configuration(self) -> str:
        response = self._send_request(requests.get, "exa/configuration")
        return response.content.decode("utf-8")

    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        # FIXME: We don't have information if the object was created or fetched. Thus, server always responds 200 (OK).
        json_str = json.dumps(software_version_set)
        response = self._send_request(requests.post, "software-version-sets", json_str=json_str)
        return int(response.content)

    def get_settings(self) -> SettingNode:
        return self._get_cached_settings().model_copy()

    @cache
    def _get_cached_settings(self) -> SettingNode:
        response = self._send_request(requests.get, "settings")
        return deserialize_setting_node(response.content)

    @cache
    def get_chip_design_record(self, dut_label: str) -> dict:
        try:
            response = self._send_request(requests.get, f"chip-design-records/{dut_label}")
        except StationControlError as err:
            if isinstance(err, NotFoundError) and self._qcm_data_client:
                return self._qcm_data_client.get_chip_design_record(dut_label)
            raise err
        return response.json()

    @cache
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        headers = {"accept": "application/octet-stream"}
        response = self._send_request(requests.get, "channel-properties", headers=headers)
        decoded_dict = unpack_channel_properties(response.content)
        return decoded_dict

    def sweep(
        self,
        sweep_definition: SweepDefinition,
    ) -> dict:
        data = serialize_sweep_job_request(sweep_definition, queue_name="sweeps")
        return self._send_request(requests.post, "sweeps", octets=data).json()

    def get_sweep(self, sweep_id: StrUUID) -> SweepData:
        response = self._send_request(requests.get, f"sweeps/{sweep_id}")
        return deserialize_sweep_data(response.json())

    def delete_sweep(self, sweep_id: StrUUID) -> None:
        self._send_request(requests.delete, f"sweeps/{sweep_id}")

    def get_sweep_results(self, sweep_id: StrUUID) -> SweepResults:
        response = self._send_request(requests.get, f"sweeps/{sweep_id}/results")
        return deserialize_sweep_results(response.content)

    def run(
        self,
        run_definition: RunDefinition,
        update_progress_callback: Callable[[Statuses], None] | None = None,
        wait_job_completion: bool = True,
    ) -> bool:
        data = serialize_run_job_request(run_definition, queue_name="sweeps")

        response = self._send_request(requests.post, "runs", octets=data)
        if wait_job_completion:
            return self._wait_job_completion(response.json()["job_id"], update_progress_callback)
        return False

    def get_run(self, run_id: StrUUID) -> RunData:
        response = self._send_request(requests.get, f"runs/{run_id}")
        return deserialize_run_data(response.json())

    def query_runs(self, **kwargs) -> ListWithMeta[RunLite]:  # type: ignore[type-arg]
        params = self._clean_query_parameters(RunData, **kwargs)
        response = self._send_request(requests.get, "runs", params=params)
        return self._deserialize_response(response, RunLiteList, list_with_meta=True)

    def create_observations(
        self, observation_definitions: Sequence[ObservationDefinition]
    ) -> ListWithMeta[ObservationData]:  # type: ignore[type-arg]
        json_str = self._serialize_model(ObservationDefinitionList(observation_definitions))
        response = self._send_request(requests.post, "observations", json_str=json_str)
        return self._deserialize_response(response, ObservationDataList, list_with_meta=True)

    def get_observations(
        self,
        *,
        mode: GetObservationsMode,
        dut_label: str | None = None,
        dut_field: str | None = None,
        tags: list[str] | None = None,
        invalid: bool | None = False,
        run_ids: list[StrUUID] | None = None,  # type: ignore[override]
        sequence_ids: list[StrUUID] | None = None,  # type: ignore[override]
        limit: int | None = None,
    ) -> list[ObservationData]:
        kwargs = {
            "mode": mode,
            "dut_label": dut_label,
            "dut_field": dut_field,
            "tags": tags,
            "invalid": invalid,
            "run_ids": run_ids,
            "sequence_ids": sequence_ids,
            "limit": limit,
        }
        params = self._clean_query_parameters(ObservationData, **kwargs)
        response = self._send_request(requests.get, "observations", params=params)
        return self._deserialize_response(response, ObservationDataList)

    def query_observations(self, **kwargs) -> ListWithMeta[ObservationData]:  # type: ignore[type-arg]
        params = self._clean_query_parameters(ObservationData, **kwargs)
        response = self._send_request(requests.get, "observations", params=params)
        return self._deserialize_response(response, ObservationDataList, list_with_meta=True)

    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        json_str = self._serialize_model(ObservationUpdateList(observation_updates))
        response = self._send_request(requests.patch, "observations", json_str=json_str)
        return self._deserialize_response(response, ObservationDataList)

    def query_observation_sets(self, **kwargs) -> ListWithMeta[ObservationSetData]:  # type: ignore[type-arg]
        params = self._clean_query_parameters(ObservationSetData, **kwargs)
        response = self._send_request(requests.get, "observation-sets", params=params)
        return self._deserialize_response(response, ObservationSetDataList, list_with_meta=True)

    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        json_str = self._serialize_model(observation_set_definition)
        response = self._send_request(requests.post, "observation-sets", json_str=json_str)
        return self._deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def get_observation_set(self, observation_set_id: StrUUID) -> ObservationSetData:
        response = self._send_request(requests.get, f"observation-sets/{observation_set_id}")
        return self._deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        json_str = self._serialize_model(observation_set_update)
        response = self._send_request(requests.patch, "observation-sets", json_str=json_str)
        return self._deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def finalize_observation_set(self, observation_set_id: StrUUID) -> None:
        self._send_request(requests.post, f"observation-sets/{observation_set_id}/finalize")

    def get_observation_set_observations(self, observation_set_id: StrUUID) -> list[ObservationLite]:
        response = self._send_request(requests.get, f"observation-sets/{observation_set_id}/observations")
        return self._deserialize_response(response, ObservationLiteList)

    def get_default_calibration_set(self) -> ObservationSetData:
        response = self._send_request(requests.get, "calibration-sets/default")
        return self._deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def get_default_calibration_set_observations(self) -> list[ObservationLite]:
        response = self._send_request(requests.get, "calibration-sets/default/observations")
        return self._deserialize_response(response, ObservationLiteList)

    def get_default_dynamic_quantum_architecture(self) -> DynamicQuantumArchitecture:
        response = self._send_request(requests.get, "calibration-sets/default/dynamic-quantum-architecture")
        return self._deserialize_response(response, DynamicQuantumArchitecture)  # type: ignore[return-value]

    @cache
    def get_dynamic_quantum_architecture(self, calibration_set_id: StrUUID) -> DynamicQuantumArchitecture:
        response = self._send_request(
            requests.get, f"calibration-sets/{calibration_set_id}/dynamic-quantum-architecture"
        )
        return self._deserialize_response(response, DynamicQuantumArchitecture)  # type: ignore[return-value]

    def get_default_calibration_set_quality_metrics(self) -> QualityMetrics:
        response = self._send_request(requests.get, "calibration-sets/default/metrics")
        return self._deserialize_response(response, QualityMetrics)  # type: ignore[return-value]

    def get_calibration_set_quality_metrics(self, calibration_set_id: StrUUID) -> QualityMetrics:
        response = self._send_request(requests.get, f"calibration-sets/{calibration_set_id}/metrics")
        return self._deserialize_response(response, QualityMetrics)  # type: ignore[return-value]

    def get_duts(self) -> list[DutData]:
        response = self._send_request(requests.get, "duts")
        return self._deserialize_response(response, DutList)

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        params = {"dut_label": dut_label}
        response = self._send_request(requests.get, "dut-fields", params=params)
        return self._deserialize_response(response, DutFieldDataList)

    def query_sequence_metadatas(self, **kwargs) -> ListWithMeta[SequenceMetadataData]:  # type: ignore[type-arg]
        params = self._clean_query_parameters(SequenceMetadataData, **kwargs)
        response = self._send_request(requests.get, "sequence-metadatas", params=params)
        return self._deserialize_response(response, SequenceMetadataDataList, list_with_meta=True)

    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        json_str = self._serialize_model(sequence_metadata_definition)
        response = self._send_request(requests.post, "sequence-metadatas", json_str=json_str)
        return self._deserialize_response(response, SequenceMetadataData)  # type: ignore[return-value]

    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        # FIXME: We don't have information if the object was created or updated. Thus, server always responds 200 (OK).
        json_str = self._serialize_model(sequence_result_definition)
        response = self._send_request(
            requests.put, f"sequence-results/{sequence_result_definition.sequence_id}", json_str=json_str
        )
        return self._deserialize_response(response, SequenceResultData)  # type: ignore[return-value]

    def get_sequence_result(self, sequence_id: StrUUID) -> SequenceResultData:
        response = self._send_request(requests.get, f"sequence-results/{sequence_id}")
        return self._deserialize_response(response, SequenceResultData)  # type: ignore[return-value]

    @cache
    def get_static_quantum_architecture(self, dut_label: str) -> StaticQuantumArchitecture:
        response = self._send_request(requests.get, f"static-quantum-architectures/{dut_label}")
        return self._deserialize_response(response, StaticQuantumArchitecture)  # type: ignore[return-value]

    def get_job(self, job_id: StrUUID) -> JobData:
        response = self._send_request(requests.get, f"jobs/{job_id}")
        return self._deserialize_response(response, JobData)  # type: ignore[return-value]

    def abort_job(self, job_id: StrUUID) -> None:
        self._send_request(requests.post, f"jobs/{job_id}/abort")

    def _wait_job_completion(self, job_id: str, update_progress_callback: Callable[[Statuses], None] | None) -> bool:
        logger.info("Waiting for job ID: %s", job_id)
        update_progress_callback = update_progress_callback or (lambda status: None)
        try:
            job_status = self._poll_job_status_until_execution_start(job_id, update_progress_callback)
            if JobExecutorStatus(job_status) not in JobExecutorStatus.terminal_statuses():
                self._poll_job_status_until_terminal(job_id, update_progress_callback)
        except KeyboardInterrupt as exc:
            logger.info("Caught %s, revoking job %s", exc, job_id)
            self.abort_job(uuid.UUID(job_id))
            return True
        return False

    def _poll_job_status_until_execution_start(
        self, job_id: str, update_progress_callback: Callable[[Statuses], None]
    ) -> JobExecutorStatus:
        # Keep polling job status as long as it's PENDING, and update progress with `update_progress_callback`.
        max_seen_position = 0
        while True:
            job = self._poll_job(job_id)
            if job.job_status >= JobExecutorStatus.EXECUTION_STARTED:  # type: ignore[operator]
                if max_seen_position:
                    update_progress_callback([("Progress in queue", max_seen_position, max_seen_position)])
                return job.job_status
            position = job.position

            if position == 0:
                sleep(1)
                continue
            max_seen_position = max(max_seen_position, position)  # type: ignore[type-var,assignment]
            update_progress_callback([("Progress in queue", max_seen_position - position, max_seen_position)])  # type: ignore[operator]
            sleep(1)

    def _poll_job_status_until_terminal(
        self,
        job_id: str,
        update_progress_callback: Callable[[Statuses], None],
    ) -> None:
        # Keep polling job status until it finishes, and update progress with `update_progress_callback`.
        while True:
            job = self._poll_job(job_id)
            update_progress_callback(job.job_result.parallel_sweep_progress)
            if job.job_status in JobExecutorStatus.terminal_statuses():
                return
            sleep(1)

    def _poll_job(self, job_id: str) -> JobData:
        response = self._send_request(requests.get, f"jobs/{job_id}")
        job = self._deserialize_response(response, JobData)
        if job.job_status == JobExecutorStatus.FAILED:  # type: ignore[union-attr]
            raise InternalServerError(f"Job: {job.job_id}\n{job.job_error}")  # type: ignore[union-attr]  # type: ignore[union-attr]
        return job  # type: ignore[return-value]

    @staticmethod
    def _serialize_model(model: BaseModel) -> str:
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

    # TODO SW-1387: Remove this when using v1 API, not needed
    def _check_api_versions(self):
        client_api_version = self._get_client_api_version()
        # Parse versions using standard packaging.version implementation.
        # For that purpose, we need to convert our custom " (local editable)" to follow packaging.version syntax.
        server_api_version = parse(
            self.get_about()["software_versions"]["iqm-station-control-client"].replace(" (local editable)", "+local")
        )

        if client_api_version.major != server_api_version.major:
            raise ValueError(
                f"station-control-client version '{client_api_version}' is not compatible with the station control "
                f"server, please use station-control-client version compatible with version '{server_api_version}'."
            )

        if client_api_version.local or server_api_version.local:
            logger.warning(
                "Client ('%s') and/or server ('%s') is using a local version of the station-control-client. "
                "Client and server compatibility cannot be guaranteed.",
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
    def _clean_query_parameters(model: Any, **kwargs) -> dict[str, Any]:
        if issubclass(model, PydanticBase) and "invalid" in model.model_fields and "invalid" not in kwargs:
            # Get only valid items by default, "invalid=None" would return also invalid ones.
            # This default has to be set on the client side, server side uses default "None".
            kwargs["invalid"] = False
        return _remove_empty_values(kwargs)

    @staticmethod
    def _deserialize_response(
        response: requests.Response,
        model_class: type[TypePydanticBase | ListModel[list[TypePydanticBase]]],  # type: ignore[type-arg]
        *,
        list_with_meta: bool = False,
    ) -> TypePydanticBase | ListWithMeta[TypePydanticBase]:  # type: ignore[type-arg]
        # Use "model_validate_json(response.text)" instead of "model_validate(response.json())".
        # This validates the provided data as a JSON string or bytes object.
        # If your incoming data is a JSON payload, this is generally considered faster.
        if list_with_meta:
            response_with_meta = ResponseWithMeta.model_validate_json(response.text)  # type: ignore[var-annotated]
            if response_with_meta.meta and response_with_meta.meta.errors:
                logger.warning(
                    "Errors in station control response:\n  - %s", "\n  - ".join(response_with_meta.meta.errors)
                )
            return ListWithMeta(model_class.model_validate(response_with_meta.items), meta=response_with_meta.meta)  # type: ignore[arg-type]
        model = model_class.model_validate_json(response.text)
        return model  # type: ignore[return-value]


def _remove_empty_values(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the given dict without values that are None or {}."""
    return {key: value for key, value in kwargs.items() if value not in [None, {}]}
