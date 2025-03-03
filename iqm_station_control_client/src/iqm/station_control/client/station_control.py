# Copyright 2024 IQM
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
"""Station control client implementation."""

from collections.abc import Callable, Sequence
from functools import cache
from importlib.metadata import version
import json
import logging
import os
from time import sleep
from typing import Any, TypeVar
import uuid

from iqm.models.channel_properties import ChannelProperties
from opentelemetry import propagate, trace
from packaging.version import Version, parse
import requests

from exa.common.data.setting_node import SettingNode
from exa.common.errors.exa_error import RequestError
from exa.common.qcm_data.chad_model import CHAD
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
    serialize_run_task_request,
    serialize_sweep_task_request,
)
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.client.serializers.sweep_serializers import deserialize_sweep_data
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    DutData,
    DutFieldData,
    GetObservationsMode,
    ObservationData,
    ObservationDefinition,
    ObservationLite,
    ObservationSetData,
    ObservationSetDefinition,
    ObservationSetUpdate,
    ObservationUpdate,
    RunData,
    RunDefinition,
    RunLite,
    SequenceMetadataData,
    SequenceMetadataDefinition,
    SequenceResultData,
    SequenceResultDefinition,
    SoftwareVersionSet,
    Statuses,
    SweepData,
    SweepDefinition,
    SweepResults,
)
from iqm.station_control.interface.pydantic_base import PydanticBase

logger = logging.getLogger(__name__)
T = TypeVar("T")


class StationControlClient:
    """Station control client implementation.

    Current implementation uses HTTP calls to the remote station control service,
    that is controlling the station control instance.

    Args:
        root_url: Remote station control service URL.
        get_token_callback: A callback function that returns a token (str) which will be passed in Authorization header
            in all requests.

    Station control client implements generic query methods for certain objects,
    like :meth:`query_observations`, :meth:`query_observation_sets`, and :meth:`query_sequence_metadatas`.
    These methods accept only keyword arguments as parameters, which are based on the syntax ``field__lookup=value``.
    Note double-underscore in the name, to separate field names like ``dut_field`` from lookup types like ``in``.
    The syntax is based on Django implementation, documented
    `here <https://docs.djangoproject.com/en/5.0/ref/models/querysets/#field-lookups>`__ and
    `here <https://docs.djangoproject.com/en/5.0/ref/contrib/postgres/fields/#querying-arrayfield>`__.

    As a convenience, when no lookup type is provided (like in ``dut_label="foo"``),
    the lookup type is assumed to be exact (``dut_label__exact="foo"``). Other supported lookup types are:

        - range: Range test (inclusive).
            For example, ``created_timestamp__range=(datetime(2023, 10, 12), datetime(2024, 10, 14))``
        - in: In a given iterable; often a list, tuple, or queryset.
            For example, ``dut_field__in=["QB1.frequency", "QB2.readout.frequency"]``
        - icontains: Case-insensitive containment test.
            For example, ``origin_uri__icontains="local"``
        - overlap: Returns objects where the data shares any results with the values passed.
            For example, ``tags__overlap=["calibration=good", "2023-12-04"]``
        - contains: The returned objects will be those where the values passed are a subset of the data.
            For example, ``tags__contains=["calibration=good", "2023-12-04"]``
        - isnull: Takes either True or False, which correspond to SQL queries of IS NULL and IS NOT NULL, respectively.
            For example, ``end_timestamp__isnull=False``

    In addition to model fields (like "dut_label", "dut_field", "created_timestamp", "invalid", etc.),
    all of our generic query methods accept also following shared query parameters:

        - latest: str. Return only the latest item for this field, based on "created_timestamp".
            For example, ``latest="invalid"`` would return only one result (latest "created_timestamp")
            for each different "invalid" value in the database. Thus, maximum three results would be returned,
            one for each invalid value of `True`, `False`, and `None`.
        - order_by: str. Prefix with "-" for descending order, for example "-created_timestamp".
        - limit: int: Default 20. If 0 (or negative number) is given, then pagination is not used, i.e. limit=infinity.
        - offset: int. Default 0.

    Our generic query methods are not fully generalized yet, thus not all fields and lookup types are supported.
    Check query methods own documentation for details about currently supported query parameters.

    Generic query methods will return a list of objects, but with additional (optional) "meta" attribute,
    which contains metadata, like pagination details. The client can ignore this data,
    or use it to implement pagination logic for example to fetch all results available.

    """

    def __init__(self, root_url: str, get_token_callback: Callable[[], str] | None = None):
        self.root_url = root_url
        self._enable_opentelemetry = os.environ.get("JAEGER_OPENTELEMETRY_COLLECTOR_ENDPOINT", None) is not None
        self._get_token_callback = get_token_callback
        self._check_api_versions()
        qcm_url = os.environ.get("CHIP_DESIGN_RECORD_FALLBACK_URL", None)
        self._qcm_data_client = QCMDataClient(qcm_url) if qcm_url else None

    @cache
    def get_about(self) -> dict:
        """Return information about the station control."""
        response = self._send_request(requests.get, "about")
        return response.json()

    @cache
    def get_configuration(self) -> dict:
        """Return the configuration of the station control."""
        response = self._send_request(requests.get, "configuration")
        return response.json()

    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        """Get software version set ID from the database, or create if it doesn't exist."""
        # FIXME: We don't have information if the object was created or fetched. Thus, server always responds 200 (OK).
        data = json.dumps(software_version_set)
        response = self._send_request(requests.post, "software-version-sets", data=data)
        return int(response.content)

    def get_settings(self) -> SettingNode:
        """Return a tree representation of the default settings as defined in the configuration file."""
        return self._get_cached_settings().copy()

    @cache
    def _get_cached_settings(self) -> SettingNode:
        response = self._send_request(requests.get, "settings", headers={"Content-Type": "application/octet-stream"})
        return deserialize_setting_node(response.content)

    @cache
    def get_chip_design_record(self, dut_label: str) -> dict:
        """Get a raw chip design record matching the given chip label."""
        try:
            response = self._send_request(requests.get, f"chip-design-records/{dut_label}")
        except RequestError as err:
            if str(err).find("An error occurred on the server with the error code 404") >= 0 and self._qcm_data_client:
                return self._qcm_data_client.get_chip_design_record(dut_label)
            response = self._send_request(requests.get, f"chads/{dut_label}")
        return response.json()

    @cache
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        """Get channel properties from the station.

        Channel properties contain information regarding hardware limitations e.g. sampling rate, granularity
        and supported instructions.

        Returns:
            Mapping from channel name to AWGProperties or ReadoutProperties

        """
        headers = {"accept": "application/octet-stream"}
        response = self._send_request(requests.get, "channel-properties/", headers=headers)
        decoded_dict = unpack_channel_properties(response.content)
        return decoded_dict

    @cache
    def get_chad(self, dut_label: str) -> CHAD:
        """DEPRECATED."""
        logger.warning(
            "StationControlClient.get_chad is deprecated, use StationControlClient.get_chip_design_record instead."
        )
        try:
            response = self._send_request(requests.get, f"chads/{dut_label}")
            return CHAD(**response.json())
        except RequestError as err:
            if str(err).find("An error occurred on the server with the error code 404") >= 0 and self._qcm_data_client:
                chad_dict = self._qcm_data_client.get_chad(dut_label)
                return CHAD(**chad_dict)
            raise err

    @cache
    def get_qubit_design_properties(self, dut_label: str) -> dict:
        """DEPRECATED."""
        logger.warning(
            "StationControlClient.get_qubit_design_properties is deprecated, "
            "use StationControlClient.get_chip_design_record instead."
        )
        try:
            response = self._send_request(requests.get, f"qubit-design-properties/{dut_label}")
            return response.json()
        except RequestError as err:
            if str(err).find("An error occurred on the server with the error code 404") >= 0 and self._qcm_data_client:
                return self._qcm_data_client.get_qubit_design_properties(dut_label)
            raise err

    def sweep(
        self,
        sweep_definition: SweepDefinition,
    ) -> dict:
        """Execute an N-dimensional sweep of selected variables and save sweep and results.

        The raw data for each spot in the sweep is saved as numpy arrays,
        and the complete data for the whole sweep is saved as an x-array dataset
        which has the `sweep_definition.sweeps` as coordinates and
        data of `sweep_definition.return_parameters` data as DataArrays.

        The values of `sweep_definition.playlist` will be uploaded to the controllers given by the keys of
        `sweep_definition.playlist`.

        Args:
            sweep_definition: The content of the sweep to be created.

        Returns:
            Dict containing the task ID  and sweep ID, and corresponding hrefs, of a successful sweep execution
            in monolithic mode or successful submission to the task queue in remote mode.

        Raises:
            RequestError if submitting a sweep failed.

        """
        data = serialize_sweep_task_request(sweep_definition, queue_name="sweeps")
        headers = {"Content-Type": "application/octet-stream"}
        return self._send_request(requests.post, "sweeps", data=data, headers=headers).json()

    def get_sweep(self, sweep_id: uuid.UUID) -> SweepData:
        """Get N-dimensional sweep data from the database."""
        response = self._send_request(requests.get, f"sweeps/{sweep_id}")
        return deserialize_sweep_data(response.json())

    def revoke_sweep(self, sweep_id: uuid.UUID) -> None:
        """Either remove a sweep task from the queue, or abort it gracefully if it's already executing.

        If the task was already executing when revoked, the status of the task will be set to ``"INTERRUPTED"``.
        If the task had not started yet, the status will be set to ``"REVOKED"``.
        If the task is not found or is already finished nothing happens.
        """
        self._send_request(requests.post, f"sweeps/{sweep_id}/revoke")

    def delete_sweep(self, sweep_id: uuid.UUID) -> None:
        """Delete sweep in the database."""
        self._send_request(requests.delete, f"sweeps/{sweep_id}/delete")

    def get_sweep_results(self, sweep_id: uuid.UUID) -> SweepResults:
        """Get N-dimensional sweep results from the database."""
        response = self._send_request(requests.get, f"sweeps/{sweep_id}/results")
        return deserialize_sweep_results(response.content)

    def run(
        self,
        run_definition: RunDefinition,
        update_progress_callback: Callable[[Statuses], None] | None = None,
        wait_task_completion: bool = True,
    ) -> bool:
        """Execute an N-dimensional sweep of selected variables and save run, sweep and results."""
        data = serialize_run_task_request(run_definition, queue_name="sweeps")

        headers = {"Content-Type": "application/octet-stream"}
        response = self._send_request(requests.post, "runs", data=data, headers=headers)
        if wait_task_completion:
            return self._wait_task_completion(response.json()["task_id"], update_progress_callback)
        return False

    def get_run(self, run_id: uuid.UUID) -> RunData:
        """Get run data from the database."""
        response = self._send_request(requests.get, f"runs/{run_id}")
        return deserialize_run_data(response.json())

    def query_runs(self, **kwargs) -> ListWithMeta[RunLite]:
        """Query runs from the database.

        Runs are queried by the given query parameters. Currently supported query parameters:
            - run_id: uuid.UUID
            - run_id__in: list[uuid.UUID]
            - sweep_id: uuid.UUID
            - sweep_id__in: list[uuid.UUID]
            - username: str
            - username__in: list[str]
            - username__contains: str
            - username__icontains: str
            - experiment_label: str
            - experiment_label__in: list[str]
            - experiment_label__contains: str
            - experiment_label__icontains: str
            - experiment_name: str
            - experiment_name__in: list[str]
            - experiment_name__contains: str
            - experiment_name__icontains: str
            - software_version_set_id: int
            - software_version_set_id__in: list[int]
            - begin_timestamp__range: tuple[datetime, datetime]
            - end_timestamp__range: tuple[datetime, datetime]
            - end_timestamp__isnull: bool

        Returns:
            Queried runs with some query related metadata.

        """
        params = self._clean_query_parameters(RunData, **kwargs)
        response = self._send_request(requests.get, "runs", params=params)
        return self._create_list_with_meta(response, RunLiteList)

    def create_observations(
        self, observation_definitions: Sequence[ObservationDefinition]
    ) -> ListWithMeta[ObservationData]:
        """Create observations in the database.

        Args:
            observation_definitions: A sequence of observation definitions,
                each containing the content of the observation which will be created.

        Returns:
            Created observations, each including also the database created fields like ID and timestamps.

        """
        data = ObservationDefinitionList(observation_definitions).model_dump_json()
        response = self._send_request(requests.post, "observations", data=data)
        return self._create_list_with_meta(response, ObservationDataList)

    def get_observations(
        self,
        *,
        mode: GetObservationsMode,
        dut_label: str | None = None,
        dut_field: str | None = None,
        tags: list[str] | None = None,
        invalid: bool | None = False,
        run_ids: list[uuid.UUID] | None = None,
        sequence_ids: list[uuid.UUID] | None = None,
        limit: int | None = None,
    ) -> list[ObservationData]:
        """Get observations from the database.

        Observations are queried by the given query parameters.

        Args:
            mode: The "mode" used to query the observations. Possible values "all_latest", "tags_and", or "tags_or".

                  - "all_latest":Query all the latest observations for the given ``dut_label``.
                    No other query parameters are accepted.
                  - "tags_and": Query observations. Query all the observations that have all the given ``tags``.
                    By default, only valid observations are included.
                    All other query parameters can be used to narrow down the query,
                    expect "run_ids" and "sequence_ids".
                  - "tags_or": Query all the latest observations that have at least one of the given ``tags``.
                    Additionally, ``dut_label`` must be given. No other query parameters are used.
                  - "sequence": Query observations originating from a list of run and/or sequence IDs.
                    No other query parameters are accepted.
            dut_label: DUT label of the device the observations pertain to.
            dut_field: Name of the property the observation is about.
            tags: Human-readable tags of the observation.
            invalid: Flag indicating if the object is invalid. Automated systems must not use invalid objects.
                If ``None``, both valid and invalid objects are included.
            run_ids: The run IDs for which to query the observations.
            sequence_ids: The sequence IDs for which to query the observations.
            limit: Indicates the maximum number of items to return.

        Returns:
            Observations, each including also the database created fields like ID and timestamps.

        """
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
        return ObservationDataList.model_validate(response.json())

    def query_observations(self, **kwargs) -> ListWithMeta[ObservationData]:
        """Query observations from the database.

        Observations are queried by the given query parameters. Currently supported query parameters:
            - observation_id: int
            - observation_id__in: list[int]
            - dut_label: str
            - dut_field: str
            - dut_field__in: list[str]
            - tags__overlap: list[str]
            - tags__contains: list[str]
            - invalid: bool
            - source__run_id__in: list[uuid.UUID]
            - source__sequence_id__in: list[uuid.UUID]
            - source__type: str
            - uncertainty__isnull: bool
            - created_timestamp__range: tuple[datetime, datetime]
            - observation_set_ids__overlap: list[uuid.UUID]
            - observation_set_ids__contains: list[uuid.UUID]

        Returns:
            Queried observations with some query related metadata.

        """
        params = self._clean_query_parameters(ObservationData, **kwargs)
        response = self._send_request(requests.get, "observations", params=params)
        return self._create_list_with_meta(response, ObservationDataList)

    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        """Update observations in the database.

        Args:
            observation_updates: A sequence of observation updates,
                each containing the content of the observation which will be updated.

        Returns:
            Updated observations, each including also the database created fields like ID and timestamps.

        """
        data = ObservationUpdateList(observation_updates).model_dump_json()
        response = self._send_request(requests.patch, "observations", data=data)
        return ObservationDataList.model_validate(response.json())

    def query_observation_sets(self, **kwargs) -> ListWithMeta[ObservationSetData]:
        """Query observation sets from the database.

        Observation sets are queried by the given query parameters. Currently supported query parameters:
            - observation_set_id: UUID
            - observation_set_id__in: list[UUID]
            - observation_set_type: Literal["calibration-set", "generic-set", "quality-metric-set"]
            - observation_ids__overlap: list[int]
            - observation_ids__contains: list[int]
            - describes_id: UUID
            - describes_id__in: list[UUID]
            - invalid: bool
            - created_timestamp__range: tuple[datetime, datetime]
            - end_timestamp__isnull: bool
            - dut_label: str
            - dut_label__in: list[str]

        Returns:
            Queried observation sets with some query related metadata

        """
        params = self._clean_query_parameters(ObservationSetData, **kwargs)
        response = self._send_request(requests.get, "observation-sets", params=params)
        return self._create_list_with_meta(response, ObservationSetDataList)

    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        """Create an observation set in the database.

        Args:
            observation_set_definition: The content of the observation set to be created.

        Returns:
            The content of the observation set.

        Raises:
            ExaError: If creation failed.

        """
        data = observation_set_definition.model_dump_json()
        response = self._send_request(requests.post, "observation-sets", data=data)
        return ObservationSetData.model_validate(response.json())

    def get_observation_set(self, observation_set_id: uuid.UUID) -> ObservationSetData:
        """Get an observation set from the database.

        Args:
            observation_set_id: Observation set to retrieve.

        Returns:
            The content of the observation set.

        Raises:
            ExaError: If retrieval failed.

        """
        response = self._send_request(requests.get, f"observation-sets/{observation_set_id}")
        return ObservationSetData.model_validate(response.json())

    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        """Update an observation set in the database.

        Args:
            observation_set_update: The content of the observation set to be updated.

        Returns:
            The content of the observation set.

        Raises:
            ExaError: If updating failed.

        """
        data = observation_set_update.model_dump_json()
        response = self._send_request(requests.patch, "observation-sets", data=data)
        return ObservationSetData.model_validate(response.json())

    def finalize_observation_set(self, observation_set_id: uuid.UUID) -> None:
        """Finalize an observation set in the database.

        A finalized set is nearly immutable, allowing to change only ``invalid`` flag after finalization.

        Args:
            observation_set_id: Observation set to finalize.

        Raises:
            ExaError: If finalization failed.

        """
        self._send_request(requests.post, f"observation-sets/{observation_set_id}/finalize")

    def get_observation_set_observations(self, observation_set_id: uuid.UUID) -> list[ObservationLite]:
        """Get the constituent observations of an observation set from the database.

        Args:
            observation_set_id: UUID of the observation set to retrieve.

        Returns:
            Observations belonging to the given observation set.

        """
        response = self._send_request(requests.get, f"observation-sets/{observation_set_id}/observations")
        return ObservationLiteList.model_validate(response.json())

    def get_duts(self) -> list[DutData]:
        """Get DUTs of the station control."""
        response = self._send_request(requests.get, "duts")
        return DutList.model_validate(response.json())

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        """Get DUT fields for the specified DUT label from the database."""
        params = {"dut_label": dut_label}
        response = self._send_request(requests.get, "dut-fields", params=params)
        return DutFieldDataList.model_validate(response.json())

    def query_sequence_metadatas(self, **kwargs) -> ListWithMeta[SequenceMetadataData]:
        """Query sequence metadatas from the database.

        Sequence metadatas are queried by the given query parameters. Currently supported query parameters:
            - origin_id: str
            - origin_id__in: list[str]
            - origin_uri: str
            - origin_uri__icontains: str
            - created_timestamp__range: tuple[datetime, datetime]

        Returns:
            Sequence metadatas with some query related metadata.

        """
        params = self._clean_query_parameters(SequenceMetadataData, **kwargs)
        response = self._send_request(requests.get, "sequence-metadatas", params=params)
        return self._create_list_with_meta(response, SequenceMetadataDataList)

    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        """Create sequence metadata in the database."""
        data = sequence_metadata_definition.model_dump_json()
        response = self._send_request(requests.post, "sequence-metadatas", data=data)
        return SequenceMetadataData.model_validate(response.json())

    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        """Save sequence result in the database.

        This method creates the object if it doesn't exist and completely replaces the "data" and "final" if it does.
        Timestamps are assigned by the database. "modified_timestamp" is not set on initial creation,
        but it's updated on each subsequent call.
        """
        data = sequence_result_definition.model_dump_json()
        # FIXME: We don't have information if the object was created or updated. Thus, server always responds 200 (OK).
        response = self._send_request(
            requests.put, f"sequence-results/{sequence_result_definition.sequence_id}", data=data
        )
        return SequenceResultData.model_validate(response.json())

    def get_sequence_result(self, sequence_id: uuid.UUID) -> SequenceResultData:
        """Get sequence result from the database."""
        response = self._send_request(requests.get, f"sequence-results/{sequence_id}")
        return SequenceResultData.model_validate(response.json())

    def get_task(self, task_id: uuid.UUID) -> dict:
        """Get task data."""
        response = self._send_request(requests.get, f"tasks/{task_id}")
        return response.json()

    def _wait_task_completion(self, task_id: str, update_progress_callback: Callable[[Statuses], None] | None) -> bool:
        logger.info("Celery task ID: %s", task_id)
        update_progress_callback = update_progress_callback or (lambda status: None)
        try:
            task_status = self._poll_task_status_while_in_pending(task_id, update_progress_callback)
            if task_status == "PROGRESS":
                self._poll_task_status_while_in_progress(task_id, update_progress_callback)
        except KeyboardInterrupt as exc:
            logger.info("Caught %s, revoking task %s", exc, task_id)
            self._send_request(requests.post, f"tasks/{task_id}/revoke")
            return True
        return False

    def _poll_task_status_while_in_pending(
        self, task_id: str, update_progress_callback: Callable[[Statuses], None]
    ) -> str:
        # Keep polling task status as long as it's PENDING, and update progress with `update_progress_callback`.
        max_seen_position = 0
        was_warned = False
        while True:
            task = self._poll_task(task_id)
            if task["task_status"] != "PENDING":
                if max_seen_position:
                    update_progress_callback([("Progress in queue", max_seen_position, max_seen_position)])
                return task["task_status"]
            position = task["position"]
            if position == 0:
                sleep(1)
                continue
            max_seen_position = max(max_seen_position, position)
            if task["is_position_capped"] and not was_warned:
                logger.info(
                    "Over %s tasks before this in queue. Progress will be shown when in top %s.",
                    max_seen_position,
                    max_seen_position,
                )
                was_warned = True
            else:
                update_progress_callback([("Progress in queue", max_seen_position - position, max_seen_position)])
            sleep(1)

    def _poll_task_status_while_in_progress(
        self,
        task_id: str,
        update_progress_callback: Callable[[Statuses], None],
    ) -> None:
        # Keep polling task status as long as it's PROGRESS, and update progress with `update_progress_callback`.
        while True:
            task = self._poll_task(task_id)
            update_progress_callback(task["task_result"].get("parallel_sweep_progress", []))
            if task["task_status"] != "PROGRESS":
                return
            sleep(1)

    def _poll_task(self, task_id: str) -> dict:
        task = self._send_request(requests.get, f"tasks/{task_id}").json()
        if task["task_status"] == "FAILURE":
            raise RequestError(f"Server-side error. Details: {task}")
        return task

    def _send_request(
        self, http_method: Callable[..., requests.Response], url_path: str, **kwargs
    ) -> requests.Response:
        # Send the request and return the response.
        # Will raise an error if respectively an error response code is returned.
        # http_method should be any of requests.[post|get|put|head|delete|patch|options]
        params = kwargs.get("params", {})
        headers = kwargs.get("headers", {})
        data = kwargs.get("data", None)
        if self._enable_opentelemetry:
            parent_span_context = trace.set_span_in_context(trace.get_current_span())
            propagate.inject(carrier=headers, context=parent_span_context)
        # If token callback exists, use it to retrieve the token and add it to the headers
        if self._get_token_callback:
            headers["Authorization"] = self._get_token_callback()

        # Build request options explicitly
        http_request_options = {"params": params, "data": data, "headers": headers}
        # Remove None and {} values
        http_request_options = {key: value for key, value in http_request_options.items() if value not in [None, {}]}
        url = f"{self.root_url}/{url_path}"
        response = http_method(url, **http_request_options)
        if not response.ok:
            try:
                response_json = response.json()
                error_message = response_json["detail"]
            except json.JSONDecodeError:
                error_message = response.text
            raise RequestError(
                f"An error occurred on the server with the error code {response.status_code}: {error_message}"
            )
        return response

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

    @staticmethod
    def _get_client_api_version() -> Version:
        return parse(version("iqm-station-control-client"))

    @staticmethod
    def _clean_query_parameters(model: Any, **kwargs) -> dict:
        if issubclass(model, PydanticBase) and "invalid" in model.model_fields.keys() and "invalid" not in kwargs:
            # Get only valid items by default, "invalid=None" would return also invalid ones.
            # This default has to be set on the client side, server side uses default "None".
            kwargs["invalid"] = False
        # Remove None and {} values
        return {key: value for key, value in kwargs.items() if value not in [None, {}]}

    @staticmethod
    def _create_list_with_meta(response: requests.Response, list_model: type[ListModel[list[T]]]) -> ListWithMeta[T]:
        response_with_meta = ResponseWithMeta(**response.json())
        if response_with_meta.meta and response_with_meta.meta.errors:
            logger.warning("Errors in station control response:\n  - %s", "\n  - ".join(response_with_meta.meta.errors))
        return ListWithMeta(list_model.model_validate(response_with_meta.items), meta=response_with_meta.meta)
