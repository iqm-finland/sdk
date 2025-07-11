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
"""Station control interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import logging
from typing import TypeVar
from uuid import UUID

from iqm.models.channel_properties import ChannelProperties

from exa.common.data.setting_node import SettingNode
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    DutData,
    DutFieldData,
    DynamicQuantumArchitecture,
    GetObservationsMode,
    JobData,
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
from iqm.station_control.interface.pydantic_base import PydanticBase

logger = logging.getLogger(__name__)
TypePydanticBase = TypeVar("TypePydanticBase", bound=PydanticBase)


class StationControlInterface(ABC):
    """Station control interface.

    Station control interface implementation should implement generic query methods for certain objects,
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
            For example, ``dut_field__in=["QB1.frequency", "gates.measure.constant.QB2.frequency"]``
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

    @abstractmethod
    def get_about(self) -> dict:
        """Return information about the station control."""

    @abstractmethod
    def get_health(self) -> dict:
        """Return the status of the station control service."""

    @abstractmethod
    def get_configuration(self) -> dict:
        """Return the configuration of the station control."""

    @abstractmethod
    def get_exa_configuration(self) -> str:
        """Return the recommended EXA configuration from the server."""

    @abstractmethod
    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        """Get software version set ID from the database, or create one if it doesn't exist."""

    @abstractmethod
    def get_settings(self) -> SettingNode:
        """Return a tree representation of the default settings as defined in the configuration file."""

    @abstractmethod
    def get_chip_design_record(self, dut_label: str) -> dict:
        """Get a raw chip design record matching the given chip label."""

    @abstractmethod
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        """Get channel properties from the station.

        Channel properties contain information regarding hardware limitations e.g. sampling rate, granularity
        and supported instructions.

        Returns:
            Mapping from channel name to AWGProperties or ReadoutProperties.

        """

    @abstractmethod
    def sweep(self, sweep_definition: SweepDefinition) -> dict:
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
            ExaError if submitting a sweep failed.

        """

    @abstractmethod
    def get_sweep(self, sweep_id: UUID) -> SweepData:
        """Get N-dimensional sweep data from the database."""

    @abstractmethod
    def delete_sweep(self, sweep_id: UUID) -> None:
        """Delete sweep in the database."""

    @abstractmethod
    def get_sweep_results(self, sweep_id: UUID) -> SweepResults:
        """Get N-dimensional sweep results from the database."""

    @abstractmethod
    def run(
        self,
        run_definition: RunDefinition,
        update_progress_callback: Callable[[Statuses], None] | None = None,
        wait_job_completion: bool = True,
    ) -> bool:
        """Execute an N-dimensional sweep of selected variables and save run, sweep and results."""

    @abstractmethod
    def get_run(self, run_id: UUID) -> RunData:
        """Get run data from the database."""

    @abstractmethod
    def query_runs(self, **kwargs) -> ListWithMeta[RunLite]:  # type: ignore[type-arg]
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

    @abstractmethod
    def create_observations(
        self, observation_definitions: Sequence[ObservationDefinition]
    ) -> ListWithMeta[ObservationData]:  # type: ignore[type-arg]
        """Create observations in the database.

        Args:
            observation_definitions: A sequence of observation definitions,
                each containing the content of the observation which will be created.

        Returns:
            Created observations, each including also the database created fields like ID and timestamps.

        """

    @abstractmethod
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

    @abstractmethod
    def query_observations(self, **kwargs) -> ListWithMeta[ObservationData]:  # type: ignore[type-arg]
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

    @abstractmethod
    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        """Update observations in the database.

        Args:
            observation_updates: A sequence of observation updates,
                each containing the content of the observation which will be updated.

        Returns:
            Updated observations, each including also the database created fields like ID and timestamps.

        """

    @abstractmethod
    def query_observation_sets(self, **kwargs) -> ListWithMeta[ObservationSetData]:  # type: ignore[type-arg]
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

    @abstractmethod
    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        """Create an observation set in the database.

        Args:
            observation_set_definition: The content of the observation set to be created.

        Returns:
            The content of the observation set.

        Raises:
            ExaError: If creation failed.

        """

    @abstractmethod
    def get_observation_set(self, observation_set_id: UUID) -> ObservationSetData:
        """Get an observation set from the database.

        Args:
            observation_set_id: Observation set to retrieve.

        Returns:
            The content of the observation set.

        Raises:
            ExaError: If retrieval failed.

        """

    @abstractmethod
    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        """Update an observation set in the database.

        Args:
            observation_set_update: The content of the observation set to be updated.

        Returns:
            The content of the observation set.

        Raises:
            ExaError: If updating failed.

        """

    @abstractmethod
    def finalize_observation_set(self, observation_set_id: UUID) -> None:
        """Finalize an observation set in the database.

        A finalized set is nearly immutable, allowing to change only ``invalid`` flag after finalization.

        Args:
            observation_set_id: Observation set to finalize.

        Raises:
            ExaError: If finalization failed.

        """

    @abstractmethod
    def get_observation_set_observations(self, observation_set_id: UUID) -> list[ObservationLite]:
        """Get the constituent observations of an observation set from the database.

        Args:
            observation_set_id: UUID of the observation set to retrieve.

        Returns:
            Observations belonging to the given observation set.

        """

    @abstractmethod
    def get_default_calibration_set(self) -> ObservationSetData:
        """Get default calibration set from the database."""

    @abstractmethod
    def get_default_calibration_set_observations(self) -> list[ObservationLite]:
        """Get default calibration set observations from the database."""

    @abstractmethod
    def get_default_dynamic_quantum_architecture(self) -> DynamicQuantumArchitecture:
        """Get dynamic quantum architecture for the default calibration set."""

    @abstractmethod
    def get_dynamic_quantum_architecture(self, calibration_set_id: UUID) -> DynamicQuantumArchitecture:
        """Get dynamic quantum architecture for the given calibration set ID.

        Returns:
            Dynamic quantum architecture of the station for the given calibration set ID.

        """

    @abstractmethod
    def get_default_calibration_set_quality_metrics(self) -> QualityMetrics:
        """Get the latest quality metrics for the current default calibration set."""

    @abstractmethod
    def get_calibration_set_quality_metrics(self, calibration_set_id: UUID) -> QualityMetrics:
        """Get the latest quality metrics for the given calibration set ID."""

    @abstractmethod
    def get_duts(self) -> list[DutData]:
        """Get DUTs of the station control."""

    @abstractmethod
    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        """Get DUT fields for the specified DUT label from the database."""

    @abstractmethod
    def query_sequence_metadatas(self, **kwargs) -> ListWithMeta[SequenceMetadataData]:  # type: ignore[type-arg]
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

    @abstractmethod
    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        """Create sequence metadata in the database."""

    @abstractmethod
    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        """Save sequence result in the database.

        This method creates the object if it doesn't exist and completely replaces the "data" and "final" if it does.
        Timestamps are assigned by the database. "modified_timestamp" is not set on initial creation,
        but it's updated on each subsequent call.
        """

    @abstractmethod
    def get_sequence_result(self, sequence_id: UUID) -> SequenceResultData:
        """Get sequence result from the database."""

    @abstractmethod
    def get_static_quantum_architecture(self, dut_label: str) -> StaticQuantumArchitecture:
        """Get static quantum architecture of the station for the given DUT label.

        Returns:
            Static quantum architecture of the station for the given DUT label.

        """

    @abstractmethod
    def get_job(self, job_id: UUID) -> JobData:
        """Get job data."""

    @abstractmethod
    def abort_job(self, job_id: UUID) -> None:
        """Either remove a job from the queue, or abort it gracefully if it's already executing.

        The status of the job will be set to ``JobStatus.ABORTED``.
        If the job is not found or is already finished nothing happens.
        """
