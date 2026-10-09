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
"""Standardized interface for quantum data persistence and historical retrieval.

This module provides the :class:`StorageInterface`, which decouples data storage
and retrieval logic from specific database or filesystem implementations.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from typing_extensions import deprecated

from exa.common.helpers.deprecation import format_deprecated
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    DutFieldData,
    DynamicQuantumArchitecture,
    ObservationData,
    ObservationDefinition,
    ObservationSetData,
    ObservationSetDefinition,
    ObservationSetUpdate,
    ObservationUpdate,
    RunData,
    RunLite,
    SequenceMetadataData,
    SequenceMetadataDefinition,
    SequenceResultData,
    SequenceResultDefinition,
    StrUUID,
)
from iqm.station_control.interface.models.observation_set import CalibrationSet, QualityMetricSet
from iqm.station_control.interface.models.type_aliases import StrUUIDOrDefault
from iqm.station_control.interface.serializable import Serializable


class StorageInterface(Serializable):
    """Interface for managing the persistence and retrieval of quantum data.

    This interface handles all passive operations: querying historical records,
    managing metadata for observations and sequences, and retrieving
    characterization data.

    Generic query methods (:meth:`query_observations`, :meth:`query_runs`, etc.)
    utilize a Django-inspired double-underscore lookup syntax: ``field__lookup=value``.
    Note double-underscore in the name, to separate field names like ``dut_field``
    from lookup types like ``in``.

    As a convenience, when no lookup type is provided (like in ``dut_label="foo"``),
    the lookup type is assumed to be exact (``dut_label__exact="foo"``).
    Other supported lookup types are:

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
    Check query method's own documentation for details about currently supported query parameters.

    Generic query methods will return a list of objects, but with additional (optional) "meta" attribute,
    which contains metadata, like pagination details. The client can ignore this data,
    or use it to implement pagination logic for example to fetch all results available.

    """

    # TODO (Marko): dut_label vs. chip_label?
    @abstractmethod
    def get_chip_design_record(self, dut_label: str) -> dict[str, Any]:
        """Get the chip design record for the given DUT label."""

    @abstractmethod
    def get_run(self, run_id: StrUUID) -> RunData:
        """Get run data from the database."""

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
    def query_observation_sets(self, **kwargs) -> ListWithMeta[ObservationSetData]:
        """Query observation sets from the database.

        Observation sets are queried by the given query parameters. Currently supported query parameters:
            - observation_set_id: UUID
            - observation_set_id__in: list[UUID]
            - observation_set_type: Literal["calibration-set", "generic-set", "quality-metric-set"]
            - describes_id: UUID
            - describes_id__in: list[UUID]
            - invalid: bool
            - created_timestamp__range: tuple[datetime, datetime]
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
            IQMError: If creation failed.

        """

    @abstractmethod
    def get_observation_set(self, observation_set_id: StrUUID) -> ObservationSetData:
        """Get an observation set from the database.

        Args:
            observation_set_id: Observation set to retrieve.

        Returns:
            The content of the observation set.

        Raises:
            IQMError: If retrieval failed.

        """

    @abstractmethod
    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        """Update an observation set in the database.

        Args:
            observation_set_update: The content of the observation set to be updated.

        Returns:
            The content of the observation set.

        Raises:
            IQMError: If updating failed.

        """

    @abstractmethod
    def finalize_observation_set(self, observation_set_id: StrUUID) -> None:
        """Finalize an observation set in the database.

        A finalized set is nearly immutable, allowing to change only ``invalid`` flag after finalization.

        Args:
            observation_set_id: Observation set to finalize.

        Raises:
            IQMError: If finalization failed.

        """

    @abstractmethod
    def get_observation_set_observations(self, observation_set_id: StrUUID) -> list[ObservationData]:
        """Get the constituent observations of an observation set from the database.

        Args:
            observation_set_id: UUID of the observation set to retrieve.

        Returns:
            Observations belonging to the given observation set.

        """

    @abstractmethod
    def get_calibration_set(self, calibration_set_id: StrUUIDOrDefault) -> CalibrationSet:
        """Get a calibration set from the database for the given calibration set ID."""

    @abstractmethod
    def get_dynamic_quantum_architecture(self, calibration_set_id: StrUUIDOrDefault) -> DynamicQuantumArchitecture:
        """Get the dynamic quantum architecture for the given calibration set ID.

        Returns:
            Dynamic quantum architecture of the quantum computer for the given calibration set ID.

        """

    @abstractmethod
    def get_calibration_set_quality_metric_set(self, calibration_set_id: StrUUIDOrDefault) -> QualityMetricSet:
        """Get the latest quality metric set for the given calibration set ID."""

    @deprecated(format_deprecated(old="StorageInterface.get_dut_fields", new=None, since="2026-08-21"))
    @abstractmethod
    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        """Get available fields for a specific DUT label from the database.

        Scans every field ever observed for the DUT, independent of any actual query need. Prefer filtering
        observations directly (e.g. by tags or observation set) instead of enumerating all dut fields upfront.

        This method will be removed in a future release.
        """

    @abstractmethod
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

    @abstractmethod
    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        """Create and persist new sequence metadata in the database."""

    @abstractmethod
    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        """Save sequence result in the database.

        This method creates the object if it doesn't exist and completely replaces the "data" and "final" if it does.
        Timestamps are assigned by the database. "modified_timestamp" is not set on initial creation,
        but it's updated on each subsequent call.
        """

    @abstractmethod
    def get_sequence_result(self, sequence_id: StrUUID) -> SequenceResultData:
        """Get sequence result from the database."""
