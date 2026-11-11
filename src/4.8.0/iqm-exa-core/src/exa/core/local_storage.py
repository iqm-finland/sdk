#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""A fully local, in-memory or file-system implementation of :class:`.StorageInterface`.

This module provides :class:`LocalStorage`, which keeps all data in plain Python objects
in the current process, with no database or server required. It only implements the
observation and observation set related methods of :class:`.StorageInterface`; everything
else raises :class:`NotImplementedError`. It is meant for small-scale local use (notebooks,
scripts, tests), not for production workloads with large amounts of data.

Optionally, an instance can be pointed at a JSON file on disk: existing state is loaded
from it on construction, and every create/update operation transparently persists the
new state back to it.
"""

from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from itertools import islice
import json
from pathlib import Path
from typing import Any, TypeVar
import uuid

from exa.common.errors import ConflictError, NotFoundError, ValidationError
from iqm.station_control.interface.models import (
    DutFieldData,
    DynamicQuantumArchitecture,
    ObservationData,
    ObservationDefinition,
    ObservationSetData,
    ObservationSetDefinition,
    ObservationSetType,
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
from iqm.station_control.interface.pagination import pop_deprecated_offset
from iqm.station_control.interface.storage_interface import StorageInterface

T = TypeVar("T")

_SHARED_QUERY_PARAMS = {"latest", "order_by", "limit"}


def _to_uuid(value: StrUUID) -> uuid.UUID:
    """Normalize a string-or-UUID value into a :class:`uuid.UUID`."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


def _hashable(value: Any) -> Any:
    """Convert a ``latest=`` group-by value (e.g. a list-valued field like ``tags``) into a hashable key."""
    return tuple(value) if isinstance(value, list) else value


def _observation_set_created_timestamp_in_range(obs_set: ObservationSetData, value: Any) -> bool:
    """Check the ``created_timestamp__range`` filter for ``query_observation_sets``."""
    start, end = value
    return start <= obs_set.created_timestamp <= end


_OBSERVATION_MATCHERS: dict[str, Callable[[ObservationData, Any], bool]] = {
    "observation_id": lambda obs, value: obs.observation_id == value,
    "observation_id__in": lambda obs, value: obs.observation_id in value,
    "dut_label": lambda obs, value: obs.dut_label == value,
    "dut_field": lambda obs, value: obs.dut_field == value,
    "dut_field__in": lambda obs, value: obs.dut_field in value,
    "tags__overlap": lambda obs, value: bool(set(obs.tags) & set(value)),
    "tags__contains": lambda obs, value: set(value) <= set(obs.tags),
    "invalid": lambda obs, value: obs.invalid == value,
    "source__type": lambda obs, value: obs.source.get("type") == value,
    "source__run_id__in": lambda obs, value: obs.source.get("run_id") in value,
    "source__sequence_id__in": lambda obs, value: obs.source.get("sequence_id") in value,
    "observation_set_ids__overlap": lambda obs, value: bool(
        set(obs.observation_set_ids) & {_to_uuid(v) for v in value}
    ),
    "observation_set_ids__contains": lambda obs, value: {_to_uuid(v) for v in value} <= set(obs.observation_set_ids),
}

_OBSERVATION_SET_MATCHERS: dict[str, Callable[[ObservationSetData, Any], bool]] = {
    "observation_set_id": lambda obs_set, value: obs_set.observation_set_id == _to_uuid(value),
    "observation_set_id__in": lambda obs_set, value: obs_set.observation_set_id in {_to_uuid(v) for v in value},
    "observation_set_type": lambda obs_set, value: obs_set.observation_set_type == value,
    "describes_id": lambda obs_set, value: obs_set.describes_id == _to_uuid(value),
    "describes_id__in": lambda obs_set, value: obs_set.describes_id in {_to_uuid(v) for v in value},
    "invalid": lambda obs_set, value: obs_set.invalid == value,
    "created_timestamp__range": _observation_set_created_timestamp_in_range,
    "dut_label": lambda obs_set, value: obs_set.dut_label == value,
    "dut_label__in": lambda obs_set, value: obs_set.dut_label in value,
    "end_timestamp__isnull": lambda obs_set, value: (obs_set.end_timestamp is None) == value,
}


class LocalStorage(StorageInterface):
    """Local, in-memory implementation of :class:`.StorageInterface`.

    Only observation and observation set related methods are implemented; all other
    methods raise :class:`NotImplementedError`. State lives in memory by default; pass
    ``path`` to also persist it to a JSON file, which is loaded on construction (if it
    already exists) and transparently rewritten after every create/update operation.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._observations: dict[int, ObservationData] = {}
        self._observation_sets: dict[uuid.UUID, ObservationSetData] = {}
        self._next_observation_id: int = 1
        self._path = Path(path) if path is not None else None
        if self._path is not None and self._path.exists():
            self._load_state(json.loads(self._path.read_text()))

    def _query(self, items: list[T], matchers: dict[str, Callable[[T, Any], bool]], **kwargs: Any) -> Iterator[T]:
        """Filter, group, sort and limit ``items`` according to shared query semantics.

        A filter whose value is ``None`` is treated as "not applied" (matching the convention used by
        callers such as :mod:`exa.core.database_wrapper`, which always pass every optional lookup but use
        ``None`` to mean "don't filter on this"); use the dedicated ``__isnull`` lookup to actually filter
        by a field being ``None``.

        ``latest="tags"`` is special-cased to group by ``(dut_field, tags)`` instead of by ``tags`` alone,
        matching the remote backend's behavior (see the FIXME in :func:`.load_get_latest_for_tags_cache`).
        Otherwise, every dut_field sharing the same tag set would collapse into a single observation.
        """
        pop_deprecated_offset(kwargs)
        filters = {key: value for key, value in kwargs.items() if key not in _SHARED_QUERY_PARAMS and value is not None}
        unsupported = filters.keys() - matchers.keys()
        if unsupported:
            raise ValueError(f"Unsupported query parameter(s): {', '.join(sorted(unsupported))}")
        filtered = [item for item in items if all(matchers[key](item, value) for key, value in filters.items())]

        latest = kwargs.get("latest")
        if latest is not None:
            best: dict[Any, T] = {}
            for item in filtered:
                group = _hashable(getattr(item, latest))
                if latest == "tags":
                    group = (getattr(item, "dut_field", None), group)
                if group not in best or item.created_timestamp > best[group].created_timestamp:  # type: ignore[attr-defined]
                    best[group] = item
            filtered = list(best.values())

        order_by = kwargs.get("order_by")
        field, reverse = (order_by.lstrip("-"), order_by.startswith("-")) if order_by else ("created_timestamp", False)
        filtered.sort(key=lambda item: getattr(item, field), reverse=reverse)

        limit = kwargs.get("limit")
        return iter(filtered) if limit is None or limit <= 0 else islice(filtered, limit)

    def _resolve_dut_label(
        self, observation_set_type: ObservationSetType, observation_ids: Sequence[int]
    ) -> str | None:
        """Resolve the ``dut_label`` implied by the given constituent observations."""
        if observation_set_type is ObservationSetType.GENERIC_SET:
            return None
        dut_labels = {self._get_observation(oid).dut_label for oid in observation_ids}
        if len(dut_labels) > 1:
            raise ValidationError(f"All observations in a set must link to the same DUT, got {dut_labels}.")
        return next(iter(dut_labels), None)

    def _get_observation(self, observation_id: int) -> ObservationData:
        """Look up an observation by id, raising :class:`NotFoundError` if missing."""
        try:
            return self._observations[observation_id]
        except KeyError:
            raise NotFoundError(f"Observation {observation_id} does not exist.") from None

    def _get_observation_set(self, observation_set_id: StrUUID) -> ObservationSetData:
        """Look up an observation set by id, raising :class:`NotFoundError` if missing."""
        try:
            return self._observation_sets[_to_uuid(observation_set_id)]
        except KeyError:
            raise NotFoundError(f"Observation set {observation_set_id} does not exist.") from None

    def create_observations(self, observation_definitions: Sequence[ObservationDefinition]) -> list[ObservationData]:
        """Assign each definition a new sequential id and the current timestamp, then persist if configured."""
        created = []
        for definition in observation_definitions:
            observation_id = self._next_observation_id
            self._next_observation_id += 1
            now = datetime.now(UTC)
            # Read only ObservationDefinition's own fields by name (not **model_dump()), since callers may
            # pass a subclass (e.g. LocalObservation) with extra fields of its own, like `created_timestamp`,
            # that would otherwise collide with the ones assigned below.
            observation = ObservationData(
                dut_label=definition.dut_label,
                dut_field=definition.dut_field,
                value=definition.value,
                unit=definition.unit,
                uncertainty=definition.uncertainty,
                invalid=definition.invalid,
                source=definition.source,
                tags=definition.tags,
                observation_id=observation_id,
                created_timestamp=now,
                modified_timestamp=now,
                observation_set_ids=[],
            )
            self._observations[observation_id] = observation
            created.append(observation)
        self._persist()
        return created

    def query_observations(self, **kwargs: Any) -> Iterator[ObservationData]:
        """Filter, sort and limit the in-memory observations; see :data:`_OBSERVATION_MATCHERS` for lookups."""
        return self._query(list(self._observations.values()), _OBSERVATION_MATCHERS, **kwargs)

    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        """Replace each observation in memory with a copy carrying the updated ``invalid`` flag, then persist."""
        updated = []
        for update in observation_updates:
            existing = self._get_observation(update.observation_id)
            new_observation = existing.model_copy(
                update={"invalid": update.invalid, "modified_timestamp": datetime.now(UTC)}
            )
            self._observations[update.observation_id] = new_observation
            updated.append(new_observation)
        self._persist()
        return updated

    def query_observation_sets(self, **kwargs: Any) -> Iterator[ObservationSetData]:
        """Filter, sort and limit the in-memory sets; see :data:`_OBSERVATION_SET_MATCHERS` for lookups."""
        return self._query(list(self._observation_sets.values()), _OBSERVATION_SET_MATCHERS, **kwargs)

    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        """Create the set in memory, resolve its ``dut_label`` from constituents, and persist."""
        for observation_id in observation_set_definition.observation_ids:
            self._get_observation(observation_id)
        dut_label = self._resolve_dut_label(
            observation_set_definition.observation_set_type, observation_set_definition.observation_ids
        )
        observation_set_id = uuid.uuid4()
        observation_set = ObservationSetData(
            **observation_set_definition.model_dump(),
            dut_label=dut_label,
            observation_set_id=observation_set_id,
            created_timestamp=datetime.now(UTC),
            end_timestamp=None,
        )
        self._observation_sets[observation_set_id] = observation_set
        for observation_id in observation_set_definition.observation_ids:
            observation = self._observations[observation_id]
            observation.observation_set_ids = [*observation.observation_set_ids, observation_set_id]
        self._persist()
        return observation_set

    def get_observation_set(self, observation_set_id: StrUUID) -> ObservationSetData:
        """Look up the observation set by id in memory."""
        return self._get_observation_set(observation_set_id)

    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        """Merge new observation ids into the set in memory, recompute ``dut_label``, and persist.

        Raises:
            ConflictError: If the set is already finalized and new observation ids are given.

        """
        existing = self._get_observation_set(observation_set_update.observation_set_id)
        finalized = existing.end_timestamp is not None
        new_observation_ids = observation_set_update.observation_ids or []
        if finalized and new_observation_ids:
            raise ConflictError(
                f"Observation set {existing.observation_set_id} is already finalized, cannot add observations to it."
            )

        added_ids = [oid for oid in new_observation_ids if oid not in existing.observation_ids]
        for observation_id in added_ids:
            self._get_observation(observation_id)
        combined_ids = [*existing.observation_ids, *added_ids]
        dut_label = self._resolve_dut_label(existing.observation_set_type, combined_ids)

        updated = existing.model_copy(
            update={
                "observation_ids": combined_ids,
                "dut_label": dut_label,
                "invalid": observation_set_update.invalid,
            }
        )
        self._observation_sets[updated.observation_set_id] = updated
        for observation_id in added_ids:
            observation = self._observations[observation_id]
            observation.observation_set_ids = [*observation.observation_set_ids, updated.observation_set_id]
        self._persist()
        return updated

    def finalize_observation_set(self, observation_set_id: StrUUID) -> None:
        """Set ``end_timestamp`` on the set in memory and persist; a no-op if already finalized."""
        existing = self._get_observation_set(observation_set_id)
        if existing.end_timestamp is None:
            self._observation_sets[existing.observation_set_id] = existing.model_copy(
                update={"end_timestamp": datetime.now(UTC)}
            )
            self._persist()

    def get_observation_set_observations(self, observation_set_id: StrUUID) -> list[ObservationData]:
        """Look up the set's constituent observations from the in-memory observation store."""
        observation_set = self._get_observation_set(observation_set_id)
        return [self._observations[oid] for oid in observation_set.observation_ids]

    def get_calibration_set(self, calibration_set_id: StrUUIDOrDefault) -> CalibrationSet:
        """Not implemented: LocalStorage has no concept of calibration sets."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_calibration_set'.")

    def get_dynamic_quantum_architecture(self, calibration_set_id: StrUUIDOrDefault) -> DynamicQuantumArchitecture:
        """Not implemented: LocalStorage has no concept of calibration sets."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_dynamic_quantum_architecture'.")

    def get_calibration_set_quality_metric_set(self, calibration_set_id: StrUUIDOrDefault) -> QualityMetricSet:
        """Not implemented: LocalStorage has no concept of calibration or quality metric sets."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_calibration_set_quality_metric_set'.")

    def get_chip_design_record(self, dut_label: str) -> dict[str, Any]:
        """Not implemented: LocalStorage has no concept of chip design records."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_chip_design_record'.")

    def get_run(self, run_id: StrUUID) -> RunData:
        """Not implemented: LocalStorage has no concept of runs."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_run'.")

    def query_runs(self, **kwargs: Any) -> Iterator[RunLite]:
        """Not implemented: LocalStorage has no concept of runs."""
        raise NotImplementedError("LocalStorage doesn't implement 'query_runs'.")

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        """Not implemented: LocalStorage does not track which dut fields have been observed."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_dut_fields'.")

    def query_sequence_metadatas(self, **kwargs: Any) -> Iterator[SequenceMetadataData]:
        """Not implemented: LocalStorage has no concept of sequence metadata."""
        raise NotImplementedError("LocalStorage doesn't implement 'query_sequence_metadatas'.")

    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        """Not implemented: LocalStorage has no concept of sequence metadata."""
        raise NotImplementedError("LocalStorage doesn't implement 'create_sequence_metadata'.")

    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        """Not implemented: LocalStorage has no concept of sequence results."""
        raise NotImplementedError("LocalStorage doesn't implement 'save_sequence_result'.")

    def get_sequence_result(self, sequence_id: StrUUID) -> SequenceResultData:
        """Not implemented: LocalStorage has no concept of sequence results."""
        raise NotImplementedError("LocalStorage doesn't implement 'get_sequence_result'.")

    def serialize(self) -> dict[str, Any]:
        """Serialize the current state to a dict; overrides :class:`.Serializable`.

        Not meant to be called directly: it exists to satisfy :class:`.Serializable`,
        and is used internally to implement disk persistence.
        """
        return {
            "next_observation_id": self._next_observation_id,
            "observations": [observation.model_dump(mode="json") for observation in self._observations.values()],
            "observation_sets": [
                observation_set.model_dump(mode="json") for observation_set in self._observation_sets.values()
            ],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LocalStorage":
        """Build an instance from a dict produced by :meth:`serialize`; overrides :class:`.Serializable`.

        Not meant to be called directly: it exists to satisfy :class:`.Serializable`,
        and is used internally to implement disk persistence.
        """
        storage = cls()
        storage._load_state(data)
        return storage

    def _load_state(self, data: dict[str, Any]) -> None:
        """Populate this instance's in-memory state from a dict produced by :meth:`serialize`."""
        self._next_observation_id = data["next_observation_id"]
        self._observations = {
            payload["observation_id"]: ObservationData.model_validate(payload) for payload in data["observations"]
        }
        self._observation_sets = {
            uuid.UUID(payload["observation_set_id"]): ObservationSetData.model_validate(payload)
            for payload in data["observation_sets"]
        }

    def _persist(self) -> None:
        """Write the current state to the configured path, if any."""
        if self._path is not None:
            self._path.write_text(json.dumps(self.serialize(), indent=2))
