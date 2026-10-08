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
"""Implementation of the StorageInterface for remote IQM Server communication.

This module provides the :class:`IQMServerStorage` class, which implements the
standardized storage interface via the IQM Server's REST API.
"""

from collections.abc import Sequence
from functools import cache
from typing import Any, cast

import requests

from exa.common.errors.iqm_error import IQMError, NotFoundError, ValidationError
from iqm.station_control.client.list_models import (
    DutFieldDataList,
    ObservationDataList,
    ObservationDefinitionList,
    ObservationLiteList,
    ObservationSetDataList,
    ObservationUpdateList,
    RunLiteList,
    SequenceMetadataDataList,
)
from iqm.station_control.client.serializers import deserialize_run_data
from iqm.station_control.client.station_control_client import StationControlClient
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
from iqm.station_control.interface.models.observation_set import (
    CalibrationSet,
    ObservationSetType,
    QualityMetrics,
    QualityMetricSet,
)
from iqm.station_control.interface.models.type_aliases import StrUUIDOrDefault
from iqm.station_control.interface.storage_interface import StorageInterface


# ruff: noqa: D102
class StationControlStorage(StorageInterface):
    """Concrete implementation of the StorageInterface using an IQM Server backend."""

    def __init__(self, client: StationControlClient):
        self.client = client

    @cache
    def get_chip_design_record(self, dut_label: str) -> dict[str, Any]:
        try:
            response = self.client.send_request(requests.get, f"chip-design-records/{dut_label}")
        except IQMError as err:
            # Enable clients to specify their own QCM connection,
            # in case they are hooked up to a station using a single CHEDDAR file
            # but want to load the CHEDDAR of some other DUT.
            if isinstance(err, NotFoundError) and self.client._qcm_data_client:
                return self.client._qcm_data_client.get_chip_design_record(dut_label)
            raise err
        return response.json()

    def get_run(self, run_id: StrUUID) -> RunData:
        response = self.client.send_request(requests.get, f"runs/{run_id}")
        return deserialize_run_data(response.json())

    def query_runs(self, **kwargs: Any) -> ListWithMeta[RunLite]:
        params = self.client.clean_query_parameters(RunData, **kwargs)
        response = self.client.send_request(requests.get, "runs", params=params)
        return self.client.deserialize_response(response, RunLiteList, list_with_meta=True)

    def create_observations(
        self, observation_definitions: Sequence[ObservationDefinition]
    ) -> ListWithMeta[ObservationData]:
        json_data = self.client.serialize_model(ObservationDefinitionList(list(observation_definitions)))
        response = self.client.send_request(requests.post, "observations", json_data=json_data, timeout=1200)
        return self.client.deserialize_response(response, ObservationDataList, list_with_meta=True)

    def query_observations(self, **kwargs: Any) -> ListWithMeta[ObservationData]:
        params = self.client.clean_query_parameters(ObservationData, **kwargs)
        # TODO (Marko): Handle old logic temporarily on the client side
        if {"dut_label", "tags__overlap", "latest"} <= params.keys() and params["latest"] == "tags":
            params["tags"] = params.pop("tags__overlap")
            params["mode"] = "tags_or"
            params.pop("latest")
            list_with_meta = False
        else:
            list_with_meta = True
        response = self.client.send_request(requests.get, "observations", params=params)
        return self.client.deserialize_response(response, ObservationDataList, list_with_meta=list_with_meta)

    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        json_data = self.client.serialize_model(ObservationUpdateList(list(observation_updates)))
        response = self.client.send_request(requests.patch, "observations", json_data=json_data)
        return self.client.deserialize_response(response, ObservationDataList)

    def query_observation_sets(self, **kwargs: Any) -> ListWithMeta[ObservationSetData]:
        params = self.client.clean_query_parameters(ObservationSetData, **kwargs)
        response = self.client.send_request(requests.get, "observation-sets", params=params)
        return self.client.deserialize_response(response, ObservationSetDataList, list_with_meta=True)

    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        json_data = self.client.serialize_model(observation_set_definition)
        response = self.client.send_request(requests.post, "observation-sets", json_data=json_data)
        return self.client.deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def get_observation_set(self, observation_set_id: StrUUID) -> ObservationSetData:
        response = self.client.send_request(requests.get, f"observation-sets/{observation_set_id}")
        return self.client.deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        json_data = self.client.serialize_model(observation_set_update)
        response = self.client.send_request(requests.patch, "observation-sets", json_data=json_data)
        return self.client.deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def finalize_observation_set(self, observation_set_id: StrUUID) -> None:
        self.client.send_request(requests.post, f"observation-sets/{observation_set_id}/finalize")

    def get_observation_set_observations(self, observation_set_id: StrUUID) -> list[ObservationData]:
        observation_set = self.get_observation_set(observation_set_id)
        response = self.client.send_request(requests.get, f"observation-sets/{observation_set_id}/observations")
        observations = self.client.deserialize_response(response, ObservationLiteList)
        return [
            ObservationData.from_observation_lite(observation, dut_label=observation_set.dut_label)
            for observation in observations
        ]

    def get_calibration_set(self, calibration_set_id: StrUUIDOrDefault) -> CalibrationSet:
        if calibration_set_id == "default":
            response = self.client.send_request(requests.get, "calibration-sets/default")
            observation_set = cast(ObservationSetData, self.client.deserialize_response(response, ObservationSetData))
        else:
            response = self.client.send_request(requests.get, f"observation-sets/{calibration_set_id}")
            observation_set = cast(ObservationSetData, self.client.deserialize_response(response, ObservationSetData))
            if observation_set.observation_set_type is not ObservationSetType.CALIBRATION_SET:
                raise ValidationError("Given 'calibration_set_id' doesn't belong to calibration set.")
        response = self.client.send_request(
            requests.get, f"observation-sets/{observation_set.observation_set_id}/observations"
        )
        observations = self.client.deserialize_response(response, ObservationLiteList)
        return CalibrationSet(
            **observation_set.model_dump(),
            observations=[
                ObservationData.from_observation_lite(observation, dut_label=observation_set.dut_label)
                for observation in observations
            ],
        )

    def get_dynamic_quantum_architecture(self, calibration_set_id: StrUUIDOrDefault) -> DynamicQuantumArchitecture:
        response = self.client.send_request(
            requests.get, f"calibration-sets/{calibration_set_id}/dynamic-quantum-architecture"
        )
        return self.client.deserialize_response(response, DynamicQuantumArchitecture)  # type: ignore[return-value]

    def get_calibration_set_quality_metric_set(self, calibration_set_id: StrUUIDOrDefault) -> QualityMetricSet:
        response = self.client.send_request(requests.get, f"calibration-sets/{calibration_set_id}/metrics")
        quality_metrics = cast(QualityMetrics, self.client.deserialize_response(response, QualityMetrics))
        observations = [
            ObservationData.from_observation_lite(observation, dut_label=quality_metrics.dut_label)
            for observation in quality_metrics.observations
        ]
        return QualityMetricSet(**quality_metrics.model_dump(exclude={"observations"}), observations=observations)

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        params = {"dut_label": dut_label}
        response = self.client.send_request(requests.get, "dut-fields", params=params)
        return self.client.deserialize_response(response, DutFieldDataList)

    def query_sequence_metadatas(self, **kwargs: Any) -> ListWithMeta[SequenceMetadataData]:
        params = self.client.clean_query_parameters(SequenceMetadataData, **kwargs)
        response = self.client.send_request(requests.get, "sequence-metadatas", params=params)
        return self.client.deserialize_response(response, SequenceMetadataDataList, list_with_meta=True)

    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        json_data = self.client.serialize_model(sequence_metadata_definition)
        response = self.client.send_request(requests.post, "sequence-metadatas", json_data=json_data)
        return self.client.deserialize_response(response, SequenceMetadataData)  # type: ignore[return-value]

    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        # FIXME: We don't have information if the object was created or updated. Thus, server always responds 200 (OK).
        json_data = self.client.serialize_model(sequence_result_definition)
        response = self.client.send_request(
            requests.put, f"sequence-results/{sequence_result_definition.sequence_id}", json_data=json_data
        )
        return self.client.deserialize_response(response, SequenceResultData)  # type: ignore[return-value]

    def get_sequence_result(self, sequence_id: StrUUID) -> SequenceResultData:
        response = self.client.send_request(requests.get, f"sequence-results/{sequence_id}")
        return self.client.deserialize_response(response, SequenceResultData)  # type: ignore[return-value]

    def serialize(self) -> dict[str, Any]:
        return {
            "client": self.client.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "StationControlStorage":
        return cls(
            client=StationControlClient.deserialize(data["client"]),
        )
