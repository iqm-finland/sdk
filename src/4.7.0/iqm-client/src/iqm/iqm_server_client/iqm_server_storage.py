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
from typing import Any

from iqm.iqm_server_client.iqm_server_client import IQMServerClient
import requests

from exa.common.errors.iqm_error import NotFoundError
from iqm.station_control.client.list_models import (
    DutFieldDataList,
    ObservationDataList,
    ObservationSetDataList,
    ObservationUpdateList,
    SequenceMetadataDataList,
)
from iqm.station_control.interface.list_with_meta import ListWithMeta
from iqm.station_control.interface.models import (
    DutFieldData,
    DynamicQuantumArchitecture,
    ObservationData,
    ObservationDefinition,
    ObservationLite,
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
from iqm.station_control.interface.storage_interface import StorageInterface


def _is_observation_data_payload(payload: dict[str, Any]) -> bool:
    """Detect whether a raw observation payload already carries full ObservationData fields."""
    return "dut_label" in payload


def _parse_observations(payloads: list[dict[str, Any]], dut_label: str | None) -> list[ObservationData]:
    """Parse raw observation payloads, upgrading legacy ObservationLite-shaped entries to ObservationData.

    NOTE: once IQM Server starts serving ObservationData format observations, this can be removed.
    """
    observations = []
    for payload in payloads:
        if _is_observation_data_payload(payload):
            observations.append(ObservationData.model_validate(payload))
        else:
            observations.append(
                ObservationData.from_observation_lite(ObservationLite.model_validate(payload), dut_label=dut_label)
            )
    return observations


# ruff: noqa: D102
class IQMServerStorage(StorageInterface):
    """Concrete implementation of the StorageInterface using an IQM Server backend."""

    def __init__(self, client: IQMServerClient):
        self.client = client

    @cache
    def get_chip_design_record(self, dut_label: str) -> dict[str, Any]:
        response = self.client.send_request(
            requests.get, f"quantum-computers/{self.client.quantum_computer}/artifacts/chip-design-records"
        )

        if response.json()[0]["cheddar"]["chip_label"] == dut_label:
            return response.json()[0]
        elif self.client._qcm_data_client is not None:
            return self.client._qcm_data_client.get_chip_design_record(dut_label)
        else:
            raise NotFoundError(f"Chip design record for DUT '{dut_label}' not found.")

    def get_run(self, run_id: StrUUID) -> RunData:
        raise NotImplementedError("IQMServerStorage doesn't implement 'get_run' yet.")

    def query_runs(self, **kwargs: Any) -> ListWithMeta[RunLite]:
        raise NotImplementedError("IQMServerStorage doesn't implement 'query_runs' yet.")

    def create_observations(
        self, observation_definitions: Sequence[ObservationDefinition]
    ) -> ListWithMeta[ObservationData]:
        """Not yet supported.

        Since `IqmServerExecutor` does not proxy the job sumbission, observations also can't be saved,
        due to there not being any job in station-control to link them to.
        """
        raise NotImplementedError("Saving observations is not yet supported on IQM Server.")

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
        response = self.client.send_request(requests.get, f"{self.client.proxy_path}/observations", params=params)
        return self.client.deserialize_response(response, ObservationDataList, list_with_meta=list_with_meta)

    def update_observations(self, observation_updates: Sequence[ObservationUpdate]) -> list[ObservationData]:
        json_data = self.client.serialize_model(ObservationUpdateList(list(observation_updates)))
        response = self.client.send_request(
            requests.patch, f"{self.client.proxy_path}/observations", json_data=json_data
        )
        return self.client.deserialize_response(response, ObservationDataList)

    def query_observation_sets(self, **kwargs: Any) -> ListWithMeta[ObservationSetData]:
        params = self.client.clean_query_parameters(ObservationSetData, **kwargs)
        response = self.client.send_request(requests.get, f"{self.client.proxy_path}/observation-sets", params=params)
        return self.client.deserialize_response(response, ObservationSetDataList, list_with_meta=True)

    def create_observation_set(self, observation_set_definition: ObservationSetDefinition) -> ObservationSetData:
        json_data = self.client.serialize_model(observation_set_definition)
        response = self.client.send_request(
            requests.post, f"{self.client.proxy_path}/observation-sets", json_data=json_data
        )
        return self.client.deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def get_observation_set(self, observation_set_id: StrUUID) -> ObservationSetData:
        response = self.client.send_request(
            requests.get, f"{self.client.proxy_path}/observation-sets/{observation_set_id}"
        )
        return self.client.deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def update_observation_set(self, observation_set_update: ObservationSetUpdate) -> ObservationSetData:
        json_data = self.client.serialize_model(observation_set_update)
        response = self.client.send_request(
            requests.patch, f"{self.client.proxy_path}/observation-sets", json_data=json_data
        )
        return self.client.deserialize_response(response, ObservationSetData)  # type: ignore[return-value]

    def finalize_observation_set(self, observation_set_id: StrUUID) -> None:
        self.client.send_request(
            requests.post, f"{self.client.proxy_path}/observation-sets/{observation_set_id}/finalize"
        )

    def get_observation_set_observations(self, observation_set_id: StrUUID) -> list[ObservationData]:
        observation_set = self.get_observation_set(observation_set_id)
        response = self.client.send_request(
            requests.get, f"{self.client.proxy_path}/observation-sets/{observation_set_id}/observations"
        )
        return _parse_observations(response.json(), dut_label=observation_set.dut_label)

    def get_calibration_set(self, calibration_set_id: StrUUIDOrDefault) -> CalibrationSet:
        response = self.client.send_request(
            requests.get, f"calibration-sets/{self.client.quantum_computer}/{calibration_set_id}"
        )
        payload = response.json()
        observations = _parse_observations(payload.get("observations", []), dut_label=payload.get("dut_label"))
        return CalibrationSet.model_validate({**payload, "observations": observations})

    def get_dynamic_quantum_architecture(self, calibration_set_id: StrUUIDOrDefault) -> DynamicQuantumArchitecture:
        response = self.client.send_request(
            requests.get,
            f"calibration-sets/{self.client.quantum_computer}/{calibration_set_id}/dynamic-quantum-architecture",
        )
        return self.client.deserialize_response(response, DynamicQuantumArchitecture)  # type: ignore[return-value]

    def get_calibration_set_quality_metric_set(self, calibration_set_id: StrUUIDOrDefault) -> QualityMetricSet:
        response = self.client.send_request(
            requests.get,
            f"calibration-sets/{self.client.quantum_computer}/{calibration_set_id}/metrics",
        )
        payload = response.json()
        observations = _parse_observations(payload.get("observations", []), dut_label=payload.get("dut_label"))
        return QualityMetricSet.model_validate({**payload, "observations": observations})

    def get_dut_fields(self, dut_label: str) -> list[DutFieldData]:
        params = {"dut_label": dut_label}
        response = self.client.send_request(requests.get, f"{self.client.proxy_path}/dut-fields", params=params)
        return self.client.deserialize_response(response, DutFieldDataList)

    def query_sequence_metadatas(self, **kwargs: Any) -> ListWithMeta[SequenceMetadataData]:
        params = self.client.clean_query_parameters(SequenceMetadataData, **kwargs)
        response = self.client.send_request(requests.get, f"{self.client.proxy_path}/sequence-metadatas", params=params)
        return self.client.deserialize_response(response, SequenceMetadataDataList, list_with_meta=True)

    def create_sequence_metadata(
        self, sequence_metadata_definition: SequenceMetadataDefinition
    ) -> SequenceMetadataData:
        json_data = self.client.serialize_model(sequence_metadata_definition)
        response = self.client.send_request(
            requests.post, f"{self.client.proxy_path}/sequence-metadatas", json_data=json_data
        )
        return self.client.deserialize_response(response, SequenceMetadataData)  # type: ignore[return-value]

    def save_sequence_result(self, sequence_result_definition: SequenceResultDefinition) -> SequenceResultData:
        # FIXME: We don't have information if the object was created or updated. Thus, server always responds 200 (OK).
        json_data = self.client.serialize_model(sequence_result_definition)
        response = self.client.send_request(
            requests.put,
            f"{self.client.proxy_path}/sequence-results/{sequence_result_definition.sequence_id}",
            json_data=json_data,
        )
        return self.client.deserialize_response(response, SequenceResultData)  # type: ignore[return-value]

    def get_sequence_result(self, sequence_id: StrUUID) -> SequenceResultData:
        response = self.client.send_request(requests.get, f"{self.client.proxy_path}/sequence-results/{sequence_id}")
        return self.client.deserialize_response(response, SequenceResultData)  # type: ignore[return-value]

    def serialize(self) -> dict[str, Any]:
        return {
            "client": self.client.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "IQMServerStorage":
        return cls(
            client=IQMServerClient.deserialize(data["client"]),
        )
