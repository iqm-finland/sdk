# Copyright 2024-2025 IQM
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

"""Provider of calibration sets and quality metrics from remote IQM servers."""

from copy import deepcopy
import logging
import uuid

from iqm.pulla.interface import CalibrationSet, CalibrationSetId
from iqm.pulla.utils import calset_from_observations
from iqm.station_control.client.iqm_server.iqm_server_client import IqmServerClient
from iqm.station_control.interface.models import ObservationSetData
from iqm.station_control.interface.station_control import StationControlInterface

logger = logging.getLogger(__name__)

CalibrationDataFetchException = RuntimeError


class CalibrationDataProvider:
    """Access calibration info via station control client and cache data in memory."""

    def __init__(self, station_control: StationControlInterface):
        self._station_control = station_control
        self._calibration_sets: dict[CalibrationSetId, CalibrationSet] = {}

    def get_calibration_set(self, cal_set_id: CalibrationSetId) -> CalibrationSet:
        """Get the calibration set contents from the database and cache it."""
        logger.debug("Get the calibration set from the database: cal_set_id=%s", cal_set_id)
        try:
            if cal_set_id not in self._calibration_sets:
                cal_set_values = self.get_calibration_set_values(cal_set_id)
                self._calibration_sets[cal_set_id] = cal_set_values
            return deepcopy(self._calibration_sets[cal_set_id])
        except Exception as e:
            raise CalibrationDataFetchException("Could not fetch calibration set from the database.") from e

    def get_latest_calibration_set(self, chip_label: str) -> tuple[CalibrationSet, CalibrationSetId]:
        """Get the latest calibration set id for ``chip_label`` from the database, return it and the set contents."""
        logger.debug("Get the latest calibration set for chip label: chip_label=%s", chip_label)
        try:
            if isinstance(self._station_control, IqmServerClient):
                latest_cal_set_id = self._station_control.get_latest_calibration_set_id(chip_label)
            else:
                latest_calibration_set = self._get_latest_calibration_set(chip_label)
                latest_cal_set_id = latest_calibration_set.observation_set_id
            calset = self.get_calibration_set(latest_cal_set_id)
        except Exception as e:
            raise CalibrationDataFetchException(
                f"Could not fetch latest calibration set id from the database: {e}"
            ) from e
        return calset, latest_cal_set_id

    def _get_latest_calibration_set(self, dut_label: str) -> ObservationSetData:
        observation_sets = self._station_control.query_observation_sets(
            observation_set_type="calibration-set",
            dut_label=dut_label,
            invalid=False,
            end_timestamp__isnull=False,  # Finalized
            order_by="-end_timestamp",
            limit=1,
        )
        return observation_sets[0]

    def get_calibration_set_values(self, calibration_set_id: uuid.UUID) -> CalibrationSet:
        """Get saved calibration set observations by UUID.

        Args:
            calibration_set_id: UUID of the calibration set to retrieve.

        Returns:
            Dictionary of observations belonging to the given calibration set.

        """
        if isinstance(self._station_control, IqmServerClient):
            calibration_set_values = self._station_control.get_calibration_set_values(calibration_set_id)
        else:
            observation_set = self._station_control.get_observation_set(calibration_set_id)
            if observation_set.observation_set_type != "calibration-set":
                raise ValueError("Observation set type is not 'calibration-set'")
            observations = self._station_control.get_observation_set_observations(calibration_set_id)
            calibration_set_values = calset_from_observations(observations)
        return calibration_set_values
