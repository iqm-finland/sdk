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
from uuid import UUID

from iqm.iqm_server_client.iqm_server_client import _IQMServerClient

from iqm.pulla.interface import CalibrationSetValues
from iqm.pulla.utils import calset_from_observations
from iqm.station_control.interface.models import StrUUID

logger = logging.getLogger(__name__)

CalibrationDataFetchException = RuntimeError


class CalibrationDataProvider:
    """Access calibration info via IQM Server and cache data in memory."""

    def __init__(self, iqm_server_client: _IQMServerClient):
        self._iqm_server_client = iqm_server_client
        self._calibration_sets: dict[UUID, CalibrationSetValues] = {}

    def get_calibration_set_values(self, calibration_set_id: UUID) -> CalibrationSetValues:
        """Get the calibration set contents from the database and cache it."""
        logger.debug("Get the calibration set from the database: cal_set_id=%s", calibration_set_id)
        try:
            if calibration_set_id not in self._calibration_sets:
                cal_set_values = self._get_calibration_set_values(calibration_set_id)
                self._calibration_sets[calibration_set_id] = cal_set_values
            return deepcopy(self._calibration_sets[calibration_set_id])
        except Exception as e:
            raise CalibrationDataFetchException("Could not fetch calibration set from the database.") from e

    def get_default_calibration_set(self) -> tuple[CalibrationSetValues, UUID]:
        """Get the default calibration set id from the database, return it and the set contents."""
        logger.debug("Get the default calibration set")
        try:
            default_calibration_set = self._iqm_server_client.get_calibration_set("default")
            default_calibration_set_values = self.get_calibration_set_values(default_calibration_set.observation_set_id)
        except Exception as e:
            raise CalibrationDataFetchException(
                f"Could not fetch default calibration set id from the database: {e}"
            ) from e
        return default_calibration_set_values, default_calibration_set.observation_set_id

    def _get_calibration_set_values(self, calibration_set_id: StrUUID) -> CalibrationSetValues:
        """Get saved calibration set observations by UUID.

        Args:
            calibration_set_id: UUID of the calibration set to retrieve or "default".

        Returns:
            Dictionary of observations belonging to the given calibration set.

        """
        calibration_set = self._iqm_server_client.get_calibration_set(calibration_set_id)
        calibration_set_values = calset_from_observations(calibration_set.observations)
        return calibration_set_values
