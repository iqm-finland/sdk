# Copyright 2024 IQM client developers
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
"""This module contains definitions of IQM Server API endpoints."""

from enum import Enum, auto
from posixpath import join


class APIEndpoint(Enum):
    """Supported API endpoints."""

    GET_JOB_RESULT = auto()
    GET_JOB_REQUEST_PARAMETERS = auto()
    GET_JOB_CALIBRATION_SET_ID = auto()
    GET_JOB_CIRCUITS_BATCH = auto()
    GET_JOB_ERROR_LOG = auto()
    SUBMIT_JOB = auto()
    GET_JOB_COUNTS = auto()
    GET_JOB_STATUS = auto()
    GET_JOB_TIMELINE = auto()
    ABORT_JOB = auto()


class APIConfig:
    """Provides supported API endpoints for a given API variant."""

    def __init__(self, station_control_url: str):
        """Args:
        station_control_url: URL of the IQM server,
            e.g. https://test.qc.iqm.fi/station or https://cocos.resonance.meetiqm.com/garnet

        """
        self.station_control_url = station_control_url
        self.urls = self._get_api_urls()

    @staticmethod
    def _get_api_urls() -> dict[APIEndpoint, str]:
        """Returns:
        Relative URLs for each supported API endpoints.

        """
        return {
            # TODO SW-1434: Use StationControlClient methods for communication instead of REST endpoints
            APIEndpoint.GET_JOB_RESULT: "jobs/%s/measurements",
            APIEndpoint.GET_JOB_REQUEST_PARAMETERS: "jobs/%s/request_parameters",
            APIEndpoint.GET_JOB_CALIBRATION_SET_ID: "jobs/%s/calibration_set_id",
            APIEndpoint.GET_JOB_CIRCUITS_BATCH: "jobs/%s/circuits_batch",
            APIEndpoint.GET_JOB_ERROR_LOG: "jobs/%s/error_log",
            APIEndpoint.SUBMIT_JOB: "circuits",
            APIEndpoint.GET_JOB_COUNTS: "circuits/%s/counts",
            APIEndpoint.GET_JOB_STATUS: "jobs/%s/status",
            APIEndpoint.GET_JOB_TIMELINE: "jobs/%s/timeline",
            APIEndpoint.ABORT_JOB: "jobs/%s/abort",
        }

    def is_supported(self, endpoint: APIEndpoint) -> bool:
        """Args:
            endpoint: API endpoint.

        Returns:
            True if the endpoint is supported, False otherwise.

        """
        return endpoint in self.urls

    def url(self, endpoint: APIEndpoint, *args) -> str:
        """Args:
            endpoint: API endpoint.
            args: Arguments to be passed to the URL.

        Returns:
            URL for the given endpoint.

        Raises:
            ValueError: If the endpoint is not supported.

        """
        url = self.urls.get(endpoint, "")
        return join(self.station_control_url, url % args)
