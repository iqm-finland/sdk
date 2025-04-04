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
import warnings


class APIEndpoint(Enum):
    """Supported API endpoints."""

    CONFIGURATION = auto()
    SUBMIT_JOB = auto()
    GET_JOB_REQUEST_PARAMETERS = auto()
    GET_JOB_RESULT = auto()
    GET_JOB_STATUS = auto()
    GET_JOB_CALIBRATION_SET_ID = auto()
    GET_JOB_CIRCUITS_BATCH = auto()
    GET_JOB_TIMELINE = auto()
    GET_JOB_ERROR_LOG = auto()
    GET_JOB_COUNTS = auto()
    ABORT_JOB = auto()
    DELETE_JOB = auto()
    HEALTH = auto()
    ABOUT = auto()
    CLIENT_LIBRARIES = auto()

    # Calibration and Calibration Service endpoints
    CALIBRATION_SERVICE_CONFIGURATION = auto()
    QUANTUM_ARCHITECTURE = auto()
    STATIC_QUANTUM_ARCHITECTURE = auto()
    CHANNEL_PROPERTIES = auto()
    QUALITY_METRICS_LATEST = auto()
    QUALITY_METRICS = auto()
    QUALITY_METRICS_MONITORING = auto()
    CALIBRATED_GATES = auto()
    START_CALIBRATION_JOB = auto()
    ABORT_CALIBRATION_JOB = auto()
    CALIBRATION_SERVICE_JOBS = auto()
    CALIBRATION = auto()


class APIVariant(Enum):
    """Supported API versions and variants."""

    V1 = "V1"  # QCCSW Cocos-based circuits execution
    V2 = "V2"  # Station-Control-based circuits execution
    RESONANCE_COCOS_V1 = "RESONANCE_COCOS_V1"  # IQM Resonance Cocos API


class APIConfig:
    """Provides supported API endpoints for a given API variant."""

    def __init__(self, variant: APIVariant, iqm_server_url: str):
        """Args:
        variant: API variant.
        iqm_server_url: URL of the IQM server,
                        e.g. https://test.qc.iqm.fi/cocos or https://cocos.resonance.meetiqm.com/garnet for .V1
                        or https://test.qc.iqm.fi for .V2

        """
        self.variant = variant

        # V1 API is deprecated and will be removed in a future release.
        # V2 API is supported by the QCCSW starting with version 3.3.0.
        if self.variant == APIVariant.V1:
            warnings.warn(
                "The V1 API is deprecated and will be removed in a future release. Please use the V2 API instead.",
                DeprecationWarning,
            )

        self.iqm_server_url = iqm_server_url
        self.urls = self._get_api_urls()

    def _get_api_urls(self) -> dict[APIEndpoint, str]:
        """Returns:
        Relative URLs for each supported API endpoints.

        """
        if self.variant == APIVariant.RESONANCE_COCOS_V1:
            return {
                APIEndpoint.SUBMIT_JOB: "jobs",
                APIEndpoint.GET_JOB_RESULT: "jobs/%s",
                APIEndpoint.GET_JOB_STATUS: "jobs/%s/status",
                APIEndpoint.GET_JOB_COUNTS: "jobs/%s/counts",
                APIEndpoint.ABORT_JOB: "jobs/%s/abort",
                APIEndpoint.QUANTUM_ARCHITECTURE: "quantum-architecture",
                APIEndpoint.CALIBRATED_GATES: "api/v1/calibration/%s/gates",
                APIEndpoint.CLIENT_LIBRARIES: "info/client-libraries",
            }
        if self.variant == APIVariant.V1:
            return {
                APIEndpoint.CONFIGURATION: "configuration",
                APIEndpoint.QUALITY_METRICS_LATEST: "calibration/metrics/latest",
                APIEndpoint.QUALITY_METRICS: "calibration/metrics/%s",
                APIEndpoint.SUBMIT_JOB: "jobs",
                APIEndpoint.GET_JOB_RESULT: "jobs/%s",
                APIEndpoint.GET_JOB_STATUS: "jobs/%s/status",
                APIEndpoint.GET_JOB_COUNTS: "jobs/%s/counts",
                APIEndpoint.ABORT_JOB: "jobs/%s/abort",
                APIEndpoint.ABORT_CALIBRATION_JOB: "jobs/%s/abort",
                APIEndpoint.DELETE_JOB: "jobs/%s",
                APIEndpoint.QUANTUM_ARCHITECTURE: "quantum-architecture",
                APIEndpoint.STATIC_QUANTUM_ARCHITECTURE: "api/v1/quantum-architecture",
                APIEndpoint.CALIBRATED_GATES: "api/v1/calibration/%s/gates",
                APIEndpoint.QUALITY_METRICS_MONITORING: "api/v1/monitor/calibration/metrics",
                APIEndpoint.HEALTH: "health",
                APIEndpoint.ABOUT: "about",
                APIEndpoint.CLIENT_LIBRARIES: "info/client-libraries",
                APIEndpoint.START_CALIBRATION_JOB: "calibration/run",
                APIEndpoint.CALIBRATION_SERVICE_CONFIGURATION: "calibration/configuration",
                APIEndpoint.CALIBRATION_SERVICE_JOBS: "calibration/jobs",
                APIEndpoint.CALIBRATION: "api/v1/calibration/%s",
            }
        if self.variant == APIVariant.V2:
            return {
                # TODO SW-1387: Use station/v1 API
                APIEndpoint.GET_JOB_REQUEST_PARAMETERS: "station/circuits/%s/request_parameters",
                APIEndpoint.CONFIGURATION: "cocos/configuration",
                APIEndpoint.QUALITY_METRICS_LATEST: "cocos/calibration/metrics/latest",
                APIEndpoint.QUALITY_METRICS: "cocos/calibration/metrics/%s",
                APIEndpoint.SUBMIT_JOB: "station/circuits",
                APIEndpoint.GET_JOB_RESULT: "station/circuits/%s/measurements",
                APIEndpoint.GET_JOB_STATUS: "station/circuits/%s/status",
                APIEndpoint.GET_JOB_CALIBRATION_SET_ID: "station/circuits/%s/calibration_set_id",
                APIEndpoint.GET_JOB_CIRCUITS_BATCH: "station/circuits/%s/circuits_batch",
                APIEndpoint.GET_JOB_TIMELINE: "station/circuits/%s/timeline",
                APIEndpoint.GET_JOB_ERROR_LOG: "station/circuits/%s/error_log",
                APIEndpoint.GET_JOB_COUNTS: "station/circuits/%s/counts",
                APIEndpoint.ABORT_JOB: "station/circuits/%s/abort",
                APIEndpoint.ABORT_CALIBRATION_JOB: "cocos/jobs/%s/abort",
                APIEndpoint.DELETE_JOB: "station/circuits/%s",
                APIEndpoint.QUANTUM_ARCHITECTURE: "cocos/quantum-architecture",
                APIEndpoint.STATIC_QUANTUM_ARCHITECTURE: "cocos/api/v1/quantum-architecture",
                APIEndpoint.CHANNEL_PROPERTIES: "station/channel-properties",
                APIEndpoint.CALIBRATED_GATES: "cocos/api/v1/calibration/%s/gates",
                APIEndpoint.QUALITY_METRICS_MONITORING: "cocos/api/v1/monitor/calibration/metrics",
                APIEndpoint.HEALTH: "cocos/health",
                APIEndpoint.ABOUT: "cocos/about",
                APIEndpoint.CLIENT_LIBRARIES: "info/client-libraries",
                APIEndpoint.START_CALIBRATION_JOB: "cocos/calibration/run",
                APIEndpoint.CALIBRATION_SERVICE_CONFIGURATION: "cocos/calibration/configuration",
                APIEndpoint.CALIBRATION_SERVICE_JOBS: "cocos/calibration/jobs",
                APIEndpoint.CALIBRATION: "cocos/api/v1/calibration/%s",
            }
        raise ValueError(f"Unsupported API variant: {self.variant}")

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
        url = self.urls.get(endpoint)
        if url is None:
            raise ValueError(f"Unsupported API endpoint: {endpoint}")
        return join(self.iqm_server_url, url % args)
