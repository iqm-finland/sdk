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

"""QCM (Quantum Computer Management) Data API client implementation."""

from collections.abc import Callable
from functools import cache
import json
import logging
from pathlib import Path
from typing import Any

from packaging.version import Version
import requests

from exa.common.errors.exa_error import ExaError
from exa.common.errors.station_control_errors import NotFoundError, ValidationError
from exa.common.qcm_data.file_adapter import FileAdapter

MIN_SUPPORTED_CONTENT_FORMAT_VERSION = Version("1.0")
MAX_SUPPORTED_CONTENT_FORMAT_VERSION = Version("2")  # Only major is checked for max version. Minor has no effect.


class QCMDataClient:
    """Python client for QCM (Quantum Computer Management) Data API.

    Args:
        root_url: URL pointing to QCM Data service.
            This URL can point to a local file storage as well.
            In that case, the URL should point to a directory which
            has a directory structure identical to QCM Data service (for example /chip-data-records/),
            and files containing data in identical format returned by QCM Data service.
            For example, CHAD files should be named {chip_label}.json, like M156_W531_A09_L09.json, and contain
            a list instead of a single object.
        fallback_root_url: Same as `root_url`, used if a query via `root_url` returns nothing.

    """

    def __init__(self, root_url: str, fallback_root_url: str = ""):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("Initialize QCMDataClient: root_url=%s", root_url)

        if not root_url:
            raise ValueError("QCMDataClient 'root_url' cannot be empty, it must be a valid HTTP or file URL.")
        self._root_url = root_url
        self._fallback_root_url = fallback_root_url

        self.session = requests.Session()
        self.session.mount(str(Path("file://")), FileAdapter())

        # Make the cache containers local to the instances so that the reference from cache to the instance
        # gets scraped off with the instance
        # https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
        self._send_request = cache(self._send_request)  # type:ignore[method-assign]

    @property
    def root_url(self) -> str:
        """Returns the remote QCM Data service URL."""
        return self._root_url

    @root_url.setter
    def root_url(self, root_url: str) -> None:
        """Sets the remote QCM Data service URL."""
        if self._root_url != root_url:
            self._root_url = root_url

    def get_chip_design_record(self, chip_label: str) -> dict:
        """Get a raw chip design record matching the given chip label.

        Args:
            chip_label: Chip label.

        Returns:
            Data record matching the given chip label.

        """
        url_tail = f"/cheddars/{chip_label}?target=in-house"
        try:
            response = self._send_request(self.session.get, f"{self.root_url}{url_tail}")
        except NotFoundError as err:
            if self._fallback_root_url:
                response = self._send_request(self.session.get, f"{self._fallback_root_url}{url_tail}")
            else:
                raise err
        data = response.json().get("data")
        if not data:
            raise ValidationError(f"Chip design record for {chip_label} does not contain data.")
        self._validate_chip_design_record(data, chip_label)
        return data

    def _send_request(self, http_method: Callable[..., requests.Response], url: str) -> requests.Response:
        # Send the request and return the response.
        # Will raise an error if respectively an error response code is returned.
        # http_method should be any of session.[post|get|put|head|delete|patch|options]
        self.logger.debug("http_method=%s, url=%s", http_method.__name__, url)
        response = http_method(url, timeout=20)
        if not response.ok:
            try:
                response_dict = response.json()
                error_message = response_dict["detail"]
            except json.JSONDecodeError:
                error_message = response.text
            raise NotFoundError(error_message)
        return response

    @staticmethod
    def _validate_chip_design_record(chip_design_record: dict[str, Any], chip_label: str) -> None:
        chip_label_mask_set_name, _, chip_label_variant, _ = chip_label.split("_")

        content_format_version = Version(chip_design_record["content_format_version"])
        chad_mask_set_name = chip_design_record["mask_set_name"]
        chad_variant = chip_design_record["variant"]

        # Allow any release in version range.
        if (
            content_format_version < MIN_SUPPORTED_CONTENT_FORMAT_VERSION
            or content_format_version.major > MAX_SUPPORTED_CONTENT_FORMAT_VERSION.major
        ):
            raise ExaError(f"CHAD content format version '{content_format_version.public}' is not supported.")

        if chad_mask_set_name != chip_label_mask_set_name:
            raise ExaError(
                f"CHAD mask set name '{chad_mask_set_name}' doesn't match the chip label mask set name "
                f"'{chip_label_mask_set_name}'."
            )

        if chad_variant != chip_label_variant:
            raise ExaError(
                f"CHAD variant '{chad_variant}' doesn't match the chip label variant '{chip_label_variant}'."
            )
