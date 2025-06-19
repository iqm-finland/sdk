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

from exa.common.errors.station_control_errors import StationControlError


class IqmServerError(StationControlError):
    def __init__(self, message: str, status_code: str, error_code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details

    def __str__(self):
        s = f"{self.message} (status_code = {self.status_code}, error_code = {self.error_code}"
        if details := self.details:
            s += f", details = {details}"
        s += ")"
        return s
