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

"""Helper constants for SweepOptions classes."""

#: Default value for `count` value in options.
DEFAULT_COUNT: int = 101
#: Default value for `base` value in options.
DEFAULT_BASE: int = 10
#: Dictionary with all possible types of options
OPTIONS_TYPE: dict[str, str | list[str]] = {
    "start": "start",
    "stop": "stop",
    "center": "center",
    "span": "span",
    "count": "count",
    "step": "step",
    "asc": "asc",
    "start_exp": "start_exp",
    "stop_exp": "stop_exp",
    "center_exp": "center_exp",
    "span_exp": "span_exp",
    "fixed": "fixed",
    "start_stop_list": ["start", "stop", "count", "step"],
    "center_span_list": ["center", "span", "count", "step", "asc"],
    "start_stop_base_list": ["start", "stop", "count", "base"],
    "center_span_base_list": ["center", "span", "count", "base", "asc"],
}
