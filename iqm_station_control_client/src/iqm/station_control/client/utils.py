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
"""Utility functions for IQM Station Control Client."""

from tqdm.auto import tqdm

from iqm.station_control.interface.models import ProgressCallback
from iqm.station_control.interface.models.jobs import _Progress


def get_progress_bar_callback() -> ProgressCallback:
    """Returns a callback function that creates or updates existing progressbars when called."""
    progress_bars = {}

    def _create_and_update_progress_bars(statuses: list[_Progress]) -> None:
        for label, value, total in statuses:
            if label not in progress_bars:
                progress_bars[label] = tqdm(total=total, desc=label, leave=True)
            progress_bars[label].n = value
            progress_bars[label].refresh()

    return _create_and_update_progress_bars
