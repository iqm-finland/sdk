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

from collections.abc import Callable

import requests
from tqdm.auto import tqdm

from iqm.station_control.client.iqm_server.iqm_server_client import IqmServerClient
from iqm.station_control.client.station_control import StationControlClient
from iqm.station_control.interface.models import Statuses
from iqm.station_control.interface.station_control import StationControlInterface


def get_progress_bar_callback() -> Callable[[Statuses], None]:
    """Returns a callback function that creates or updates existing progressbars when called."""
    progress_bars = {}

    def _create_and_update_progress_bars(statuses: Statuses) -> None:
        for label, value, total in statuses:
            if label not in progress_bars:
                progress_bars[label] = tqdm(total=total, desc=label, leave=True)
            progress_bars[label].n = value
            progress_bars[label].refresh()

    return _create_and_update_progress_bars


def init_station_control(
    root_url: str, get_token_callback: Callable[[], str] | None = None, **kwargs
) -> StationControlInterface:
    """Initialize a new station control instance connected to the given remote.

    Client implementation is selected automatically based on the remote station: if the remote station
    is running the IQM Server software stack, then the IQM Server client implementation (with a limited
    feature set) is chosen. If the remote station is running the SC software stack, then the Station
    Control client implementation (with the full feature set) is chosen.

    Args:
        root_url: Remote station control service URL. For IQM Server remotes, this is the "Quantum Computer URL"
            value from the web dashboard.
        get_token_callback: A callback function that returns a token (str) which will be passed in Authorization
            header in all requests.

    """
    try:
        headers = {"Authorization": get_token_callback()} if get_token_callback else {}
        response = requests.get(f"{root_url}/about", headers=headers)
        response.raise_for_status()
        about = response.json()
        if isinstance(about, dict) and about.get("iqm_server") is True:
            # If about information has iqm_server flag, it means that we're communicating
            # with IQM server instead of direct Station Control service, hence we need to
            # use the specialized client
            return IqmServerClient(root_url, get_token_callback=get_token_callback, **kwargs)
        # Using direct station control by default
        return StationControlClient(root_url=root_url, get_token_callback=get_token_callback, **kwargs)
    except Exception as e:
        raise RuntimeError("Failed to initialize the client.") from e
