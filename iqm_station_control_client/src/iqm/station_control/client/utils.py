#  ********************************************************************************
#  Copyright (c) 2021-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************

"""Utility functions for IQM Station Control Client."""

from collections.abc import Callable, Iterable

from tqdm.auto import tqdm

from exa.common.data.value import ObservationValue
from iqm.station_control.interface.models import Statuses
from iqm.station_control.interface.models.observation import ObservationBase


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


def calset_from_observations(calset_observations: Iterable[ObservationBase]) -> dict[str, ObservationValue]:
    """Create a calibration set from the given observations.

    Args:
        calset_observations: observations that form a calibration set

    Returns:
        calibration set

    """
    return {obs.dut_field: obs.value for obs in calset_observations}
