# Copyright 2019-2026 IQM Finland Oy
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
"""Fits statistical error."""

import numpy as np


def statistical_uncertainty(values: np.ndarray, fit_errors: np.ndarray) -> float:
    """Given a list of fitted values, and a list of their respective errors, calculate an statistical
    estimate of the sample uncertainty.

    Args:
        values (np.ndarray): list of fit values
        fit_errors (np.ndarray): list of fit errors

    Returns:
        float: Statistical estimate of the sample uncertainty

    """
    return np.sqrt(np.sum(np.array(fit_errors) ** 2) / len(fit_errors) + (np.nanvar(values)))
