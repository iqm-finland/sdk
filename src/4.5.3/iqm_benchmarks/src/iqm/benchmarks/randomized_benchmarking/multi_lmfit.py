# Copyright 2024 IQM Benchmarks developers
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

"""Lmfit for multiple datasets.

Generalized and extended version of the shown example in the lmfit documentation:
https://lmfit.github.io/lmfit-py/examples/example_fit_multi_datasets.html#sphx-glr-examples-example-fit-multi-datasets-py
"""

from collections.abc import Callable
from inspect import signature

from lmfit import Parameters
import numpy as np


def eval_func_single_dataset(func: Callable, params: Parameters, i: int, x: np.ndarray) -> np.ndarray:
    """Returns the evaluation of the fit function for a single dataset.

    Args:
        func: The fit function to evaluate.
        params: lmfit Parameters object containing all parameter values.
        i: Index of the dataset (0-based).
        x: Independent variable values.

    Returns:
        Evaluated function values for the given dataset.

    """
    _, fit_param_names = get_param_names_from_func_signature(func)

    # assume parameter object has existing parameter names with a suffix _{i}
    args = [params[f"{name}_{i + 1}"] for name in fit_param_names]

    return func(x, *args)


def get_param_names_from_func_signature(func: Callable) -> tuple[str, list[str]]:
    """Gets the function parameter names from its signature.

    Args:
        func: The function to extract parameter names from.

    Returns:
        A tuple containing:
            - The name of the independent parameter (first parameter)
            - A list of fit parameter names (remaining parameters)

    """
    param_names = list(signature(func).parameters.keys())
    independent_param_name = param_names[0]
    fit_param_names = param_names[1:]
    return independent_param_name, fit_param_names


# use model .post_fit= to calculate fidelities.


def multi_dataset_residual(params: Parameters, x: np.ndarray, data: np.ndarray, func: Callable) -> np.ndarray:
    """Calculate total residual for fits of func to several data sets.

    Args:
        params: lmfit Parameters object containing all parameter values for all datasets.
        x: Independent variable values.
        data: 2D array where each row represents a dataset to fit.
        func: The fit function to use for calculating residuals.

    Returns:
        Flattened 1D array of residuals for all datasets.

    """
    ndata, _ = data.shape
    resid = 0.0 * data[:]

    # make residual per line
    for i in range(ndata):
        resid[i, :] = data[i, :] - eval_func_single_dataset(func, params, i, x)

    # now flatten this to a 1D array; this will be needed by minimize()
    return resid.flatten()


def create_multi_dataset_params(
    func: Callable,
    data: np.ndarray,
    initial_guesses: dict[str, float] | None = None,
    constraints: dict[str, dict[str, float]] | None = None,
    simultaneously_fit_vars: list[str] | None = None,
) -> Parameters:
    """Generates lmfit Parameter object with parameters for each line to fit.

    Args:
        func: The fit function whose parameters will be extracted.
        data: 2D array where each row represents a dataset to fit.
        initial_guesses: Optional dictionary mapping parameter names to initial values.
        constraints: Optional dictionary mapping parameter names to constraint dicts
            with 'min' and 'max' keys.
        simultaneously_fit_vars: Optional list of parameter names that should be fit
            as a single parameter across all datasets.

    Returns:
        lmfit Parameters object with parameters for each dataset, with suffixes _1, _2, etc.

    """
    fit_params = Parameters()

    _, fit_param_names = get_param_names_from_func_signature(func)

    for i, _ in enumerate(data):
        for name in fit_param_names:
            fit_params.add(f"{name}_{i + 1}")

            # set initial guesses if provided
            if initial_guesses is not None:
                if name in initial_guesses:
                    fit_params[f"{name}_{i + 1}"].value = initial_guesses[name]
            # set constraints if provided
            if constraints is not None:
                if name in constraints:
                    fit_params[f"{name}_{i + 1}"].max = constraints[name]["max"]
                    fit_params[f"{name}_{i + 1}"].min = constraints[name]["min"]

            # force params to be fit as a single parameter across datasets
            if simultaneously_fit_vars is not None:
                if name in simultaneously_fit_vars:
                    if i > 0:
                        fit_params[f"{name}_{i + 1}"].expr = f"{name}_{1}"

    return fit_params
