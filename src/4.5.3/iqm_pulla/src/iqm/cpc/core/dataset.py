# Copyright 2024-2025 IQM
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
"""Utilities related to xarray dataset handling."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial
import inspect
from typing import Any, SupportsFloat, overload

import numpy as np
from scipy.optimize import curve_fit
import xarray

from exa.common.data.parameter import DataType, Parameter
from exa.common.errors.exa_error import ExaError

FitResults = tuple[dict, np.ndarray]


def stack_along_dimension(
    data: xarray.DataArray, dimension: str | Sequence[str] | None = None, stack: str = "stack"
) -> xarray.DataArray:
    """Stack (that is, combine) all other dimensions other than the given `dimension` into a single dimension.

    The new dimension contains all the elements of the cartesian product over the other dimensions.

    With the `groupby` method, can be used to use functions which do not accept n-dimensional arrays, and repeat it
    over all other dimensions.
    Afterward, one should unstack the result to return to the original dimensionality.

    Example:
    .. code::

        data # has dimensions d1, d2, d3.
        stacked = stack_along_dimension(data, "d1", "the others")
        for prod_element, array in stacked.groupby("the others", squeeze=False):
            # `prod_element` is one of the element tuples of d2 x d3.
            # `do_math` operates on d1 and accepts `array` has only dims=(d1,):
            result = do_math(array)

    Args:
        data: Input data to combine the new dimension into
        dimension: The dimension string which will be stacked along; defaults to None
        stack: Name of the stack to insert into the DataArray provided, defaults to 'stack'

    """
    if dimension:
        if isinstance(dimension, str):
            dims_to_remove: Sequence[str] = [dimension]
        else:
            dims_to_remove = dimension
        dimensions = list(data.sizes)
        for dim in dims_to_remove:
            try:
                dimensions.remove(dim)
            except ValueError as err:
                raise ValueError(f"Tried to stack along {dim} but it is not one of dimensions {dimensions}") from err
        if not dimensions:
            raise RuntimeError(
                f"Xarray does not allow stacking data along {dimension} if it is the only dimension."
                f"Hint: try data.expand_dims('a_dummy_dimension') before stacking."
            )
        return data.stack({stack: dimensions})

    return data.stack({stack: [...]})


def compute_1d_fit(
    x_values: np.ndarray | Iterable,
    y_values: np.ndarray | Iterable,
    model: Callable,
    estimates: dict[str, float | np.ndarray],
    **curve_fit_kwargs,
) -> FitResults:
    """Compute and return the fit results for a 1-dimensional array of values

    Args:
        x_values: The 'true' data which the curve needs to fit; the independent variable
        y_values: Result values of the fit or 'dependent data'
        model: The function to use for fitting
        estimates: Guesses of possible parameter values for the fit function
        **curve_fit_kwargs: Additional named arguments for the curve fit function

    """
    # Checking that all parameters which are *required* (ie don't have default values) by the model function
    # are provided in estimates.
    # The first parameter is always the variable to evaluate for, so it is excluded from required estimates.
    required_estimates = [
        p.name for p in list(inspect.signature(model).parameters.values())[1:] if p.default is inspect.Parameter.empty
    ]
    if missing := set(required_estimates).difference(set(estimates.keys())):
        raise ValueError(
            f"Model function {model.__name__} needs arguments {required_estimates}, "
            f"but {missing} were missing from estimates."
        )
    # the estimates dict might not be ordered correctly according to the order of parameters of the model,
    # so we need to order it manually
    ordered_estimates = {p: estimates.get(p) for p in list(inspect.signature(model).parameters)[1:]}
    try:
        fit_parameters, pcov_exp = curve_fit(
            model,
            x_values,  # type: ignore[arg-type]
            y_values,  # type: ignore[arg-type]
            p0=list(ordered_estimates.values()),  # type: ignore[arg-type]
            **curve_fit_kwargs,  # type: ignore[arg-type]
        )
    except RuntimeError:
        fit_parameters = [np.nan for _ in ordered_estimates]  # type: ignore[assignment]
        pcov_exp = np.empty(
            (
                len(ordered_estimates),
                len(ordered_estimates),
            )
        )
        pcov_exp[:] = np.nan
    fit_parameters = dict(zip(ordered_estimates.keys(), fit_parameters))  # type: ignore[arg-type,assignment]
    fitted_curve = np.asarray(model(np.asarray(x_values), **fit_parameters))  # type: ignore[arg-type]

    stddev_names = list(map(lambda x: x + "_stddev", ordered_estimates.keys()))
    fit_stddevs = dict(zip(stddev_names, np.sqrt(np.diag(pcov_exp))))
    fit_parameters.update(fit_stddevs)  # type: ignore[attr-defined]

    return fit_parameters, fitted_curve  # type: ignore[return-value]


@overload
def apply_along_coordinate(
    data_array: xarray.DataArray,
    coord: str,
    func: Callable[[np.ndarray, np.ndarray], FitResults],
    returns: None = None,
    add_to: xarray.Dataset | None = None,
    result_parameter: Parameter | None = None,
    prefix: str = "",
) -> xarray.DataArray: ...


@overload
def apply_along_coordinate(
    data_array: xarray.DataArray,
    coord: str,
    func: Callable[[np.ndarray, np.ndarray], FitResults],
    returns: Sequence[str],
    add_to: xarray.Dataset | None = None,
    result_parameter: Parameter | None = None,
    prefix: str = "",
) -> tuple[xarray.DataArray, ...]: ...


def apply_along_coordinate(  # noqa: PLR0915
    data_array: xarray.DataArray,
    coord: str,
    func: Callable[[np.ndarray, np.ndarray], FitResults],
    returns: Sequence[str] | None = None,
    add_to: xarray.Dataset | None = None,
    result_parameter: Parameter | None = None,
    prefix: str = "",
) -> tuple[xarray.DataArray, ...] | xarray.DataArray:
    """Apply a 1-dimensional function along a selected coordinate axis inside a Nd data array.

    Computes the result of function `func` along selected coordinate `coord`.
    The function should map a 1d array to a new array of equal size.
    In addition, it may return a dict of scalars for each function call.

    The main use case for this is fitting, where `func` returns a fit to some raw data, plus the found fit parameters.

    For example, if `data_array` has dimensions A, B, and C, and we apply a fitting function along B, the fitter `func`
    function is evaluated for each coordinate tuple spanned by A and C and gets arrays of len(B) as input.
    The result has the same shape and dimensions as `data_array`.
    Optionally, the fit parameters evaluated at each A,C-coordinate tuple can be returned as separate data arrays if
    specified in `returns`.
    These have the shape (len(A), len(C)) and the names are the fit parameters, i.e. keys returned by `func`.

    Example:
        .. code-block::

            def my_fitter(x, y):
                # ... fit function f(x; a) to data y ...
                # Let's say a=5 produces the best fit.
                return {"a": 5}, np.random.rand(len(y))

            my_data = xarray.DataArray(
                np.random.rand(5, 3),
                dims=["x", "color"],
                coords=[[1, 2, 3, 4, 5], [0, 125, 255]]
            )

            fit, a = apply_along_coordinate(
                data_array=my_data,
                coord="x",
                func=my_fitter,
                returns=["a"]
            )
            assert my_data.sizes == fit.sizes == {'x': 5, 'color': 3}
            assert a.sizes == {"color": 3}

    Args:
        data_array: DataArray that contains the Nd data to operate on.
        coord: Name of the coordinate over which the 1d operation is performed.
            Can be any coordinate or dimension of `data_array`.
        func: A function that takes 1-dimensional data in format (x, y) and returns
            a dictionary containing the extra results (e.g. fit parameters), and the computed 1-d array.
        returns: Names of extra results to return. Each string should match one of the keys returned by
            `func`. The arrays are added to the result tuple in the same order as given in `returns`.
            Note that the computed array is always returned first and does not need to be specified.
        add_to: If given, the returned arrays are added to this dataset. The dataset is modified as a side effect.
        result_parameter: This parameter represents the computed data. By default will be generated by
            :func:`generate_fit_parameter`.
        prefix: If given, the names of all arrays in `returns` are prefixed with `<prefix>_`. Used to avoid name clashes
            with existing data variables.

    Returns:
        List of data arrays depending on `returns`. The first array is always the result produced by applying
        `func` along the axis `coord`. It has the same :math:`N` dimensions as `data_array`.
        The other arrays have :math:`N-1` dimensions and represent the extras requested in `returns`.

    """
    returns = list(returns) if returns is not None else []
    if coord in data_array.dims:
        dimension = coord
    else:
        dims = data_array.coords[coord].dims
        dimension = dims[0]
    other_dims = [d for d in data_array.dims if d != dimension]
    x_vals = data_array.coords[coord].values

    def _build_fit_array(out_arr, orig_array, attrs, fit_parameters, result_parameter):  # noqa: ANN001, ANN202
        fit_array = xarray.DataArray(out_arr, dims=orig_array.dims, coords=orig_array.coords)
        annotate(fit_array, result_parameter or generate_fit_parameter(orig_array))
        fit_array.attrs.update(
            {
                "target": attrs,
                "fit_parameters": fit_parameters,
                "stacked_dimensions": other_dims,
            }
        )
        return fit_array

    def _build_param_arrays(params_dict, keys, reduced, fit_parameters):  # noqa: ANN001, ANN202
        param_arrays = []
        for key in keys:
            arr = xarray.DataArray(np.asarray(params_dict[key]), coords=reduced.coords, dims=reduced.dims)
            annotate(arr, key, prefix)
            arr.attrs.update({"fit_parameters": fit_parameters, "stacked_dimensions": other_dims})
            param_arrays.append(arr)
        return param_arrays

    # Probe func to infer output dtypes; also capture probe result for use
    probe_y = np.asarray(data_array.isel({d: 0 for d in other_dims}).values)
    probe_y = probe_y.reshape(-1)  # Only vectorize along the core dimension
    probe_params, probe_out = func(x_vals, probe_y)
    output_dtypes = [np.asarray(probe_out).dtype]
    output_dtypes.extend(np.asarray(probe_params[k]).dtype for k in returns)
    output_dtypes.append(np.dtype(object))

    if not other_dims:
        reduced = data_array.isel({dimension: 0}, drop=True)
        y_in = np.asarray(data_array.values)
        y_in = y_in.reshape(-1)
        params, out = probe_params, probe_out  # Use probe result directly, no need to recompute
        out_arr = np.asarray(out)
        out_arr = out_arr.reshape(-1)
        fit_parameters = {(): params}
        fit_array = _build_fit_array(out_arr, data_array, data_array.attrs, fit_parameters, result_parameter)
        reduced_param_arrays = _build_param_arrays(params, returns, reduced, fit_parameters)
        if add_to is not None:
            add_many_data_arrays(add_to, (fit_array, *reduced_param_arrays))
        return (fit_array, *reduced_param_arrays)

    def _wrapper(y: np.ndarray, x: np.ndarray, _returns=returns) -> tuple[np.ndarray, *tuple[np.ndarray], dict]:  # noqa: ANN001, ANN202
        y = np.asarray(y).reshape(-1)  # reshape only along core dimension
        p, out = func(x, y)
        out = np.asarray(out).reshape(-1)
        return (out, *(p[k] for k in _returns), p)

    results = xarray.apply_ufunc(
        partial(_wrapper, _returns=returns),
        data_array,
        data_array.coords[coord],
        input_core_dims=[[dimension], [dimension]],
        output_core_dims=[[dimension], *([[]] * len(returns)), []],
        vectorize=True,
        output_dtypes=output_dtypes,
    )

    fit_array = results[0].transpose(*data_array.dims)
    param_arrays = list(results[1 : 1 + len(returns)])
    params_obj_array = results[-1].transpose(*other_dims)

    stacked_params = params_obj_array.stack(others=other_dims)
    fit_parameters = {}
    for spot, p in zip(stacked_params.coords["others"].values, stacked_params.values, strict=False):
        if not isinstance(spot, tuple):
            spot = (spot,)  # noqa: PLW2901
        fit_parameters[spot] = p

    fit_array = _build_fit_array(fit_array.values, data_array, data_array.attrs, fit_parameters, result_parameter)
    reduced_param_arrays = []
    for param_arr, key in zip(param_arrays, returns, strict=False):
        arr = param_arr.transpose(*other_dims)
        annotate(arr, key, prefix)
        arr.attrs.update({"fit_parameters": fit_parameters, "stacked_dimensions": other_dims})
        reduced_param_arrays.append(arr)
    if add_to is not None:
        add_many_data_arrays(add_to, (fit_array, *reduced_param_arrays))
    return (fit_array, *reduced_param_arrays)


def split_along_dimension(data: xarray.DataArray, dimension: str) -> list[xarray.DataArray]:
    """Slice an N-dimensional dataArray along specified dimension.

    Args:
        data: dataArray to split
        dimension: name of the dimension to split

    Returns:
        list of dataArrays corresponding to each value along `dimension`.

    """
    slices = []
    for value in data.coords[dimension].values:
        sliced = data.sel({dimension: value})
        slices.append(sliced)
    return slices


def generate_fit_parameter(data: xarray.DataArray) -> Parameter:
    """Construct a Parameter representing a fit to the given data array.

    The name of the generated parameter will be ``<name of array>_fit`` .
    The Parameter gets its unit from the dataset and its data_type from the data variable.

    Args:
        data: dataArray that contains the raw data.

    Returns:
        Generated parameter.

    """
    return Parameter(
        f"{data.name}_fit",
        f"Fit to {data.name}",
        data.attrs.get("units", ""),
        data.attrs["parameter"].data_type if "parameter" in data.attrs else DataType.ANYTHING,
    )


def annotate(
    data: xarray.DataArray,
    annotation: str | Parameter | None = None,
    prefix: str = "",
) -> xarray.DataArray:
    """Add parameter data to an array.

    Attaches a name to a DataArray. Also adds "parameter" data to the attributes.

    Various Exa functions leverage the additional metadata, mostly for display purposes.

    Args:
        data: Array to be modified.
        annotation: new name for the array. If `annotation` is a Parameter, use its name and label instead.
            If not given, `prefix` will be added to existing names.
        prefix: If given, this prefix is added to the new name and labels. If the new name already has the same
            prefix, it won't be duplicated.

    Returns:
        The modified array.

    """
    if isinstance(annotation, str):
        parameter = Parameter(annotation)
    elif annotation is None:
        parameter = data.attrs.get("parameter", Parameter(str(data.name)))
    else:
        parameter = annotation
    if prefix and not parameter.name.startswith(prefix):
        parameter = parameter.model_copy(
            update={
                "name": f"{prefix}_{parameter.name}",
                "label": f"{prefix} {parameter.label}",
            }
        )
    data.attrs["parameter"] = parameter
    data.attrs["units"] = parameter.unit
    data.attrs["standard_name"] = parameter.name
    data.attrs["long_name"] = parameter.label
    data.name = parameter.name
    return data


def convert_2d_model(
    model: Callable,
    cloned_arguments_indices: None | Sequence[int] = None,
) -> Callable:
    """Converts a function taking two independent variables and parameters to a format usable with curve_fit, by flattening
    the model by the second dimension.

    Args:
        model: the original function taking x, y independent variables
        cloned_arguments_indices: the indices of the arguments we would like to clone in the list of all the arguments
            of the original model

    Returns:
        The new model function, taking x, y, y_indices, and extended parameter list of the original model

    """  # noqa: E501
    arguments = list(inspect.signature(model).parameters)[2:]

    count = len(arguments)
    if cloned_arguments_indices is None:
        cloned_arguments_indices = []

    def _model(data, *args):  # noqa: ANN001, ANN202
        x, y, y_ind = data
        clone_number = int((len(args) - count) / len(cloned_arguments_indices)) + 1 if cloned_arguments_indices else 1
        j = 0
        red_args = []
        for i in range(count):
            if i in cloned_arguments_indices:
                indices = y_ind + j
                next_arg = np.array(args)[indices.astype(int)]
                j = j + clone_number
            else:
                next_arg = args[j]
                j = j + 1
            red_args = red_args + [next_arg]
        return model(x, y, *red_args)

    return _model


def _get_indices(model: Callable, cloned_arguments: None | Iterable[str], required_estimates: list[str]):  # noqa: ANN202
    """Convert the list of names of cloned arguments into a list of indices in the function signature corresponding
    to those.
    """
    if cloned_arguments is not None:
        extra_args = set(cloned_arguments).difference(set(required_estimates))
        if extra_args:
            raise ValueError(
                f"Model function {model.__name__} was requested to fix {extra_args}, "
                f"but it only accepts the arguments {required_estimates}."
            )
        cloned_arguments_indices = [
            i for i in range(len(required_estimates)) if required_estimates[i] in cloned_arguments
        ]
    else:
        cloned_arguments_indices = []
    return cloned_arguments_indices


def _format_fit_estimates(
    estimates: dict, required_estimates: list[str], cloned_arguments_indices: list[int], clone_number: int
) -> list[SupportsFloat | complex]:
    """Format fit estimates to be compatible with curve_fit."""
    fit_estimates: list[SupportsFloat | complex] = []
    for i, key in enumerate(required_estimates):
        if i in cloned_arguments_indices:
            if isinstance(estimates[key], list):
                if len(estimates[key]) == clone_number:
                    new_estimates = estimates[key]
                else:
                    new_estimates = clone_number * [estimates[key][0]]
            else:
                new_estimates = clone_number * [estimates[key]]
            fit_estimates = fit_estimates + new_estimates
        else:
            fit_estimates = fit_estimates + [estimates[key]]
    return fit_estimates


def _format_fit_parameters(cloned_arguments_indices, required_estimates, fit_parameters, covariance, clone_number):  # noqa: ANN001, ANN202
    """Group fit parameters corresponding to the cloned arguments together and add their standard deviations."""
    fit_parameters_formatted = {}
    stddev_vals = np.sqrt(np.diag(covariance))
    j = 0
    for i, key in enumerate(required_estimates):
        if i in cloned_arguments_indices:
            fit_parameters_formatted[key] = fit_parameters[j : j + clone_number]
            fit_parameters_formatted[key + "_stddev"] = stddev_vals[j : j + clone_number]
            j = j + clone_number
        else:
            fit_parameters_formatted[key] = fit_parameters[j]
            fit_parameters_formatted[key + "_stddev"] = stddev_vals[j]
            j = j + 1
    return fit_parameters_formatted


def compute_2d_fit(
    x_values: np.ndarray[Any, Any],
    y_values: np.ndarray[Any, Any],
    z_values: np.ndarray[Any, Any],
    model: Callable,
    estimates: dict,
    cloned_arguments: None | Iterable[str] = None,
    only_index_y: bool = False,
    **curve_fit_kwargs,
) -> FitResults:
    """Compute and return the fit results for a 2-dimensional array of values

    Args:
        x_values:  The primary independent variable
        y_values: The secondary independent variable, also used as an index
        z_values: The dependent variable
        model: The function to use for fitting
        estimates: Guesses of possible parameter values for the fit function
        cloned_arguments: list of arguments to the model which should be fitted separately for each y value
        **curve_fit_kwargs: Additional named arguments for the curve fit function

    """
    # Checking that all the parameters required by the model function are provided in estimates
    required_estimates = list(inspect.signature(model).parameters)[2:]

    missing = set(required_estimates).difference(set(estimates.keys()))
    if missing:
        raise ValueError(
            f"Model function {model.__name__} needs arguments {required_estimates}, "
            f"but {missing} were missing from estimates."
        )
    # Checking if the fixed args are actually arguments to the function
    cloned_arguments_indices = _get_indices(model, cloned_arguments, required_estimates)

    _model = convert_2d_model(model, cloned_arguments_indices)

    clone_number = len(y_values)

    x_vals_fit = np.tile(x_values, clone_number)
    y_vals_fit = np.repeat(y_values, len(x_values))
    y_ind_fit = np.repeat(np.arange(clone_number), len(x_values))

    if z_values.shape == (clone_number, len(x_values)):
        z_vals_fit = z_values.ravel()
    elif z_values.shape == (len(x_values), clone_number):
        z_vals_fit = z_values.T.ravel()
    else:
        raise ExaError(
            f"The shape of the z-array is {z_values.shape}, instead of shape of the arguments "
            f"{(len(y_values), len(x_values))}. Please fix the data."
        )

    if only_index_y:
        dep_vals_fit = np.vstack((x_vals_fit, y_ind_fit, y_ind_fit))
    else:
        dep_vals_fit = np.vstack((x_vals_fit, y_vals_fit, y_ind_fit))

    fit_estimates = _format_fit_estimates(estimates, required_estimates, cloned_arguments_indices, clone_number)

    fit_parameters, covariance = curve_fit(_model, dep_vals_fit, z_vals_fit, p0=fit_estimates, **curve_fit_kwargs)  # type: ignore[arg-type]
    fit_parameters_formatted = _format_fit_parameters(
        cloned_arguments_indices, required_estimates, fit_parameters, covariance, clone_number
    )

    fitted_curve = np.asarray(_model(dep_vals_fit, *fit_parameters)).reshape((clone_number, len(x_values)))
    if z_values.shape == (clone_number, len(x_values)):
        return fit_parameters_formatted, fitted_curve
    return fit_parameters_formatted, fitted_curve.T


@overload
def multifit_along_coordinates(
    data_array: xarray.DataArray,
    main_coord: str,
    supp_coord: str,
    func: Callable[[np.ndarray, np.ndarray, np.ndarray, Sequence[str] | None], FitResults],
    cloned_arguments: Sequence[str] = (),
    returns: None = None,
    add_to: xarray.Dataset | None = None,
    result_parameter: Parameter | None = None,
    prefix: str = "",
) -> xarray.DataArray: ...


@overload
def multifit_along_coordinates(
    data_array: xarray.DataArray,
    main_coord: str,
    supp_coord: str,
    func: Callable[[np.ndarray, np.ndarray, np.ndarray, Sequence[str] | None], FitResults],
    cloned_arguments: Sequence[str],
    returns: Sequence[str],
    add_to: xarray.Dataset | None = None,
    result_parameter: Parameter | None = None,
    prefix: str = "",
) -> tuple[xarray.DataArray, ...]: ...


def multifit_along_coordinates(  # noqa: PLR0913
    data_array: xarray.DataArray,
    main_coord: str,
    supp_coord: str,
    func: Callable[[np.ndarray, np.ndarray, np.ndarray, Sequence[str] | None], FitResults],
    cloned_arguments: Sequence[str] = (),
    returns: Sequence[str] | None = None,
    add_to: xarray.Dataset | None = None,
    result_parameter: Parameter | None = None,
    prefix: str = "",
) -> tuple[xarray.DataArray, ...] | xarray.DataArray:
    r"""Extension of :func:`apply_along_coordinate` to two dimensions.

    The main difference is the addition of `cloned_arguments`, which enable the user to
    fit some parameters once per each value on the second, supplementary coordinate. The function generates a 2d fit,
    some 1d arrays for the cloned arguments, and 0d arrays for the non-cloned ones, so it needs to know which are
    cloned and which are not. These dimensions are automatically extended by additional dimensions on the data, and
    so this function supports arbitrary additional dimensions.

    For example, let's say we want to fit a family of exponential curves :math:`a_i\exp(-x/b)` where
    :math:`a_i` and :math:`b` are the fitting parameters and :math:`x` is the primary dimension.
    The value of :math:`b` is the same for the whole family, whereas :math:`a_i` is different for each coordinate `i`
    of the supplementary dimension.
    In this case, we clone `"a"` so that we get a single fit value for :math:`b`, and separate fit values for
    :math:`a_i`.

    Args:
        data_array: DataArray that contains the Nd data to operate on.
        main_coord: Name of the main coordinate over which the 2d operation is performed.
            Can be any coordinate or dimension of `data_array`.
        supp_coord: Name of the supplementary coordinate over which the 2d operation is performed. This coordinate
            also define the cloning axis for the cloned arguments.
        func: A function that takes 2-dimensional data in format (x, y, z) and returns
            a dictionary containing the extra results (e.g. fit parameters), and the fitted 2d array.
        cloned_arguments: The names of `func` arguments that the fitter should clone along the supplementary
            coordinate. They also define the shape of the returns corresponding to those arguments.
        returns: Names of extra results to return. Each string should match one of the keys returned by
            `func`. The arrays are added to the result tuple in the same order as given in `returns`.
            Note that the computed array is always returned first and does not need to be specified.
        add_to: If given, the returned arrays are added to this dataset. The dataset is modified as a side effect.
        result_parameter: This parameter represents the computed data. By default will be generated by
            :func:`generate_fit_parameter`.
        prefix: If given, the names of all arrays in `returns` are prefixed with `<prefix>_`. Used to avoid name clashes
            with existing data variables.

    Returns:
        List of data arrays depending on `returns`. The first array is always the result produced by applying
        `func` along the axis `coord`. It has the same :math:`N` dimensions as `data_array`.
        The other arrays have either :math:`N-1` (if cloned) or :math:`N-2` (if not cloned) dimensions and represent
        the extras requested in `returns`.

    """
    # This function uses the indices on dimensions, which makes it a bit clunky, but it works.
    # FIXME: The 2D fitting should be really generalized to N dimensions and combined with the 1D fitting.
    returns = returns or []

    # The multifit has to be done against the correct coordinates, but the data mangling happens using dimensions.
    # There are 1-to-many coordinates for each dimension, so we need to keep track of both.
    main_dimension = str(data_array.coords[main_coord].dims[0])
    supp_dimension = str(data_array.coords[supp_coord].dims[0])

    collected = _prepare_data_arrays(data_array, main_dimension, supp_dimension, returns, cloned_arguments, prefix)

    # Stack all other dimensions, so that the data is essentially 3d.
    # For example, if `data_array` has shape (a, B, c, d, E), and we want to fit along B and E,
    # `stacked_data` has shape (a*c*d, B, E). Here we have a*c*d 'spots' (=index tuples) which have data of shape B, E
    if set(data_array.dims) == {main_dimension, supp_dimension}:
        stacked_data = [data_array]
        spots: list[tuple] | np.ndarray = [()]
        stacked_dimensions = []
    else:
        stacked = stack_along_dimension(data_array, [main_dimension, supp_dimension], "others")
        spots = stacked.coords["others"].values
        stacked_dimensions = stacked.indexes["others"].names
        stacked_data = [stacked.sel(others=s) for s in stacked.coords["others"]]

    result, fit_parameters = _multifit_at_spot(
        data_array,
        func,
        main_coord,
        supp_coord,
        main_dimension,
        supp_dimension,
        spots,
        stacked_data,
        cloned_arguments,
        collected,
        prefix,
    )

    annotate(result, result_parameter or generate_fit_parameter(data_array))
    result.attrs.update(
        {"target": data_array.attrs, "fit_parameters": fit_parameters, "stacked_dimensions": stacked_dimensions}
    )

    if add_to is not None:
        add_many_data_arrays(add_to, (result, *collected))
    return result, *collected


def _prepare_data_arrays(
    data_array: xarray.DataArray,
    main_dimension: str,
    supp_dimension: str,
    returns: Sequence[str],
    cloned_arguments: Sequence[str],
    prefix: str,
) -> list[xarray.DataArray]:
    """For each returned quantity, prepare an empty data array to later contain the fit results. The shape of this
    array will depend on whether the argument corresponding to the quantity is cloned.
    """
    collected = []
    once_reduced_array = data_array.isel({main_dimension: 0}, drop=True)
    twice_reduced_array = once_reduced_array.isel({supp_dimension: 0}, drop=True)
    for key in returns:
        main_key = key.removesuffix("_stddev")
        if main_key in cloned_arguments:
            reduced_array = once_reduced_array
        else:
            reduced_array = twice_reduced_array
        new = xarray.DataArray(np.empty(reduced_array.shape), coords=reduced_array.coords, dims=reduced_array.dims)
        annotate(new, key, prefix)
        collected.append(new)
    return collected


def _multifit_at_spot(  # noqa: ANN202, PLR0913
    data_array: xarray.DataArray,
    func: Callable,
    main_coord: str,
    supp_coord: str,
    main_dimension: str,
    supp_dimension: str,
    spots: list[tuple] | np.ndarray,
    stacked_data: list[xarray.DataArray],
    cloned_arguments: Sequence[str],
    collected: list[xarray.DataArray],
    prefix: str,
):
    """Perform the fit with the `func` at each previously prepared spot, and then assign the results to respective
    arrays prepared beforehand. Note that the data arrays inside `collected` and thus this list itself, is modified.
    """
    fit_parameters = {}
    main_selected_index = data_array.dims.index(main_dimension)
    supp_selected_index = data_array.dims.index(supp_dimension)
    result = xarray.DataArray()
    # Now we do the fit at each spot, a*c*d times in total.
    for spot, spot_data in zip(spots, stacked_data):
        x = spot_data.coords[main_coord].values
        y = spot_data.coords[supp_coord].values
        # make sure y dimension is first for the fitting
        if main_selected_index < supp_selected_index:
            spot_values = spot_data.values.T
        else:
            spot_values = spot_data.values
        parameters, out = func(x, y, spot_values, cloned_arguments)

        if result.shape == ():
            # Initialize empty `result` just in time, to get the correct dtype
            result = xarray.DataArray(
                np.empty(data_array.shape, dtype=out.dtype), dims=data_array.dims, coords=data_array.coords
            )
        fit_parameters[spot] = parameters

        # `result` has the same shape (a, B, c, d) as the original `data_array`.
        # a problem here is that this index magic depends on the order of the first and second coordinate, there
        # are three cases for "other" coordinates' indices now

        # Assign the `out` array to position (a, :, c, d, :):
        if main_selected_index < supp_selected_index:
            result.loc[
                spot[:main_selected_index]
                + (slice(None),)
                + spot[main_selected_index : supp_selected_index - 1]
                + (slice(None),)
                + spot[supp_selected_index - 1 :]
            ] = out.T.squeeze()
        else:
            result.loc[
                spot[:supp_selected_index]
                + (slice(None),)
                + spot[supp_selected_index : main_selected_index - 1]
                + (slice(None),)
                + spot[main_selected_index - 1 :]
            ] = out.squeeze()

        # Each return array has shape (a, c, d), and `spot` tells the location in that shape:
        for array in collected:
            if main_selected_index > supp_selected_index:
                supp = supp_selected_index
            else:
                supp = supp_selected_index - 1
            param_name = str(array.name).removeprefix(prefix + "_")
            main_param_name = param_name.removesuffix("_stddev")
            if main_param_name in cloned_arguments:
                array.loc[spot[:supp] + (slice(None),) + spot[supp:]] = parameters[param_name]
            else:
                array.loc[spot] = parameters[param_name]

    return result, fit_parameters


def turn_variable_to_coordinate(dataset: xarray.Dataset, variable_name_to_convert: str) -> None:
    """Turns variable of xarray dataset into a coordinate.
    The function is added because the native xarray utilities always return a new dataset, so changes don't stick
    inside analysis dataset.

    Args:
        dataset: the dataset where the change will happen
        variable_name_to_convert: name of variable which needs to be turned into a coordinate

    Returns:
        None

    """
    if variable_name_to_convert in dataset.variables:
        variable_data = dataset[variable_name_to_convert]
        dataset.coords[variable_name_to_convert] = variable_data
        dataset = dataset.drop_vars(variable_name_to_convert)
        dataset.coords[variable_name_to_convert] = variable_data
    else:
        raise ValueError(f"No variable '{variable_name_to_convert}' in the dataset.")


def add_many_data_arrays(ds: xarray.Dataset, arrays: Iterable[xarray.DataArray]) -> None:
    """Efficiently add multiple DataArrays to a Dataset in one shot.

    Builds a dictionary from names to DataArrays and updates the Dataset in place.
    Variable attrs and coords on the DataArrays are preserved.

    Args:
        ds: Dataset to add to.
        arrays: Iterable of DataArrays to be added to the dataset.

    """
    if isinstance(arrays, xarray.DataArray):
        arrays = [arrays]

    ds.update({da.name: da for da in arrays})  # Keep full DataArray, incl. attrs & coords
