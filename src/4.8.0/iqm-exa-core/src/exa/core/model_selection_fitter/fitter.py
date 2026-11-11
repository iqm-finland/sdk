#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Model-selection fitter provides fitting utilities built around lmfit-package."""

from dataclasses import replace
import logging
from typing import Any, Literal

import lmfit
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
import numpy as np
import xarray as xr

from .fitting import _fit_model_arrays
from .packaging import ModelArrays, assemble_results_dataset
from .plotting import plot_result
from .preprocessing import _process_dataset, prepare_fit_context
from .structures import FitContext, FitRequest, FitterSettings

logger = logging.getLogger(__name__)


class Fitter:
    """Fitter object that stores the models used for model-selection.

    To get started, create an instance of the fitter and add the models
    you would like to compare. Calling :meth:`fit` fits all the models
    to the given data, and selects the best one based on Bayesian information
    criterion.

    Example:

    .. code-block:: python

    from exa.core import model_selection_fitter as msf
    import xarray as xr
    import numpy as np

    fitter = msf.Fitter()
    model = msf.models.LinearModel()
    fitter.add_model(model)
    # Add another linear model with fixed offset.
    model_fixed = msf.models.LinearModel(name="Fixed")
    model_fixed.set_param_hint("b", value=0.1, vary=False)
    fitter.add_model(model_fixed)
    data = xr.DataArray(np.linspace(-2,2,21) + 0.5, dims=['time'], coords={"time": np.linspace(-1, 1, 21)})
    fit_result = fitter.fit(data=data, x="time", plot_fit=True)

    """

    def __init__(self) -> None:
        """Create a new Fitter instance."""
        self.models: list[lmfit.Model] = []
        self.results: list[xr.Dataset] = []

    def add_model(self, model: lmfit.Model) -> None:
        """Adds a model to the fitter.

        When calling fit method, all the added models are fit and the best
        model is selected.

        Args:
            model: Model to add.

        """
        self.models.append(model)

    def _normalize_options(
        self,
        *,
        independent_vars: dict[str, Any],
        fitter_options: dict[str, Any] | None,
        outlier_detection_options: dict[str, Any] | None,
        settings: FitterSettings,
    ) -> tuple[dict[str, Any], dict[str, Any], FitterSettings]:
        fitter_options_default = {"method": "leastsq", "nan_policy": "omit"}
        fitter_options_default.update(fitter_options or {})
        fitter_options_norm = fitter_options_default
        outlier_detection_options_norm = outlier_detection_options or {}

        if len(independent_vars.keys()) > 1 and fitter_options_norm["nan_policy"] != "raise":
            fitter_options_norm["nan_policy"] = "raise"
            logger.warning(
                "NaN removal is not supported for fits with more than one dimension. Changing nan_policy to raise."
            )

        if len(independent_vars.keys()) > 1 and settings.use_outlier_detection:
            settings = replace(settings, use_outlier_detection=False)
            logger.warning(
                """Outlier detections is not supported for fits with more than 1 dimension.
                Changing outlier detection to False."""
            )

        return fitter_options_norm, outlier_detection_options_norm, settings

    def _process_inputs(
        self,
        *,
        data: xr.Dataset | xr.DataArray,
        datavar: str,
        take_indices: tuple[int] | None,
        settings: FitterSettings,
        weights: xr.DataArray | None,
        independent_vars: dict[str, Any],
    ) -> tuple[tuple[xr.DataArray, list, dict], xr.DataArray | None]:
        processed_dataset = _process_dataset(
            data,
            datavar,
            take_indices,
            settings.reduction_action,
            complex_action=settings.complex_action,
            **independent_vars,
        )
        if weights is None:
            return processed_dataset, None
        processed_weights, _, _ = _process_dataset(
            weights,
            datavar,
            take_indices,
            settings.reduction_action,
            complex_action="real",
            **independent_vars,
        )
        return processed_dataset, processed_weights

    def _run_all_models(self, *, request: FitRequest, context: FitContext) -> ModelArrays:
        remaining_shape = context.remaining_shape
        data_dimensions = context.data_dimensions

        n_models = len(self.models)
        fit_result_all = np.empty((n_models,) + remaining_shape, dtype=object)
        fit_all = np.empty((n_models,) + data_dimensions, dtype=float)
        initial_guess_all = np.empty((n_models,) + data_dimensions, dtype=float)
        uncertainty_all = np.empty((n_models,) + data_dimensions, dtype=float)
        plot_defs_all = np.empty((n_models,) + remaining_shape, dtype=object)
        plots_all = (
            np.empty((n_models,) + remaining_shape, dtype=object) if request.settings.add_plot_to_dataset else None
        )
        outliers_all = (
            np.empty((n_models,) + data_dimensions, dtype=bool) if request.settings.use_outlier_detection else None
        )

        for model_index, model in enumerate(self.models):
            outputs = _fit_model_arrays(model, request=request, context=context)
            fit_result_all[model_index, ...] = outputs.fit_result
            fit_all[model_index, ...] = outputs.fit
            initial_guess_all[model_index, ...] = outputs.initial_guess
            uncertainty_all[model_index, ...] = outputs.uncertainty
            plot_defs_all[model_index, ...] = outputs.plot_definitions
            if plots_all is not None and outputs.plots is not None:
                plots_all[model_index, ...] = outputs.plots
            if outliers_all is not None and outputs.outliers is not None:
                outliers_all[model_index, ...] = outputs.outliers

        return ModelArrays(
            fit_result_all=fit_result_all,
            fit_all=fit_all,
            initial_guess_all=initial_guess_all,
            uncertainty_all=uncertainty_all,
            plot_defs_all=plot_defs_all,
            plots_all=plots_all,
            outliers_all=outliers_all,
        )

    def fit(  # noqa: PLR0913 PLR0915
        self,
        data: xr.Dataset | xr.DataArray,
        datavar: str = "",
        plot_fit: bool = False,
        add_plot_to_dataset: bool = False,
        use_outlier_detection: bool = False,
        take_indices: tuple[int] | None = None,
        complex_action: Literal["real", "imag", "rotate_to_real"] = "rotate_to_real",
        reduction_action: Literal["mean", "median"] | None = None,
        fitter_options: dict[str, Any] | None = None,
        outlier_detection_options: dict[str, Any] | None = None,
        override_model_initial_guess: bool = False,
        extra_parameter_penalty_factor: float = 1,
        select_by_bic: bool = False,
        weights: xr.DataArray | None = None,
        **independent_vars: str | list[str] | dict[Any, Any] | None,
    ) -> xr.Dataset:
        """Fits all the models known to the fitter and returns the best model.

        The independent variables are first extracted from the dataset. After that, complex action is applied
        to all the data, and the remaining dimensions are reduced by first taking only the indices given by
        ``take_index`` . Reduction action is then applied to all extra dimensions. Finally, the fit is done
        separately for all the remaining coordinates.

        At each step of the fit, all the models are fitted to the data, and the best model is chosen using
        the Akaike information criterion adjusted for small sample sizes.

        Args:
            data: Dataset with the data to be fitted. Can be also DataArray, in which case datavar is ignored.
            datavar: Data variable name. Defaults to None.
            plot_fit: If ``True`` , the best fit is plotted and shown immediately.
            add_plot_to_dataset: If ``True`` , plot of the fit is created and added to the dataset.
                If performance is critical, set this to False and use the "plot_definitions" return callable instead.
            use_outlier_detection: If ``True`` , uses leave-one-out outlier detection.
            take_indices: Picks only the extra dimensions specified by the tuple.
            complex_action: Converts complex values to real. Either 'real', 'imag', 'rotate_to_real'.
                If None, no conversion is done.
            reduction_action: Applies the given operation to reduce the remaining dimensions. Either 'mean',
                'std' or None.
            fitter_options: Key-value pairs for setting lmfit fitter options. If using lmfit models,
                these are the fitter options accepted by lmfit. If not options are specified,
                defaults to ``fitter_options = {"method": "leastsq", "nan_policy": "omit"}``
                Some ``fitter_options`` also have impact on how msf does fitting:
                *  nan_policy: if number of independent variables is > 1, changed to "raise".
            outlier_detection_options: Key-value pairs for configuring outlier detection.
            override_model_initial_guess: If True, initial guess is done using parameter values.
                If False model.guess function is used instead.
            extra_parameter_penalty_factor:
                Add an extra penalty on having more parameters in the model. Used to select the best model.
            select_by_bic: If true, select the best fit based on the bayesian information criterion (BIC). If false,
                select based on the Akaike Information Criterion corrected for small sample sizes.
            weights:
                Weight factor used to scale the residuals. The weight array must have the same shape as the data.
                Weights can also be applied through fitter_options, but then the same weights are used for all
                data points.
            **independent_vars:
                Maps the given coordinate names in the dataset to the independent variables of the model. For example,
                if the x-coordinate of the model is called x, and the corresponding coordinate in the dataset
                is 'time', give x='time' as the argument. If the model acts on multidimensional data, this can be
                specified by giving a list of coordinates. For example ``x = ['time', 'frequency']`` .

        Returns:
            Xarray Dataset containing the following data variables:
            - fit_result: Data array which elements are the ``ModelResult`` s and the coordinates are the remaining
                coordinates of the dataset an extra dimension that contains the names of the fitted models.
            - fit: The evaluated values of the fitted models.
            - initial_guess: The values for the initial guess.
            - uncertainty: The evaluated values for the uncertainty in the fitted data points.
            - plots: Plot of the fit if ``add_plot_to_dataset`` is ``True`` .
            - plot_definitions: a callable that plots a matplotlib figure, and suggested arguments for the callable.
            - best_model: The best ``ModelResult`` for each of the remaining coordinates.
            - best_model_index: The index for the best model for each of the extra dimension.
            - model_probabilities: Probability that each of the model is the best according to AIC.
            - AICs: Akaike information criterion for all the models.
            - AICcs: Akaike information criterion for small samples of all the models.
            - redchis: Reduced chi-square for all the models.
            - r2s: R2 values for all the models.
            - parameters: Fitted parameter values for the best model as separate datavars.
            - parameter_stderr: Standard errors for all the parameters as separate datavars.

        """
        settings = FitterSettings(
            plot_fit=plot_fit,
            add_plot_to_dataset=add_plot_to_dataset,
            use_outlier_detection=use_outlier_detection,
            complex_action=complex_action,
            reduction_action=reduction_action,
            override_model_initial_guess=override_model_initial_guess,
            extra_parameter_penalty_factor=extra_parameter_penalty_factor,
            select_by_bic=select_by_bic,
        )

        fitter_options, outlier_detection_options, settings = self._normalize_options(
            independent_vars=dict(independent_vars),
            fitter_options=fitter_options,
            outlier_detection_options=outlier_detection_options,
            settings=settings,
        )

        if len(self.models) == 0:
            raise ValueError("No models specified, please add one with 'add_model'.")

        processed_dataset, processed_weights = self._process_inputs(
            data=data,
            datavar=datavar,
            take_indices=take_indices,
            settings=settings,
            weights=weights,
            independent_vars=dict(independent_vars),
        )

        request = FitRequest(
            data=data,
            datavar=datavar,
            independent_vars=dict(independent_vars),
            take_indices=take_indices,
            settings=settings,
            fitter_options=fitter_options,
            outlier_detection_options=outlier_detection_options,
            weights=processed_weights,
        )

        context = prepare_fit_context(processed_dataset, independent_vars)
        arrays = self._run_all_models(request=request, context=context)
        results_for_models = assemble_results_dataset(
            models=self.models, request=request, context=context, arrays=arrays
        )
        self.results.append(results_for_models)
        return results_for_models

    def plot(
        self,
        quantity: str | list[str],
        indices: list[tuple[int] | int] | None = None,
        coordinate: str | list[str] | None = None,
        result_index: int = -1,
        ax: Axes | None = None,
        plot_style: Literal["point"] = "point",
        uncertainty_style: Literal["errorbar", "fill"] = "errorbar",
    ) -> FigureBase | None:
        """Plots the selected quantity.

        Args:
            quantity: Name of the quantity (datavar) or quantities to plot.
            indices:
                Coordinate indices to select for plotting. Can be either provided as a flat list of indices or a list
                of tuples of indices.
            coordinate: Coordinate to plot. Can also provide a list of coordinates for mult-dimensinal plots.
            result_index: index of the fit result to use for plotting.
            ax: Axes object for plotting. If None, a new axes is created.
            plot_style: Style of the plot.
            uncertainty_style: Style for uncertainties, options are 'errorbar' or 'fill'.

        Returns:
            Matplotlib figure which contains the axes.

        """
        try:
            result = self.results[result_index]
        except IndexError:
            return None
        return plot_result(
            result=result,
            quantity=quantity,
            indices=indices,
            coordinate=coordinate,
            ax=ax,
            plot_style=plot_style,
            uncertainty_style=uncertainty_style,
        )
