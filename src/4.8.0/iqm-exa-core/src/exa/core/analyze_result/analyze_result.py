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
"""AnalyzeResult class for experiment analyze() return values."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import xarray as xr

from exa.common.data import ObservationUncertainty, ObservationValue, Parameter, Setting, SettingNode
from exa.common.errors import ValidationError
from exa.core.observation import LocalObservation, LocalObservationSequence
from exa.core.plotting import Figure
from exa.core.run_result import RunResult
from exa.core.util.report import display_in_notebook, get_template, save_to_file
from iqm.cpc.core.dataset import add_many_data_arrays
from iqm.station_control.client.qon import QON

from .plot_definition import PlotDefinition


class AnalyzeResult:
    """Data object for the result of the analysis.

    Args:
        run_result: The run result containing the full dataset and run data.
        observations: Observations produced by the analysis.
        plots: A dict of plot definitions, which can be used to visualize `run_result.dataset`.
        extra: A dictionary to store extra analyze results which can't be added to any other attribute.
        error: An optional string to store the summary of a possible error during analysis.
        error_traceback: An optional string to store the traceback of a possible error during analysis.

    """

    def __init__(
        self,
        run_result: RunResult,
        *,
        observations: Sequence[LocalObservation] | None = None,
        plots: dict[str, PlotDefinition] | None = None,
        extra: dict | None = None,
        error: str | None = None,
        error_traceback: str | None = None,
    ) -> None:
        self.run_result = run_result
        self.observations = LocalObservationSequence(observations)
        self.plots = plots or {}
        self.extra = extra or {}
        self.error = error or None
        self.error_traceback = error_traceback or None

    def __repr__(self) -> str:
        ret = f"""AnalysisResult of {self.run_result.run_definition.experiment_label} {self.run_result.run_definition.run_id}
.observations:
    {self.observations}
.plots:
    {self.plots.keys()}
.extra:
    {self.extra.keys()}
"""  # noqa: E501
        if self.error is not None:
            ret += f"""
.error:
    {self.error}
"""
        return ret

    @property
    def dataset(self) -> xr.Dataset:
        """Shortcut to access `run_result.dataset`."""
        return self.run_result.dataset

    @property
    def settings(self) -> SettingNode:
        """Shortcut to access `run_result.run_definition.sweep_definition.settings`."""
        return self.run_result.run_definition.sweep_definition.settings

    def plot(self, key: str = "default", **kwargs) -> Figure:
        """Generate and show a selected plot from :attr:`plots`.

        Args:
            key: Selects which plot to show. If "default" and no "default" key exists, returns the first one.
            kwargs: Keyword arguments passed to the generating function.

        Returns:
            The generated Figure.

        Raises:
            ValidationError: If no plots exist in :attr:`plots`.
            KeyError: If `key`  does not exist in :attr:`plots`.

        """
        if not self.plots:
            raise ValidationError("No plots defined.")
        if key == "default" and key not in self.plots:
            key = next(iter(self.plots.keys()))
        run_result = kwargs.pop("run_result", self.run_result)
        fig = self.plots[key].generate_from_run_result(run_result, **kwargs)
        return fig

    def plot_all(self, keys: list[str] | None = None, **kwargs) -> list[Figure]:
        """Generate and show multiple plots from :attr:`plots`.

        Args:
            keys: The list of plot keys to show. If left `None`, will show all plots in `self``.
            kwargs: Additional arguments to :meth:`plot`.

        Returns:
            The list of the generated Figures.

        Raises:
            ValidationError: if no plots exist in :attr:`plots`.
            KeyError: if `key`  does not exist in :attr:`plots`.

        """
        if not self.plots:
            raise ValidationError("No plots defined.")
        plot_keys = keys or list(self.plots.keys())
        plots = []
        for key in plot_keys:
            fig = self.plot(key, **kwargs)
            plots.append(fig)
        return plots

    def add_arrays(self, *args: xr.DataArray) -> None:
        """Add any number of data arrays to :attr:`run_result.dataset`.

        Hint: Use :func:`.annotate` first to add parameter information.

        Args:
            args: DataArrays to add. Each one should have a name or "parameter" metadata.

        """
        add_many_data_arrays(self.run_result.dataset, args)

    def add_setting_observation(
        self,
        setting: Setting,
        value: ObservationValue,
        *,
        uncertainty: ObservationUncertainty | None = None,
        tags: list[str] | None = None,
        validation_rules: list[Callable] | None = None,
    ) -> None:
        """Add an observation to :attr:`observations` from a given setting.

        Args:
            setting: Setting to be converted into an observation.
            value: Value that is saved as an observation for ``dut_field``.
            uncertainty: The uncertainty of the observation value. ``None`` means unknown.
            tags: Tags used for observations.
            validation_rules: The list of validation rules for the observation.

        """
        settings = self.run_result.run_definition.sweep_definition.settings
        self.add_parameter_observation(
            parameter=settings.setting_with_path_name(setting).parameter,  # type: ignore[union-attr]
            value=value,
            uncertainty=uncertainty,
            tags=tags,
            validation_rules=validation_rules,
        )

    def add_parameter_observation(
        self,
        parameter: Parameter,
        value: ObservationValue,
        *,
        prefix: str = "",
        uncertainty: ObservationUncertainty | None = None,
        tags: list[str] | None = None,
        validation_rules: list[Callable] | None = None,
    ) -> None:
        """Add an observation to :attr:`observations` from a given parameter.

        Args:
            parameter: Parameter to be converted into an observation.
            value: Value that is saved as an observation for ``dut_field``.
            prefix: String, typically a component name, attached in front of the dut field, separated by a dot.
            uncertainty: The uncertainty of the observation value. ``None`` means unknown.
            tags: Tags used for observations.
            validation_rules: The list of validation rules for the observation.

        """
        self.add_observation(
            dut_field=f"{prefix}.{parameter.name}" if prefix else parameter.name,
            value=value,
            unit=parameter.unit,
            uncertainty=uncertainty,
            tags=tags,
            validation_rules=validation_rules,
        )

    def add_observation_with_qon(
        self,
        *,
        qon: QON,
        value: ObservationValue,
        unit: str,
        uncertainty: ObservationUncertainty | None = None,
        tags: list[str] | None = None,
        validation_rules: list[Callable] | None = None,
    ) -> None:
        """Add an observation to :attr:`observations`.

        Args:
            qon: Qualified observation name of the property the observation is about.
            unit: The observation unit (base SI unit, e.g. Hz and never GHz).
                Can be an empty string for quantities without unit (e.g. fidelity values).
            value: Value of the observation.
            uncertainty: The uncertainty of the observation value. ``None`` means unknown.
            tags: Tags used for observations.
            validation_rules: Validation rules for the observation.

        """
        self.observations.append(
            LocalObservation.create(
                dut_field=str(qon),
                value=value,
                unit=unit,
                uncertainty=uncertainty,
                tags=tags,
                run_id=self.run_result.run_definition.run_id,
                validation_rules=validation_rules,
            )
        )

    def add_observation(
        self,
        dut_field: str,
        value: ObservationValue,
        unit: str,
        *,
        uncertainty: ObservationUncertainty | None = None,
        tags: list[str] | None = None,
        validation_rules: list[Callable] | None = None,
    ) -> None:
        """Add an observation to :attr:`observations`.

        Args:
            dut_field: Name of the property the observation is about.
            unit: The observation unit (base SI unit, e.g. Hz and never GHz).
                Can be an empty string for quantities without unit (e.g. fidelity values).
            value: Value that is saved as an observation for ``dut_field``.
            uncertainty: The uncertainty of the observation value. ``None`` means unknown.
            tags: Tags used for observations.
            validation_rules: The list of validation rules for the observation.

        """
        self.observations.append(
            LocalObservation.create(
                dut_field=dut_field,
                value=value,
                unit=unit,
                uncertainty=uncertainty,
                tags=tags,
                run_id=self.run_result.run_definition.run_id,
                validation_rules=validation_rules,
            )
        )

    def render_report(self, target: Literal["notebook", "str"] | Path = "notebook") -> str:
        """Generate a report that displays the observations and plots from the analysis.

        Args:
            target: The target format as string or a file path. Accepted values: "notebook", "str" or :class:`Path`
                instance. For "notebook", the output is a  HTML document that is displayed in a Jupyter
                notebook. Target "str" returns HTML format in a string, and path instances saves
                document to a file.

        Returns:
            Rendered string or None if the report is rendered in notebook as a side effect.

        """
        template = get_template("experiment_standalone") if isinstance(target, Path) else get_template("experiment")
        report = template.render(
            analyze_result=self,
            index="",
            options={"disable_collapse": True},
            run_id=self.run_result.run_definition.run_id,
        )
        if target == "notebook":
            return display_in_notebook(report)

        if target == "str":
            return report

        if isinstance(target, Path):
            return save_to_file(target, report)

        raise ValueError(f"Invalid report target {target}")
