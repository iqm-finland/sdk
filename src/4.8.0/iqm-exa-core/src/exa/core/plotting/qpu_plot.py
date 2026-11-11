#  *****************************************************************************
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
#  *****************************************************************************
"""QPUPlot definition."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cached_property, partial
import logging
from pathlib import Path
from typing import Any
from uuid import UUID
import warnings

import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import numpy as np
from typing_extensions import deprecated

from exa.common.data import Parameter
from exa.common.errors import IQMError, ValidationError
from exa.common.helpers import format_deprecated
from exa.common.qcm_data import ChipLayout, ChipTopology
from exa.common.qcm_data.chad_model import CHAD
from exa.core.database_wrapper import _get_observations_from_database
from exa.core.observation import LocalObservation, ObservationStash
from exa.core.provider import ExaRuntime
from exa.core.util.dut_field_decoder import decode_dut_field
from exa.core.util.param_parsing import chunk_list
from iqm.station_control.interface.models import ObservationData, ObservationDefinition

from .plot_cumulative_distribution import _plot_cdf, _prepare_data_for_multi_cdf
from .plots import (
    _construct_observations_list,
    _plot_bar_chart,
    _prepare_data_for_bar_chart,
    format_for_display,
    plot_histogram,
)
from .qpu_core import colormaps
from .qpu_core.colormaps import IQM_COLORMAP
from .qpu_core.utils import QPUPlotObservation, print_observations
from .time_series import _plot_time_series_from_observations
from .topology import (
    _create_observation_plot_data,
    _get_chip_layout_from_csv,
    plot_topology,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageOptions:
    """All settings for saving figures.

    Attributes:
        resolution: Raster images are saved with this resolution.
        format: All figures will be saved in this format, e.g., png, jpg, pdf, svg.
        directory: The path to save the figure.

    """

    resolution: int
    format: str
    directory: Path | None = None


class QPUPlot:
    """Help creating a plot for a given QPU that shows observation data."""

    def __init__(  # noqa: PLR0913
        self,
        chip_label: str,
        qb_colormap: ListedColormap | None = None,
        tc_colormap: ListedColormap | None = None,
        cr_colormap: ListedColormap | None = None,
        observations: Sequence[ObservationDefinition] | None = None,
        aux_colormap: ListedColormap | None = None,
        save_directory: Path | None = None,
        resolution: int = 300,
        figure_format: str = "png",
        split_dut_field: Callable[[str], tuple[str, str]] | None = None,
        chip_layout: ChipLayout | None = None,
    ) -> None:
        """Initialize QPUPlot.

        Args:
            chip_label:
                The DUT label of the chip, used to find the respective chip design record.
            qb_colormap:
                If provided, color the qubits in the topology plot based on their data value,
                otherwise qubits are uncolored.
            tc_colormap:
                If provided, color the couplers in the topology plot based on their data value,
                otherwise couplers are uncolored.
            cr_colormap:
                If provided, color the computational resonators in the topology plot based on their data value,
                otherwise computational resonators are uncolored.
            aux_colormap:
                Auxiliary colormap that is used by CDF plot.
            observations:
                An optional list of observations passed on initialization, to be used in the plot.
                More observations can be added via add_observations or add_observations_from_database methods.
            save_directory: Where to save the figure.
            resolution: If figures are saved, rasterize with the given value in dpi.
            figure_format: All figures will be saved in this format, for example, png, jpg, pdf, svg.
            split_dut_field:
                How to extract component and parameter names from the DUT field. Default assumes that
                {dut_field}=={component_name}.{parameter_name}.
            chip_layout:
                Providing a chip layout is required if the metro data is not available in the chip design record.

        """
        self.dut_label = chip_label
        self._observations: list[QPUPlotObservation] = []

        # TODO remove ExaRuntime usage here, give chip_design_record as a parameter
        if chip_label == ExaRuntime.executor().get_dut_label():
            self.chip_design_record = ExaRuntime.executor().get_chip_design_record()
        else:
            self.chip_design_record = ExaRuntime.storage().get_chip_design_record(chip_label)
        self.chip_topology: ChipTopology = ChipTopology.from_chip_design_record(self.chip_design_record)

        if chip_layout:
            chip_layout_components = (
                set(chip_layout.qubits) | set(chip_layout.computational_resonators) | set(chip_layout.couplers)
            )
            chip_topology_components = (
                self.chip_topology.qubits | self.chip_topology.computational_resonators | self.chip_topology.couplers
            )

            if chip_layout_components != chip_topology_components:
                raise ValueError("Chip layout not compatible with chip design record.")
        try:
            chip_layout_from_chad: ChipLayout | None = ChipLayout.from_chip_design_record(self.chip_design_record)
        except IQMError:
            chip_layout_from_chad = None
            logger.warning("Chip layout could not be obtained from chip design records.")
        # TODO: refactor this together with selected qpu styles to scale with different coordinates being used
        if chip_layout_from_chad:
            chip_layout_from_chad.normalize_coordinates(np.sqrt(2) / 2)

        self.chip_layout: ChipLayout | None = chip_layout_from_chad or chip_layout

        if split_dut_field:
            self.split_dut_field = split_dut_field
        else:
            self.split_dut_field = partial(decode_dut_field, chip_topology=self.chip_topology)
        if observations is not None:
            self.add_observations(observations)

        self.qb_colormap = qb_colormap
        self.tc_colormap = tc_colormap
        self.cr_colormap = cr_colormap
        self.aux_colormap = aux_colormap
        self.set_colormaps()

        self.figures: list[dict[str, Figure]] = []  # we will save our figures here

        self.image_options = ImageOptions(resolution=resolution, format=figure_format, directory=save_directory)

    @property
    def chip_label(self) -> str:
        """The chip label, same as the DUT label."""
        return self.dut_label

    @cached_property
    @deprecated(format_deprecated(old="QPUPlot.chad", new="QPUPlot.chip_topology", since="2026-09-17"))
    def chad(self) -> CHAD:
        """The chip architecture definition (CHAD)."""
        return CHAD(**self.chip_design_record)

    @property
    def figs(self) -> list[Figure]:
        """List of figures rather than list of dictionary of figures for compliance."""
        plots: list[Figure] = []
        for figs in self.figures:
            plots.extend(list(figs.values()))
        return plots

    @property
    def observations(self) -> list[ObservationData | LocalObservation]:
        """The plain list of observations."""
        return [obs.observation for obs in self._observations]

    def add_observations(self, observations: ObservationDefinition | Sequence[ObservationDefinition]) -> None:
        """Add the given observations to our observations list.

        It's up to the caller of the method to decide where these observations came from.

        Args:
            observations: This will be added to the QPUPlot's observations list.

        """
        new_observations = []
        if isinstance(observations, ObservationDefinition):
            new_observations.append(observations)
        elif hasattr(observations, "source"):
            # Old observation where of type Observation seems to be deprecated.
            # This elif statement is supposed to catch those and filter out most other things...
            new_observations.append(observations)  # type: ignore[arg-type]
        elif isinstance(observations, Sequence):
            new_observations.extend(
                obs for obs in observations if isinstance(obs, ObservationDefinition) or hasattr(obs, "source")
            )

        if new_observations:
            self._observations.extend(
                _filter_dut_fields(_filter_observation_values(new_observations), self.split_dut_field)  # type: ignore[arg-type]
            )

    def add_observations_from_database(
        self,
        *,
        latest: bool = True,
        all_tags: bool = True,
        dut_fields: list[str] | None = None,
        obs_params: list[Parameter] | None = None,
        run_id: UUID | None = None,
        tags: list[str] | None = None,
        min_date: date | str | None = None,
        max_date: date | str | None = None,
    ) -> None:
        """Add observations from database, filtering based on the provided criteria.

        Args:
            latest:
                Add only the latest observation for each DUT field. Otherwise add all observations
                for the given filters.
            all_tags:
                Add the observations that satisfy all tags. Otherwise add observations that contain at least one tag
                if specified in ``tags``.
            dut_fields:
                The IDs of the type of observation in the database, e.g., component_name.parameter_name.
            obs_params:
                Add the observations with this parameter for each of the components on the chip if they have it.
            run_id:
                Add only the observations from the specified run ID.
            tags:
                Tags to filter the observations. If this is the only argument, the latest observations for these
                tags are returned.
            min_date:
                Exclude observations with earlier created_timestamp. If defined as a string, it
                should be in ISO-8601 date format. Dates are not aware of timezones. Requires ``max_date`` to be set.
            max_date:
                Only include observations with earlier created_timestamp. If defined as a string, it
                should be in ISO-8601 date format. Dates are not aware of timezones. Requires ``min_date`` to be set.

        """
        if obs_params and dut_fields:
            if any(param.name not in dut_fields for param in obs_params):
                return

        elif obs_params:
            dut_fields = []

            for param in obs_params:
                dut_fields += (
                    [f"{qb}.{param.name}" for qb in self.chip_topology.qubits_sorted]
                    + [f"{tc}.{param.name}" for tc in self.chip_topology.couplers_sorted]
                    + [f"{cr}.{param.name}" for cr in self.chip_topology.computational_resonators_sorted]
                )

        query_arguments: dict[str, Any] = {"dut_label": self.dut_label}
        if run_id:
            query_arguments["source__run_id__in"] = [run_id]

        if all_tags:
            query_arguments["tags__contains"] = tags
        else:
            query_arguments["tags__overlap"] = tags
        if latest:
            query_arguments["latest"] = "dut_field"
        elif min_date and max_date:
            query_arguments["created_timestamp__range"] = (min_date, max_date)

        queried_observations: list[ObservationData] = []
        if not dut_fields:
            query_arguments["dut_field__in"] = dut_fields
            queried_observations = list(_get_observations_from_database(**query_arguments))
        else:
            # Ensure that we do not pass so many dut fields at once that the url exceeds 2000 characters.
            for chunk in chunk_list(dut_fields, 25):
                query_arguments["dut_field__in"] = chunk
                queried_observations += _get_observations_from_database(**query_arguments)
        self._observations.extend(
            _filter_dut_fields(_filter_observation_values(queried_observations), self.split_dut_field)
        )

        return

    def add_observations_from_stash(
        self,
        stash: ObservationStash,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Get observations from an existing stash, independently of their origin.

        Args:
            stash: ObservationStash to load from.
            components: List of components to load observations for. Defaults to None.
            observations: List of observations to load. Defaults to None.
            tags: List of tags to load. Defaults to None.

        """
        local_observation_mapping = stash.get_local_observations(components, observations, tags)
        db_observation_mapping = stash.get_db_observations(components, observations, tags)
        for key in local_observation_mapping:
            self.add_observations(observations=local_observation_mapping[key])  # type: ignore[arg-type]
        for key in db_observation_mapping:
            self.add_observations(observations=db_observation_mapping[key])  # type: ignore[arg-type]

    def draw_all(self, save_directory: Path | str | None = None, **kwargs) -> None:
        """Draw all the plots for all the data we have.

        Supported plots:
            * topology plot
            * cumulative histogram (CDF)

        Args:
            save_directory: The path to store the figures.
            **kwargs: kwargs for topology and CDF plot methods.

        """
        self.draw_topology(save_directory=save_directory, **kwargs)
        self.draw_cdf(save_directory=save_directory, **kwargs)

    def draw_topology(
        self,
        save_directory: Path | str | None = None,
        include_comp_names: bool = True,
        close_fig: bool = False,
        **kwargs,
    ) -> dict[str, Figure]:
        """Draw the topology plots for all the observations.

        Args:
            save_directory: If provided, the resulting plots are saved to that directory.
            include_comp_names: If True, include labels for component names.
            close_fig: If True, closes plt figures.
            **kwargs: kwargs for topology plot methods.

        Returns:
            The topology plot for each observation parameter.

        """
        parameter_names = {obs.metric for obs in self._observations}
        topology_plots: dict[str, Figure] = {}
        if self.chip_layout is None:
            if layout := kwargs.get("chip_layout"):
                self.chip_layout = layout
            elif isinstance(csv_file := kwargs.get("csv_file"), str):
                self.chip_layout = _get_chip_layout_from_csv(self.chip_design_record, csv_file)
            else:
                raise IQMError("No chip layout provided.")
        for name in parameter_names:
            observations = [obs for obs in self._observations if obs.metric == name]
            figure_name = f"{self.dut_label} - {format_for_display(name)}" if observations else "dummy"
            plot_data = _create_observation_plot_data(
                observations, self.chip_layout, only_rounding="vmin" in kwargs or "vmax" in kwargs
            )
            try:
                topology_plots[name] = plot_topology(
                    figure_title=figure_name,
                    layout=self.chip_layout,
                    data=plot_data,
                    observation_type=name,
                    qb_colormap=self.qb_colormap,
                    tc_colormap=self.tc_colormap,
                    cr_colormap=self.cr_colormap,
                    save_path=self._create_save_path("topology", name, save_directory),
                    include_component_names=include_comp_names,
                    resolution=self.image_options.resolution,
                    close_fig=close_fig,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning("Plot %s failed", name, exc_info=exc)

        self.figures.append(topology_plots)
        return topology_plots

    def draw_cdf(
        self,
        observation_types: list[str] | set[str] | None = None,
        save_directory: Path | str | None = None,
        figsize: tuple[float, float] = (6.0, 4.0),
        **kwargs,
    ) -> dict[str, Figure]:
        """Draw a cumulative histogram plot for all the observations of the given obs_types.

        The observations for each obs_type will be drawn on its own line, but in the same plot with the rest. The
        obs_types of the same unit will be plotted on the same plot, otherwise they will be drawn on different plots.

        Args:
            observation_types: The types of observations to plot from observations gathered in QPUPlot.
            save_directory: If provided, the resulting plot is saved to that directory.
            figsize: Width, height in inches.
            **kwargs: kwargs for plot multi cdf method.

        Returns:
            The CDF plot for each unit.

        """
        if not observation_types:
            observation_types = {obs.metric for obs in self._observations}

        # Check if there are multiple different units in the same list of observation types
        obs_types_and_units = _construct_observation_types_and_units(self._observations, observation_types)

        cdf_plots: dict[str, Figure] = {}
        for d_unit in obs_types_and_units:
            file_name = "_".join(f"{obs_type}" for obs_type in obs_types_and_units[d_unit])
            data, labels = _prepare_data_for_multi_cdf(
                self._observations,
                obs_types_and_units[d_unit],
                xscale=kwargs.get("xscale") or "linear",
                show_logit_mean=kwargs.get("show_logit_mean") or False,
                color_scheme=self.aux_colormap or IQM_COLORMAP,
            )

            fig = _plot_cdf(
                plot_data=data,
                save_path=self._create_save_path("cdf", file_name, save_directory),
                resolution=self.image_options.resolution,
                xscale=kwargs.get("xscale") or "linear",
                x_label=labels["x_label"],
                show_logit_mean=kwargs.get("show_logit_mean") or False,
                color_scheme=self.aux_colormap or IQM_COLORMAP,
                fig_size=figsize,
            )
            if fig:  # if the obs_types have mismatching units, fig will be None
                cdf_plots[d_unit] = fig
        self.figures.append(cdf_plots)
        return cdf_plots

    def draw_bar_chart(
        self,
        save_directory: Path | str | None = None,
        **kwargs,
    ) -> dict[str, Figure]:
        """Draw bar chart plots for the latest observations of a given type.

        If one_per_dut_field = True, all observations (not only the latest) are plotted in a box plot.

        Args:
            save_directory: If provided, the resulting plots are saved to that directory.
            **kwargs: kwargs for plot bar chart method.

        Returns:
            The bar chart for each observation parameter.

        """
        parameter_names = {obs.metric for obs in self._observations}
        bar_chart_plots: dict[str, Figure] = {}
        for name in parameter_names:
            data_values, unique_components, plot_labels = _prepare_data_for_bar_chart(
                [obs for obs in self._observations if obs.metric == name],
                name,
                **kwargs,  # one_per_dut_field
            )
            bar_chart_plots[name] = _plot_bar_chart(
                data_values=data_values,
                unique_components=unique_components,
                unit=plot_labels["unit"],
                line_label=plot_labels["line_label"],
                save_path=self._create_save_path("bar_chart", name, save_directory),
                resolution=self.image_options.resolution,
            )

        self.figures.append(bar_chart_plots)
        return bar_chart_plots

    def draw_time_series(
        self,
        dut_fields: list[str],
        save_directory: Path | str | None = None,
        **kwargs,
    ) -> Figure:
        """Draw a time series plot for multiple DUT fields.

        Args:
            dut_fields: The fields for the time series.
            save_directory: If provided, the resulting plots are saved to that directory.
            **kwargs: kwargs for plot time series method.

        Returns:
            The time series plot.

        """
        components_and_metrics = {self.split_dut_field(dut_field) for dut_field in dut_fields}

        fig = _plot_time_series_from_observations(
            observations=self._observations,
            components_and_metrics=components_and_metrics,
            save_path=self._create_save_path("time_series", dut_fields[0], save_directory),
            resolution=self.image_options.resolution,
            **kwargs,  # moving_average_number, show_uncertainty, plot_difference
        )
        self.figures.append({f"time_series_{dut_fields[0]}": fig})
        return fig

    def draw_histogram(
        self,
        save_directory: Path | str | None = None,
        obs_types: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Figure]:
        """Plot a histogram for observation.

        Args:
            save_directory: If directory is provided, figures are saved to the directory
            obs_types: Similarly to CDF plot, histogram will only be provided for these (different histograms).
            **kwargs: kwargs for plot histogram method.

        Returns:
            The histogram for each observation parameter.

        """
        parameter_names = {obs.metric for obs in self._observations} if obs_types is None else obs_types

        histogram_plots: dict[str, Figure] = {}
        for name in parameter_names:
            histogram_plots[name] = plot_histogram(
                [obs.observation for obs in self._observations if obs.metric == name],
                name,
                split_dut_field=self.split_dut_field,
                chip_design_record=self.chip_design_record,
                save_path=self._create_save_path("histogram", name, save_directory),
                resolution=self.image_options.resolution,
                **kwargs,  # one_per_dut_field, label_all_components, split_qubits
            )
        self.figures.append(histogram_plots)
        return histogram_plots

    def summarize(self) -> None:
        """Print out a summary of the data stored within the QPUPlot object."""
        print("---------------------------------------------------")
        print("Summary for QPU plot\n")
        print("DUT label: " + self.dut_label)
        if self.chip_topology.qubits_sorted:
            print("\nQubits: -------------------------------------------")
            print("   " + "\n   ".join(self.chip_topology.qubits_sorted))
        if self.chip_topology.couplers_sorted:
            print("\nCouplers: -----------------------------------------")
            print("   " + "\n   ".join(self.chip_topology.couplers_sorted))
        print("\nObservations: -------------------------------------")
        print_observations(self.observations)  # type: ignore[arg-type]
        print("---------------------------------------------------")

    def set_save_directory(self, save_directory: Path | None = None) -> None:
        """Set the directory to which we save resulting figures to.

        Leaving this blank will clear the save_directory.
        """
        self.image_options.directory = save_directory

    def set_colormaps(
        self,
        qb_colormap=colormaps.IQM_COLORMAP,  # noqa: ANN001
        tc_colormap=colormaps.IQM_COLORMAP,  # noqa: ANN001
        cr_colormap=colormaps.IQM_COLORMAP,  # noqa: ANN001
        aux_colormap=mpl.colormaps["Dark2"],  # noqa: ANN001
    ) -> None:
        """Set the colormaps to the given ones. Default is IQM colormap for both qubits and couplers."""
        self.qb_colormap = qb_colormap
        self.tc_colormap = tc_colormap
        self.cr_colormap = cr_colormap
        self.aux_colormap = aux_colormap

    def set_qb_colormap(self, colormap=colormaps.IQM_COLORMAP) -> None:  # noqa: ANN001
        """Set the qubit colormap to the given one. Default is IQM colormap."""
        self.qb_colormap = colormap

    def set_tc_colormap(self, colormap=colormaps.IQM_COLORMAP) -> None:  # noqa: ANN001
        """Set the coupler colormap to the given one. Default is IQM colormap."""
        self.tc_colormap = colormap

    def set_cr_colormap(self, colormap=colormaps.IQM_COLORMAP) -> None:  # noqa: ANN001
        """Set the computational resonator colormap to the given one. Default is IQM colormap."""
        self.cr_colormap = colormap

    def set_aux_colormap(self, colormap=mpl.colormaps["Dark2"]) -> None:  # noqa: ANN001
        """Set the auxiliary colormap. Default is IQM colormap."""
        self.aux_colormap = colormap

    def convert_fidelity_and_error(self) -> None:
        """Converts fidelities to errors and errors to fidelities.

        Note that in order to convert, we create new local observations with the correct parameters and delete the old
        ones, so this function is not perfectly reversible.
        """
        new_obs_list: list[QPUPlotObservation] = []  # our total observations list
        for obs in self._observations:
            if "fidelity" in obs.metric or "error" in obs.metric:
                if "fidelity" in obs.metric:
                    param_name = obs.metric.replace("fidelity", "error")
                else:
                    param_name = obs.metric.replace("error", "fidelity")

                if hasattr(obs.observation, "run_id"):
                    run_id = obs.observation.run_id
                elif hasattr(obs.observation, "source") and hasattr(obs.observation.source, "experiment_run_id"):
                    run_id = obs.observation.source.experiment_run_id
                else:
                    run_id = "dummy_id"

                parameter = Parameter(name=param_name)  # both error and fidelity are unitless
                # error + fidelity = 1
                value = 1 - obs.observation.value  # type: ignore[operator]
                uncertainty = obs.observation.uncertainty
                new_obs = LocalObservation.create(
                    dut_field=f"{obs.component}.{parameter.name}",
                    value=value,
                    unit=parameter.unit,
                    uncertainty=uncertainty,
                    run_id=run_id,
                )
                new_obs.created_timestamp = (
                    obs.observation.created_timestamp
                    if hasattr(obs.observation, "created_timestamp")
                    else datetime.now()
                )

                new_obs_list.append(QPUPlotObservation(component=obs.component, metric=param_name, observation=new_obs))

            else:
                new_obs_list.append(obs)

        self._observations = new_obs_list

    def _get_save_path(self, file_name: str | None = None, save_directory: Path | None = None) -> Path | None:
        """Return a path to save to.

        If file_name not given, we don't save to anything.
        If save_directory is given, we save to that.
        If save_directory is not given, but we have a save_directory defined, save to that.
        Else return None.
        """
        if not file_name:
            return None

        if save_directory:
            return save_directory / file_name
        if self.image_options.directory:
            return self.image_options.directory / file_name

        return None

    def _create_save_path(self, title: str, parameter_name: str, save_directory: Path | str | None) -> Path | None:
        formatted_name = parameter_name.replace(".", "_")
        current_time = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        file_format = self.image_options.format
        file_name = f"{title}_{formatted_name}_{current_time}.{file_format}"
        if save_directory is not None:
            save_directory = Path(save_directory)
        return self._get_save_path(file_name, save_directory)


def _construct_observation_types_and_units(
    observations: Iterable[QPUPlotObservation], observation_types: Iterable[str]
) -> dict[str, list[str]]:
    list_of_units: list[str] = []
    for name in observation_types:
        obs_list = _construct_observations_list(observations, name)  # type: ignore[arg-type]

        if not obs_list:
            raise ValidationError(f"Unable to construct a list of observations for the observation type {name}.")
        list_of_units.append(obs_list[0].observation.unit)

    unique_units = list(set(list_of_units))
    obs_types_and_units: dict = {k: [] for k in unique_units}

    for name, unit in zip(observation_types, list_of_units):
        obs_types_and_units[unit].append(name)

    return obs_types_and_units


def _filter_observation_values(observations: list[ObservationData]) -> list[ObservationData]:
    """Filter out any observations whose values can't be cast to float.

    Args:
        observations: The data that needs to be filtered.

    Returns:
        The observations with values that can be casted to float.

    """
    filtered_observations = []
    for data in observations:
        if isinstance(data.value, np.ndarray) and (
            np.ndim(np.squeeze(data.value)) != 0
            or np.issubdtype(data.value.dtype, str)
            or np.issubdtype(data.value.dtype, complex)
        ):
            warnings.warn(
                f"Observation with DUT field {data.dut_field} and NumPy array {data.value} can't be converted "
                f"to float and won't be added to the QPUPlot observations."
            )

        elif isinstance(data.value, str | complex):
            warnings.warn(
                f"Observation with DUT field {data.dut_field} and value {data.value} can't be converted "
                f"to float and won't be added to the QPUPlot observations."
            )

        elif data.value is None or np.isnan(data.value):
            warnings.warn(
                f"Observation with DUT field {data.dut_field} and value {data.value} can't be plotted "
                f"and won't be added to the QPUPlot observations."
            )
        else:
            filtered_observations.append(data)

    return filtered_observations


def _filter_dut_fields(
    observations: list[ObservationData], split_dut_field: Callable[[str], tuple[str, str]] = decode_dut_field
) -> list[QPUPlotObservation]:
    """Filter out any observations whose DUT fields can't be interpreted.

    Args:
        observations: The data that needs to be filtered.
        split_dut_field: Method to extract component and parameter names from DUT field.
            Defaults to {component_name}.{parameter_name}.

    Returns:
        The observations with valid DUT fields.

    """
    filtered_observations: list[QPUPlotObservation] = []
    for data in observations:
        try:
            component, parameter = split_dut_field(data.dut_field)
        except ValidationError:
            warnings.warn(f"Can't parse DUT field {data.dut_field}, it won't be added to the QPUPlot observations.")
            continue
        if component.startswith(("QB", "TC", "COMP")):
            filtered_observations.append(QPUPlotObservation(component=component, metric=parameter, observation=data))
        else:
            warnings.warn(
                f"Observation with DUT field {data.dut_field} contains an unknown component {component} "
                f"and won't be added to the QPUPlot observations."
            )

    return filtered_observations
