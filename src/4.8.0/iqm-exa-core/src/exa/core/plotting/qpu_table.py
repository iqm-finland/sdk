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
"""QPU Table Rendering module."""

from collections.abc import Callable
import logging
from typing import Any

from jinja2 import Environment
import numpy as np

from exa.common.errors import ValidationError
from exa.core.observation import ObservationStash
from exa.core.util.dut_field_decoder import decode_dut_field
from iqm.station_control.interface.models import ObservationData

from .plots import format_for_display
from .qpu_table_core.units_and_scaling import create_unit_string

logger = logging.getLogger(__name__)


def parse_parameter_from_dut_field(dut_field_with_comp: str) -> tuple[str, str]:
    """Split DUT field into component and parameter or in case not possible throw a Warning.

    Args:
        dut_field_with_comp: DUT field containing component.

    Returns:
        Tuple with first element as component and second element as parameter.

    """
    final_comp = ""
    final_param = ""
    try:
        component, parameter_name = decode_dut_field(dut_field_with_comp)
    except ValidationError:
        logger.warning("cannot split DUT field %s.", dut_field_with_comp)
    else:  # TODO: This part can be removed when readout legacy observations are removed.
        comp_split = component.split("__")  # when getting DUT field from SettingsNode
        if len(comp_split) > 1:
            component = comp_split[0]
            parameter_name = ".".join(comp_split[1:] + [parameter_name])

        final_comp = component
        final_param = parameter_name

    return final_comp, final_param


def _generate_dut_field(component: str, parameter_name: str) -> str | None:
    """Generate DUT field from component and parameter while respecting the
    pulse2 gate naming scheme.

    Args:
        component: can be a qubit or coupler label.
        parameter_name: parameter from settings or observation.

    Returns:
        returns DUT field with component name in it.

    """
    return_str = None
    if component and parameter_name:
        more_split = parameter_name.split(".")
        if more_split[0] == "gates":
            return_str = ".".join(more_split[0:3] + [component] + more_split[3:])
        else:
            return_str = ".".join([component, parameter_name])
    return return_str


def get_scaling_and_units(initial_stash: ObservationStash, selected_components: list[str]) -> dict[str, dict[str, Any]]:
    """Collect the values for selected components from stash and uses that
    to compute new units and scales.

    Args:
        initial_stash: observation stash to run computation on.
        selected_components: component labels, typically qubits or couplers.

    Returns:
        all_selected_params_with_scale_and_units: dictionary of parameters with scales and units.

    """
    all_selected_params_with_scale_and_units: dict[str, dict[str, Any]] = {}

    # First we collect all parameters only for selected components and
    # for those with units, we also save their values in a list.
    for obs in list(initial_stash.get_latest_observations().values()):
        component, parameter_name = parse_parameter_from_dut_field(obs.dut_field)

        if component in selected_components:
            # Create a key if there isnt one.
            if all_selected_params_with_scale_and_units.get(parameter_name) is None:
                all_selected_params_with_scale_and_units[parameter_name] = {
                    "scaling": 1.0,
                    "unit": getattr(obs, "unit", ""),
                }
                if isinstance(obs.value, np.ndarray):
                    all_selected_params_with_scale_and_units[parameter_name]["values"] = obs.value.flatten()
                else:
                    all_selected_params_with_scale_and_units[parameter_name]["values"] = [obs.value]

            else:
                if isinstance(obs.value, np.ndarray):
                    temp_arr = all_selected_params_with_scale_and_units[parameter_name]["values"]
                    all_selected_params_with_scale_and_units[parameter_name]["values"] = np.concatenate(
                        (temp_arr, obs.value.flatten())
                    )
                else:
                    all_selected_params_with_scale_and_units[parameter_name]["values"].append(obs.value)
                if (
                    hasattr(obs, "unit")
                    and obs.unit != all_selected_params_with_scale_and_units[parameter_name]["unit"]
                ):
                    logger.warning("%s has different units than others", obs.dut_field)

    # Now for parameters with values, we calculate the scaling and new units,
    # and remove the values before we return the dictionary
    for parameter_info in all_selected_params_with_scale_and_units.values():
        if parameter_info["unit"]:
            scaling, unit_string = create_unit_string(
                parameter_info["values"],
                parameter_info["unit"],
            )
            parameter_info["scaling"] = scaling
            parameter_info["unit"] = unit_string
        parameter_info.pop("values", None)

    return all_selected_params_with_scale_and_units


def format_value(value: float, decimals_to_print: int) -> str:
    """Autoscale to power of 10 and respect decimal precision."""
    if abs(value) != 0.0:  # Avoid zero in log
        power_of_ten = int(np.log10(np.abs(value)))  # Avoid negative values in log
        if abs(power_of_ten) > decimals_to_print:
            scaled_val = value / (10**power_of_ten)
            return f"{format(scaled_val, f'#.{decimals_to_print}f')}e{power_of_ten}"
    return format(value, f"#.{decimals_to_print}f")


def nice_print(value: float | np.ndarray, decimals_to_print: int = 3) -> str:
    """Scales every value to nearest power of 10, even if units are not specified.
    Also works on arrays by iterating over individual values.

    Args:
        value: array or float value from observation.
        decimals_to_print: no of digits in decimals places to print.

    Return:
        string of value rounded to nearest power of 10, with specified decimal places.

    """
    if isinstance(value, np.ndarray):
        flatten_array = value.flatten()
        if len(flatten_array) > 6:
            return_str = "[" + ", ".join([format_value(i, decimals_to_print) for i in flatten_array[:3]])
            return_str = return_str + ", ...,"
            return return_str + ", ".join([format_value(i, decimals_to_print) for i in flatten_array[-3:]]) + "]"
        return "[" + ", ".join([format_value(i, decimals_to_print) for i in flatten_array]) + "]"
    return format_value(value, decimals_to_print)


def preetify_with_given_units(
    all_selected_params_with_scale_and_units: dict[str, dict[str, Any]] | None, decimals_to_print: int = 3
) -> Callable[[Any, dict[str, dict[str, Any]], int], str]:
    """Wrapper function that creates a preety printing function with user specified scale and units."""

    def _pretty_value(
        observation: ObservationData,
        all_selected_params_with_scale_and_units: dict[str, dict[str, Any]]
        | None = all_selected_params_with_scale_and_units,
        decimals_to_print: int = decimals_to_print,
    ) -> str:
        """Return observation value with correct unit and prefix."""
        returns_str = "N/A"
        if observation:
            _, parameter_name = parse_parameter_from_dut_field(observation.dut_field)
            if all_selected_params_with_scale_and_units and parameter_name in all_selected_params_with_scale_and_units:
                if isinstance(observation.value, int | float | complex | np.ndarray):
                    returns_str = str(
                        nice_print(
                            observation.value * all_selected_params_with_scale_and_units[parameter_name]["scaling"],
                            decimals_to_print,
                        )
                    )
                elif isinstance(observation.value, list | set | tuple):
                    if len(observation.value) > 6:
                        flattened_list = list(observation.value)
                        if isinstance(flattened_list[0], int | float | complex):
                            returns_str = "[" + ",".join(
                                [format_value(i, decimals_to_print) for i in flattened_list[:3]]
                            )
                            returns_str += " ... "
                            returns_str += ",".join([format_value(i, decimals_to_print) for i in flattened_list[-3:]])
                            returns_str += "]"
                        else:
                            returns_str = str(str(flattened_list[:3])[:-1] + " ... " + str(flattened_list[-3:])[1:])
                else:
                    returns_str = str(observation.value)
            else:
                returns_str = nice_print(observation.value, decimals_to_print)  # type: ignore[arg-type]
        return returns_str

    return _pretty_value


_table_jinja_template = """

<table>
    <tr>
        <th> {{ "  " ~ component_type ~ "  "}}</th>
        {% for temp_header, temp_unit_and_scaling in unique_headers.items() %}
            <th style="text-align: center;">{{pretty_header(temp_header)}}<br>{{temp_unit_and_scaling["unit"]}}</th>
        {% endfor %}
    </tr>
    {% if enable_dut_field -%}
    <tr>
        <th> DUT FIELD </th>
        {% for temp_header, temp_unit_and_scaling in unique_headers.items() %}
            <th style="text-align: center;">{{temp_header}}</th>
        {% endfor %}
    </tr>
    {% endif %}
    {% for temp_comp in selected_component %}
        <tr>
        <td style="text-align: left;">{{ temp_comp }}</td>
        {% for temp_header in unique_headers %}
            {% set obs_name = dut_assembler(temp_comp,temp_header) %}
            {% set obs = get_latest_obs(obs_name) %}
            {% set obs_value = pretty_value(obs) %}

            <td style="text-align: right;font-family: monospace;">{{ obs_value }}</td>
            {% endfor %}
        </tr>

    {% endfor %}
</table>
"""


_table_jinja_template_transpose = """

<table>
    <tr>
        <th> {{ "  " ~ component_type ~ "  "}}</th>
        {% if enable_dut_field -%}
            <th> DUT FIELD </th>
        {% endif %}
        {% for temp_comp in selected_component %}
            <th style="text-align: center;">{{ temp_comp }}</th>
        {% endfor %}
    </tr>
    {% for temp_header, temp_unit_and_scaling in unique_headers.items() %}
        <tr>
        {% if enable_dut_field -%}
            <td style="text-align: center;">{{pretty_header(temp_header)}}</td>
        {% endif %}
        <th>{{temp_header}}<br>{{temp_unit_and_scaling["unit"]}}</th>

        {% for temp_comp in selected_component %}
            {% set obs_name = dut_assembler(temp_comp,temp_header) %}
            {% set obs = get_latest_obs(obs_name) %}
            {% set obs_value = pretty_value(obs) %}

            <td style="text-align: right;font-family: monospace;">{{ obs_value }}</td>
        {% endfor %}
        </tr>
    {% endfor %}
</table>
"""


def generate_table(  # noqa: PLR0913
    initial_stash: ObservationStash,
    selected_components: list[str],
    table_component_type: str = "Component",
    decimals_to_print: int = 3,
    list_of_parameters_to_display: list[str] | None = None,
    params_with_scale_and_units: dict[str, dict[str, Any]] | None = None,
    display_dut_field: bool = False,
    transpose_table: bool = False,
    blacklist_parameters: list[str] | None = None,
) -> str:
    """Returns HTML file in string format containing table for selected components
    and column headers specified by :list_of_parameters_to_display:.

    Args:
        initial_stash: observation stash from which observation will be used.
        selected_components: component label for which table is to be created.
        table_component_type: component type, usually qubit or coupler.
        decimals_to_print: digits to print after decimal.
        list_of_parameters_to_display: list of parameter names to be displayed.
        params_with_scale_and_units: custom units and scaling for parameters.
        display_dut_field: display observation parameters.
        transpose_table: tranpose table for displaying.
        blacklist_parameters: list of parameters to not display.

    Returns:
        HTML Table in string format that.

    """
    env = Environment()
    table_template = (
        env.from_string(_table_jinja_template_transpose) if transpose_table else env.from_string(_table_jinja_template)
    )

    all_selected_params_with_scale_and_units: dict[str, dict[str, Any]] | None = params_with_scale_and_units
    if params_with_scale_and_units is None:
        all_selected_params_with_scale_and_units = get_scaling_and_units(initial_stash, selected_components)
    new_pretty_val_func = preetify_with_given_units(all_selected_params_with_scale_and_units, decimals_to_print)

    # Only keep parameters to be displayed. This dict will always have some values.

    final_params_with_scale_and_units: dict[str, dict[str, Any]] = all_selected_params_with_scale_and_units  # type: ignore[assignment]
    if list_of_parameters_to_display:
        final_params_with_scale_and_units = {}
        for parameter_name in list_of_parameters_to_display:
            final_params_with_scale_and_units[parameter_name] = all_selected_params_with_scale_and_units.get(  # type: ignore[union-attr,assignment]
                parameter_name,
                None,  # type: ignore[arg-type]
            )

    if blacklist_parameters:
        for parameter_name in blacklist_parameters:
            if parameter_name in final_params_with_scale_and_units:
                del final_params_with_scale_and_units[parameter_name]

    return table_template.render(
        pretty_value=new_pretty_val_func,
        get_latest_obs=initial_stash.get_latest_observation,
        unique_headers=final_params_with_scale_and_units,
        pretty_header=format_for_display,
        selected_component=selected_components,
        component_type=table_component_type,
        enable_dut_field=display_dut_field,
        dut_assembler=_generate_dut_field,
    )


def display_all_parameters_for_components(initial_stash: ObservationStash, selected_comps: list[str]) -> list[str]:
    """Displays all parameters from initial stash for given components."""
    params = []
    for obs in list(initial_stash.get_latest_observations().values()):
        component, parameter_name = parse_parameter_from_dut_field(obs.dut_field)
        if component in selected_comps and parameter_name not in params:
            params.append(parameter_name)
    return params
