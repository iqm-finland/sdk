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
"""Tools and Jinja templates for reporting."""

from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, FunctionLoader, Template

from iqm.pulse.playlist.playlist import Playlist

DEFAULT_STYLE = """
.report_body {
    margin: 0 auto;
    text-align: center;
}
.exa-h1, .exa-h2, .exa-h3 {
    margin: 16px;
    letter-spacing: 2px;
    text-align: center;
    margin-bottom: 16px;
}
.exa-p {
    text-align: left;
    white-space: pre-wrap;
}

.exa-h1-error {
    background: red;
    color: white;
}
.exa-h2-error, .exa-h3-error, .exa-p-error {
    color: red;
}

.exa-image {
    max-width: 100%;
}
.table-content {
    text-align: center;
}

.exa-ul {
}
.exa-li {
}

.exa-table {
    margin-left: auto;
    margin-right: auto;
    width: 100%;
    margin-top: 16px;
    padding: 4px;
}

.exa-th {
    padding: 4px;
    border: 1px solid black;
    background: rgba(0,212,255,1);
    margin: 0;
}

.exa-td {
    padding: 4px;
    border: 1px solid black;
    margin: 0;
}

.dropdown-button {
    width: 100%;
    border: 0px solid black;
    background: rgb(0,212,255);
    background: linear-gradient(54deg, rgba(0,212,255,1) 0%, rgba(9,121,78,0.2777485994397759) 40%, rgba(244,250,251,1) 100%);
    border-radius: 5px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.2);
    text-align: left;
    font-size: 2rem;
    color: black;
    letter-spacing: 1.5px;
    padding: 14px;
    margin-top: 4px;
    margin-bottom: 4px;
    transition: 0.15s ease-in-out;
    box-shadow: -1px 3px 3px rgba(50, 45, 45, 0.226);
}

.dropdown-button-error {
    background: red;
}

.dropdown-button:hover {
    transform: translate(3px, -3px);
    box-shadow: -4px 4px 3px rgba(50, 45, 45, 0.226);
}
"""  # noqa: E501


DEFAULT_EXPERIMENT_TEMPLATE = """
{% if disable_style is not defined -%}
<style>
{% include "default_style" %}
</style>
{% endif %}
{% if options.disable_collapse is not defined -%}
<button
    class="dropdown-button {% if analyze_result.error -%}dropdown-button-error{% endif -%}"
    data-toggle="collapse"
    data-target="#exp{{ run_id }}"
>{{ index }}. {{ analyze_result.run_result.run_definition.experiment_label }}
</button>
{% endif -%}

{% if analyze_result.error -%}
<h3 class="exa-h3 exa-h3-error">Experiment analysis error: {{ analyze_result.error }}</h3>
<p class="exa-p exa-p-error">{{ analyze_result.error_traceback }}</p>
{% endif -%}


<div id="exp{{ run_id  }}" {% if options.disable_collapse is not defined -%}class="collapse"{% endif -%}>
    <h3 class="exa-h3">
        Observations from {{ analyze_result.run_result.run_definition.experiment_label }} {{ run_id }}
    </h3>

    <table class="exa-table">
        <tr><th class="exa-th">Field</th><th class="exa-th">Value</th><th class="exa-th">Unit</th></tr>
        {% for observation in analyze_result.observations %}
            <tr><td class="exa-td">{{ observation.dut_field }}</td>
            <td class="exa-td">{{ observation.pretty_value() }}</td>
            <td class="exa-td">{{ observation.unit }}</td></tr>
        {% endfor %}
    </table>
    <div>
        <h3 class="exa-h3">Plots from {{ analyze_result.run_result.run_definition.experiment_label }}</h3>
        {% for plot in analyze_result.plots.values() %}
            <img class="exa-image" src="{{ plot.embed_base64(analyze_result.run_result) }}" />
        {% endfor %}
    </div>
</div>
"""

DEFAULT_SEQUENCE_TEMPLATE = """
{% if disable_style is not defined -%}
<style>
{% include "default_style" %}
</style>
{% endif %}
<div class="report_body">
<h1 class="exa-h1 {% if result.get_errors() -%}exa-h1-error{% endif -%}">
{{ result.execution_result.context.sequence.name }}
</h1>

{% if result.get_errors() -%}
<h2 class="exa-h2 exa-h2-error">This sequence was executed with the following errors:</h2>
<ul class="exa-ul">
    {% for error in result.get_errors() -%}
        <li class="exa-li"><p class="exa-p exa-p-error">{{ error }}</p></li>
    {% endfor %}
</ul>
{% endif %}

{% if result.execution_result.context.error -%}
<h3 class="exa-h3 exa-h3-error">Sequence-level execution error: {{ result.execution_result.context.error }}</h3>
<p class="exa-p exa-p-error">{{ result.execution_result.context.error_traceback }}</p>
{% endif %}

{% if result.error -%}
<h3 class="exa-h3 exa-h3-error">Sequence-level analysis error: {{ result.error }}</h3>
<p class="exa-p exa-p-error">{{ result.error_traceback }}</p>
{% endif %}

<h2 class="exa-h2">Sequence-level Observations</h2>
    <table class="exa-table">
        <tr><th class="exa-th">Field</th><th class="exa-th">Value</th><th class="exa-th">Unit</th></tr>
        {% for observation in result.observations -%}
                <tr><td class="exa-td">{{ observation.dut_field }}</td>
                <td class="exa-td">{{ observation.pretty_value() }}</td>
                <td class="exa-td">{{ observation.unit }}</td></tr>
        {% endfor %}
    </table>

<h2 class="exa-h2">Sequence-level plots</h2>
    {% for key in result.plots.keys() %}
    <h3 class="exa-h3">{{ key }}</h3>
    <img class="exa-image" src="{{ result.plots[key].embed_base64(result.dataset) }}" />
    {% endfor %}

{% if options.disable_nodes is not defined -%}
<h2 class="exa-h2">Sequence Nodes</h2>
{% for analyze_result, node in zip(result.execution_result.analyze_results, result.execution_result.context.executed_nodes) -%}
    {% set index = loop.index -%}
    {% if node.type == "experiment" -%}
        {% set disable_style = True -%}
        {% set run_id = node.run_id %}
        {% include "experiment" %}
    {% elif node.type == "sequence" -%}
        {% set temp = result %}
        {% set result = analyze_result %}
        {% if options.disable_collapse is not defined -%}
        <button
            class="dropdown-button {% if result.get_errors() -%}dropdown-button-error{% endif -%}"
            data-toggle="collapse"
            data-target="#exp{{ node.sequence_run_id }}">{{ index }}. {{ analyze_result.execution_result.context.sequence.name }}
        </button>
        {% endif -%}
        <div id="exp{{ node.sequence_run_id }}" {% if options.disable_collapse is not defined -%}class="collapse"{% endif -%}>
            {% set disable_style = True -%}
            {% include "sequence" %}
        </div>
        {% set result = temp %}
        
    {% endif -%}
{% endfor -%}
{% endif -%}
</div>
"""  # noqa: E501

DEFAULT_EXPERIMENT_STANDALONE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href=
"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
    <script src=
"https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js">
    </script>
    <script src=
"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js">
    </script>
    <style>
body {
    width: 60%;
    margin: 0 auto;
    text-align: center;
}
    </style>
</head>
<body>

{% include "experiment" %}

</body>
"""

DEFAULT_SEQUENCE_STANDALONE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href=
"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
    <script src=
"https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js">
    </script>
    <script src=
"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js">
    </script>
    <style>
body {
    width: 60%;
    margin: 0 auto;
    text-align: center;
}
    </style>
    <title>{{ result.execution_result.context.sequence.name }}</title>
</head>
<body>

{% include "sequence" %}

</body>
"""


def display_in_notebook(report: str):  # noqa: ANN201
    """Display a report in notebook."""
    # Imported outside top level since notebook dependency brings ipython
    from IPython.display import HTML, display  # noqa: PLC0415 (import-outside-top-level)

    display(HTML(report))


def save_to_file(target: Path, report: str):  # noqa: ANN201
    """Save a report to a file."""
    with target.open("wb") as handle:
        handle.write(report.encode("utf-8"))


def get_template(
    template_type: Literal["sequence", "sequence_standalone", "experiment", "experiment_standalone", "default_style"],
) -> Template:
    """Function for getting the correct template to be used.

    Creates a Jinja2 environment and chooses the template to be given based on the template_type parameter.

    Args:
        template_type: Determines the template to provide

    Raises:
        ValueError: If the argument given is not any of the supported formats.

    Returns:
        Template: Jinja2 Template

    """

    def load_template(name: str):  # noqa: ANN202
        if name == "experiment":
            return DEFAULT_EXPERIMENT_TEMPLATE
        if name == "experiment_standalone":
            return DEFAULT_EXPERIMENT_STANDALONE_TEMPLATE
        if name == "sequence":
            return DEFAULT_SEQUENCE_TEMPLATE
        if name == "sequence_standalone":
            return DEFAULT_SEQUENCE_STANDALONE_TEMPLATE
        if name == "default_style":
            return DEFAULT_STYLE
        raise ValueError(f"no such template {name}")

    function_loader = FunctionLoader(load_template)
    env = Environment(loader=function_loader)
    return env.get_template(template_type, globals={"zip": zip})


def render_segment(schedule: Playlist, index: int, waveforms: dict[str, dict[int, str]]):  # noqa: ANN201
    """Function for rendering a segment to notebook."""
    path = Path(__file__).parent / "templates"
    file_loader = FileSystemLoader(path)
    env = Environment(loader=file_loader)
    template_file = "segment_inspection.jinja2"
    template = env.get_template(template_file, globals={"round": round})
    report = template.render(schedule=schedule, index=index, waveforms=waveforms)
    # Imported outside top level since notebook dependency brings ipython
    display_in_notebook(report)
