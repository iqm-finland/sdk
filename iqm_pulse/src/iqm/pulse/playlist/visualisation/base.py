#  ********************************************************************************
#
# Copyright 2024 IQM
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
"""Function for visualising playlists."""

import base64
from collections.abc import Sequence
from dataclasses import asdict
import html
import io
import json
from pathlib import Path
from typing import Any

from iqm.models.playlist.waveforms import Waveform
from jinja2 import Environment, FileSystemLoader
import numpy as np

from iqm.pulse.playlist.playlist import Playlist


def _numpy_to_builtin_types(data: list[tuple[str, Any]]):  # noqa: ANN202
    """Convert selected ``numpy`` types to Python's built-in types.

    This helper function is to be used for converting dataclasses into
    dictionaries with ``asdict``.

    For example, ``Playlist`` dataclasses can get converted to dictionaries to
    be rendered as HTML/JavaScript in "playlist inspector". If such classes
    were to contain ``numpy`` types they could not be handled by JavaScript.
    """
    converted_data = {}

    for key, value in data:
        if isinstance(value, np.floating):
            converted_data[key] = float(value)
        elif isinstance(value, np.ndarray):
            converted_data[key] = value.tolist()
        else:
            converted_data[key] = value

    return converted_data


def _get_waveform(wave: Waveform, scale: float, wave_q: Waveform | None = None, scale_q: float | None = None) -> str:
    import matplotlib.pyplot as plt

    def fig_to_base64(fig):  # noqa: ANN001, ANN202
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)

        return base64.b64encode(img.getvalue())

    scaled_wave = wave.sample() * scale
    if wave_q:
        scaled_wave_q = wave_q.sample() * scale_q
    html_text = ""
    with plt.ioff():
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim([-1, 1])
        ax.plot(scaled_wave, marker=".")
        if wave_q:
            ax.plot(scaled_wave_q, marker=".")
        encoded = fig_to_base64(fig)
        my_html = '<img class="waveform-image" src="data:image/png;base64, {} ">'.format(encoded.decode("utf-8"))
        html_text = my_html
        plt.close(fig)
    return html_text


def _playlist_as_a_dict(playlist: Playlist, segment_indices: Sequence[int]) -> dict:
    playlists: list = []
    waveform_dict: dict[str, dict] = {}
    for channel in playlist.channel_descriptions:
        waveform_dict[channel] = {}

    for idx in segment_indices:
        segment = playlist.segments[idx]
        schedule_dict: dict[str, dict] = {}
        for channel, instruction_list in segment.instructions.items():
            schedule_dict[channel] = {}
            channel_desc = playlist.channel_descriptions.get(channel, None)
            instructions: list[dict] = []
            for instruction_idx in instruction_list:
                instruction = channel_desc.instruction_table[instruction_idx]
                instruction_dict = {}
                instruction_dict["name"] = instruction.operation.__class__.__name__
                params = asdict(instruction.operation, dict_factory=_numpy_to_builtin_types)
                if instruction.operation.__class__.__name__ == "RealPulse":
                    instruction_dict["wave_img_idx"] = instruction_idx
                    if str(instruction_idx) not in waveform_dict[channel]:
                        waveform_dict[channel][str(instruction_idx)] = _get_waveform(
                            instruction.operation.wave, instruction.operation.scale
                        )
                elif instruction.operation.__class__.__name__ == "IQPulse":
                    instruction_dict["wave_img_idx"] = instruction_idx
                    if str(instruction_idx) not in waveform_dict[channel]:
                        waveform_dict[channel][str(instruction_idx)] = _get_waveform(
                            instruction.operation.wave_i,
                            instruction.operation.scale_i,
                            instruction.operation.wave_q,
                            instruction.operation.scale_q,
                        )
                elif instruction.operation.__class__.__name__ == "ReadoutTrigger":
                    if instruction.operation.probe_pulse.operation.__class__.__name__ == "MultiplexedIQPulse":
                        params = {
                            "probe_pulse": instruction.operation.probe_pulse.operation.__class__.__name__,
                            "entries": [
                                json.dumps(
                                    {
                                        "name": entry[0].operation.__class__.__name__,
                                        **asdict(entry[0].operation, dict_factory=_numpy_to_builtin_types),
                                    },
                                    indent=2,
                                )
                                for entry in instruction.operation.probe_pulse.operation.entries
                            ],
                        }
                params["duration"] = round(
                    instruction.duration_samples / channel_desc.channel_config.sampling_rate * 1e9, 3
                )

                instruction_dict["params"] = params
                instructions.append(instruction_dict)
            schedule_dict[channel]["instructions"] = instructions

        # squeeze start and end waits
        for squeeze_idx in (0, -1):
            common_wait = np.inf
            for channel, content in schedule_dict.items():
                if "instructions" in content and content["instructions"][squeeze_idx]["name"] == "Wait":
                    common_wait = min(common_wait, content["instructions"][squeeze_idx]["params"]["duration"])
                else:
                    common_wait = 0
                    break
            if common_wait:
                for channel, content in schedule_dict.items():
                    content["instructions"][squeeze_idx]["params"]["duration"] -= common_wait
                    qualif = "end" if squeeze_idx == -1 else "start"
                    common_wait_instr = {
                        "name": f"Wait at {qualif}",
                        "params": {"duration": common_wait, "truncated_duration": 150},
                    }
                    content["instructions"].insert(squeeze_idx, common_wait_instr)
        playlists.append(schedule_dict)
    ret_dict = {"playlists": playlists, "waveforms": waveform_dict}

    return ret_dict


def inspect_playlist(playlist: Playlist, segments: Sequence[int] = (0,)) -> str:
    """Creates an HTML string from the given playlist and segments.

    The output can be viewed in a browser or in a Jupyter notebook using ``IPython.core.display.HTML``.

    Args:
        playlist: The Playlist to be visualised
        segments: Indices of the Playlist segments to inspect.

    Returns:
       The generated raw HTML string.

    """
    path = Path(__file__).parent / "templates"
    file_loader = FileSystemLoader(path)
    env = Environment(loader=file_loader)
    template_file = "playlist_inspection.jinja2"
    for value in segments:
        try:
            playlist.segments[value]
        except IndexError as exc:
            end_range = ""
            if len(playlist.segments) > 1:
                end_range = f"-{len(playlist.segments) - 1}"
            raise IndexError(f"Index '{value}' not in range of segments 0{end_range}") from exc
    template = env.get_template(template_file, globals={"round": round})
    json_format = _playlist_as_a_dict(playlist=playlist, segment_indices=segments)
    vis_js_style = Path(path / "static" / "vis-timeline-graph2d.min.css", encoding="utf-8").read_text(encoding="utf-8")
    vis_js_script = Path(path / "static" / "vis-timeline-graph2d.min.js", encoding="utf-8").read_text(encoding="utf-8")
    moment_js_script = Path(path / "static" / "moment.min.js", encoding="utf-8").read_text(encoding="utf-8")

    html_text = template.render(
        jsonobj=json_format,
        VisJsScript=vis_js_script,
        VisJsStyle=vis_js_style,
        MomentJS=moment_js_script,
        segment_indices=list(segments),
    )
    return f"""<iframe allowfullscreen="true"
        style="background: #F4F0EA; width: 100%; height: 1000;"
        width=1000 height=600
        srcdoc="{html.escape(html_text)}"></iframe>"""
