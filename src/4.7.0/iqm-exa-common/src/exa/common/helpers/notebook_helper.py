# Copyright 2026 IQM
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

"""Jupyter notebook rendering helpers for :class:`.SettingNode`."""

from __future__ import annotations

from html import escape
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from exa.common.data.setting_node import SettingNode

_VALUE_COLORS = {"number": "lightgreen", "string": "orange", "none": "gray", "other": "inherit"}
"""Display colours per serialized value type."""


def value_type_and_display(value: Any) -> tuple[str, str]:
    """Classify a value and format it for display, mirroring the jinja template.

    Args:
        value: The (already SI-rescaled) value to classify and format.

    Returns:
        A ``(type, display)`` pair where ``type`` is one of ``none``, ``bool``,
        ``number``, ``string``, ``sequence`` or ``other``.

    """
    if value is None:
        return "none", "not set/auto"
    if isinstance(value, bool):
        return "bool", str(value)
    if isinstance(value, numbers.Real):
        return "number", "{:_.6g}".format(float(value)).replace("_", " ").strip()
    if isinstance(value, str):
        return "string", value
    if isinstance(value, (list, tuple, np.ndarray)):
        return "sequence", "[" + ", ".join(value_type_and_display(item)[1] for item in value) + "]"
    return "other", str(value)


def json_safe(value: Any) -> Any:
    """Convert a value into a JSON-serializable form for display tooltips.

    Args:
        value: Arbitrary setting value (may contain numpy types).

    Returns:
        A JSON-safe representation; non-convertible objects fall back to ``str``.

    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _settings_row(setting: dict[str, Any]) -> str:
    """Render one serialized setting as an HTML table row."""
    vtype = setting["type"]
    if vtype == "bool":
        color = "skyblue" if setting["display"] == "True" else "tomato"
    else:
        color = _VALUE_COLORS.get(vtype, "inherit")
    display = escape(str(setting["display"]))
    value_style = f"color:{color}"
    if vtype == "string":
        display = f'"{display}"'
        value_style += ";font-family:monospace"  # mirror the template's monospace string values
    row_style = ' style="color:gray"' if setting["read_only"] else ""
    title = escape(f"{setting['raw']} {setting['unit']}".strip())
    return (
        f"<tr{row_style}>"
        f'<td style="text-align:left"><span style="font-family:monospace;font-weight:bold" '
        f'title="{title}">{escape(str(setting["key"]))}</span></td>'
        f'<td><span style="{value_style}" title="{title}">{display}</span></td>'
        f'<td style="text-align:left;color:lightgray">{escape(str(setting["unit"]))}</td>'
        f'<td style="text-align:left">{escape(str(setting["label"]))}</td>'
        "</tr>"
    )


def node_header_html(key: str, name: str, cls: str) -> str:
    """Render the per-node header line (key + name + class)."""
    return (
        f'<span style="font-family:monospace;font-weight:bold">{escape(key)}:</span> '
        f'<span style="font-size:smaller;color:gray">(Name: "{escape(name)}", class: {escape(cls)})</span>'
    )


def settings_html(settings: list[dict[str, Any]]) -> str:
    """Render a single level's serialized settings as an HTML table (empty string if none)."""
    if not settings:
        return ""
    return '<table style="margin-left:1em">' + "".join(_settings_row(s) for s in settings) + "</table>"


def serialize_level(node: SettingNode) -> dict[str, Any]:
    """Serialize only ``node``'s own level (its settings and immediate children) for display.

    The children are described by metadata only (not recursed into), which is what makes the
    widget renderer lazy: deeper levels are serialized on demand when the user expands them.

    Args:
        node: The node whose own level is serialized.

    Returns:
        A dict with ``name``, ``cls``, ``settings`` and ``children`` keys.

    """
    settings = []
    for key, setting in node.settings.items():
        si_value, si_unit = node._withsiprefix(setting.value, setting.unit)
        vtype, display = value_type_and_display(si_value)
        settings.append(
            {
                "key": key,
                "label": setting.label,
                "type": vtype,
                "display": display,
                "unit": si_unit or "",
                "raw": json_safe(setting.value),
                "read_only": bool(setting.read_only),
            }
        )
    children = [
        {
            "key": key,
            "name": sub.name,
            "cls": sub.__class__.__qualname__,
            "has_children": bool(sub.settings or sub.subtrees),
        }
        for key, sub in node.subtrees.items()
    ]
    return {"name": node.name, "cls": node.__class__.__qualname__, "settings": settings, "children": children}


def fallback_html(node: SettingNode) -> str:
    """Cheap root-only HTML summary, used when no live widget manager is available."""
    return (
        '<p style="font-family:monospace">'
        f"SettingNode <b>{escape(node.name)}</b> ({escape(node.__class__.__qualname__)}): "
        f"{len(node.settings)} settings, {len(node.subtrees)} subtrees. "
        "Interactive view requires a running kernel with ipywidgets."
        "</p>"
    )


def fallback_text(node: SettingNode) -> str:
    """Plain-text root-only summary, used as the widget's ``text/plain`` degradation message.

    This goes in ``text/plain`` (the lowest-priority MIME) rather than ``text/html`` so that it never
    shadows the widget view when a widget manager is present, while still showing something readable in
    static / no-manager contexts.
    """
    return (
        f'SettingNode "{node.name}" ({node.__class__.__qualname__}): '
        f"{len(node.settings)} settings, {len(node.subtrees)} subtrees. "
        "Interactive view requires a running kernel with ipywidgets."
    )


def repr_mimebundle(
    node: SettingNode, include: set[str] | None = None, exclude: set[str] | None = None
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Backing implementation of :meth:`.SettingNode._repr_mimebundle_`.

    Builds the lazy ipywidgets renderer when ``ipywidgets`` is importable; otherwise returns a static
    root-only summary. The widget bundle also carries cheap ``text/html`` / ``text/plain`` fallbacks so
    non-widget frontends still show something.

    Args:
        node: The node to render.
        include: MIME types to include, forwarded to the widget's mimebundle.
        exclude: MIME types to exclude, forwarded to the widget's mimebundle.

    Returns:
        A MIME bundle mapping MIME type to representation.

    """
    try:
        from exa.common.helpers.widgets.setting_node_widget import (  # noqa: PLC0415 (import-outside-top-level)
            build_setting_node_widget,
        )
    except ImportError:
        return {"text/html": fallback_html(node)}

    widget = build_setting_node_widget(node)
    result = widget._repr_mimebundle_(include=include, exclude=exclude)
    # ``_repr_mimebundle_`` may return a plain ``{mime: data}`` dict, a ``(data, metadata)`` tuple, or
    # ``None`` (ipywidgets returns the dict form; the tuple is handled defensively). Add cheap
    # ``text/html`` / ``text/plain`` fallbacks so non-widget frontends still render something; the widget
    # view takes display precedence when a widget manager is present.
    if result is None:
        return {"text/html": fallback_html(node)}
    if isinstance(result, tuple):
        data, metadata = result
        data = dict(data)
        data["text/html"] = fallback_html(node)
        data["text/plain"] = fallback_text(node)
        return data, metadata
    data = dict(result)
    data["text/html"] = fallback_html(node)
    data["text/plain"] = fallback_text(node)
    return data
