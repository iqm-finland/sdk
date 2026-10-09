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

"""Lazy Jupyter renderer for :class:`.SettingNode` built from stock ``ipywidgets``.

A :class:`.SettingNode` tree can be huge (tens of thousands of settings for a 50+ qubit system), and
rendering the whole tree up front is slow and produces a very large DOM. This builds a collapsible view
from stock ipywidgets: each subtree is a ``Button`` (styled as a summary row) plus a hidden ``VBox`` body
that is only built the first time the user opens it, and wide levels are paginated with a "Show more"
button, so the initial display cost is proportional to the root level alone.

It uses only core ipywidgets (``Button``/``VBox``/``HTML``), which are bundled into the Jupyter frontends
(incl. VS Code) - no custom JavaScript, no bundling, no comm wiring. Styling is done through our own
scoped CSS classes (``exa-*``) rather than the frontend's internal widget classes, so it is stable across
frontends.

This module imports ``ipywidgets`` at import time. It is imported lazily by
:func:`exa.common.helpers.notebook_helper.repr_mimebundle`, which falls back to a static summary when
``ipywidgets`` is missing, so importing :mod:`exa.common.data.setting_node` never requires ``ipywidgets``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import ipywidgets as widgets  # type: ignore[import-untyped]

from exa.common.helpers.notebook_helper import node_header_html, serialize_level, settings_html

if TYPE_CHECKING:
    from exa.common.data.setting_node import SettingNode

# Scoped styling: every rule is prefixed with ``.exa-settingnode`` so it only affects our subtree, and
# only our own ``exa-*`` classes are targeted (never the frontend's internal widget classes). The
# buttons are flattened into <details>-like summary rows, with a CSS ``::before`` triangle toggled by the
# ``exa-open`` class (codepoint escapes keep the triangle out of the Python source).
_STYLE = """
<style>
.exa-settingnode { line-height: 1.35; }
.exa-settingnode .jupyter-widgets { margin: 0 !important; }
.exa-settingnode .exa-summary,
.exa-settingnode .exa-more {
  width: auto !important;
  min-height: 0 !important;
  height: auto !important;
  line-height: 1.5 !important;
  margin: 0 !important;
  padding: 0 .3em !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  cursor: pointer;
}
.exa-settingnode .exa-summary { text-align: left; color: inherit !important; font-family: monospace; }
.exa-settingnode .exa-summary::before { content: "\\25B8\\00a0"; color: gray; }
.exa-settingnode .exa-summary.exa-open::before { content: "\\25BE\\00a0"; }
.exa-settingnode .exa-summary:hover { text-decoration: underline; }
.exa-settingnode .exa-more { color: #2196f3 !important; font-style: italic; text-align: left; }
.exa-settingnode .exa-body { padding-left: 1em; }
.exa-settingnode .exa-root-header { font-family: monospace; }
.exa-settingnode table { margin: 0 0 0 1em; border-collapse: collapse; }
.exa-settingnode td { padding: .12em .5em; line-height: 1.35; }
</style>
"""

_DEFAULT_CHUNK = 50
"""Number of child sections rendered per "Show more" step (override with ``EXA_SETTINGNODE_CHUNK``)."""


def build_setting_node_widget(node: SettingNode) -> widgets.Widget:
    """Build a lazy, collapsible ipywidgets view of ``node``.

    Only ``node``'s own level is built immediately (its settings plus one collapsed button per child
    subtree); each child level is built on demand the first time its button is clicked, and cached.

    Args:
        node: The root :class:`.SettingNode` to render.

    Returns:
        An ipywidgets widget suitable for display in a Jupyter frontend.

    """
    header = widgets.HTML(
        f'<div class="exa-root-header">{node_header_html(node.name, node.name, node.__class__.__qualname__)}</div>'
    )
    container = widgets.VBox([widgets.HTML(_STYLE), header, _make_box(node)])
    container.add_class("exa-settingnode")
    return container


def _make_box(node: SettingNode) -> widgets.Widget:
    """Render a single node level: its settings as HTML plus a lazy, chunked section per child subtree.

    To keep wide levels (hundreds of children) responsive, only a chunk of child sections is created up
    front; if more remain, a single "Show more" button appends the next chunk on demand. The chunk size
    is ``EXA_SETTINGNODE_CHUNK`` (default ``_DEFAULT_CHUNK``).
    """
    prefix: list[widgets.Widget] = []
    table = settings_html(serialize_level(node)["settings"])
    if table:
        prefix.append(widgets.HTML(table))
    pending = list(node.subtrees.items())
    if not prefix and not pending:
        return widgets.HTML("<i>(empty)</i>")

    chunk = max(1, int(os.environ.get("EXA_SETTINGNODE_CHUNK", str(_DEFAULT_CHUNK))))
    box = widgets.VBox()
    sections: list[widgets.Widget] = []
    more_button = widgets.Button()
    more_button.add_class("exa-more")
    cursor = {"index": 0}

    # Note that this local function binds sections and cursor and mutates them
    # in-fly so that we can maximize caching of the already pre-computed widgets
    def show_more(_button: widgets.Button | None = None) -> None:
        end = min(cursor["index"] + chunk, len(pending))
        for key, sub in pending[cursor["index"] : end]:
            sections.append(_make_child_section(key, sub))
        cursor["index"] = end
        remaining = len(pending) - end
        tail = []
        if remaining:
            more_button.description = f"Show {min(chunk, remaining)} more  ({remaining} remaining)"
            tail = [more_button]
        box.children = tuple(prefix + sections + tail)

    # Bind click listener
    more_button.on_click(show_more)
    # Populate first batch of children
    show_more()
    return box


def _make_child_section(key: str, node: SettingNode) -> widgets.Widget:
    """Build a collapsible section for one child subtree: a summary button and a lazily-built body."""
    summary = f'{key}   (Name: "{node.name}", class: {node.__class__.__qualname__})'
    button = widgets.Button(description=summary, layout=widgets.Layout(width="auto"))
    button.add_class("exa-summary")
    body = widgets.VBox([])
    body.add_class("exa-body")
    body.layout.display = "none"
    state = {"open": False, "loaded": False}

    def on_click(
        _button: object = None,
        _node: SettingNode = node,
        _body: widgets.VBox = body,
        _summary_button: widgets.Button = button,
        _state: dict = state,
    ) -> None:
        _state["open"] = not _state["open"]
        if _state["open"]:
            if not _state["loaded"]:
                _body.children = (_make_box(_node),)
                _state["loaded"] = True
            _body.layout.display = ""
            _summary_button.add_class("exa-open")
        else:
            _body.layout.display = "none"
            _summary_button.remove_class("exa-open")

    button.on_click(on_click)
    return widgets.VBox([button, body])
