# Copyright 2025 IQM
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
"""Utility functions for IQM Station Control Client."""

from datetime import datetime
from html import escape
from importlib import import_module

from tqdm import tqdm

from iqm.station_control.interface.models.jobs import ProgressCallback, _Progress


def _is_notebook_session() -> bool:
    try:
        shell = import_module("IPython.core.getipython").get_ipython()
    except ImportError:
        return False

    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds_part = divmod(total_seconds, 60)
    hours, minutes_part = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes_part:02d}:{seconds_part:02d}"
    return f"{minutes_part:02d}:{seconds_part:02d}"


def _format_rate(rate: float) -> str:
    if rate >= 100:
        return f"{rate:.0f}"
    if rate >= 10:
        return f"{rate:.1f}"
    return f"{rate:.2f}"


def _render_progress_bar(
    label: str,
    value: int,
    total: int,
    *,
    elapsed_secs: float,
    rate: float,
    remaining_secs: float,
) -> str:
    safe_total = max(total, 1)
    percent = 100 * value / safe_total
    elapsed_text = _format_duration(elapsed_secs)
    remaining_text = _format_duration(remaining_secs) if remaining_secs >= 0 else "??:??"
    rate_text = _format_rate(rate)
    return (
        "<div style='margin: 0.4rem 0;'>"
        f"<div style='font-family: monospace;'>{escape(label)}: {value}/{total}</div>"
        "<div style='width: 100%; background: #e5e7eb; border-radius: 999px; overflow: hidden;'>"
        f"<div style='width: {percent:.1f}%; height: 0.7rem; background: #2563eb;'></div>"
        "</div>"
        f"<div style='font-family: monospace; color: #4b5563; font-size: 0.9em;'>"
        f"[{elapsed_text}<{remaining_text}, {rate_text}it/s]"
        "</div>"
        "</div>"
    )


def get_progress_bar_callback() -> ProgressCallback:
    """Return a callback that creates or updates progress bars from status updates.

    Notebook rendering keeps one display handle per status label so multiple bars can
    update in place at the same time. Labels therefore need to be unique across
    concurrently active loops; if two loops reuse the same label they will update the
    same rendered bar.
    """
    use_notebook_display = _is_notebook_session()
    ipython_display = import_module("IPython.display") if use_notebook_display else None
    progress_bars = {}
    notebook_progress_bars = {}
    notebook_started_at: dict[str, datetime] = {}

    def _create_and_update_progress_bars(statuses: list[_Progress]) -> None:
        for label, value, total in statuses:
            if use_notebook_display and ipython_display is not None:
                now = datetime.now()
                if label not in notebook_started_at:
                    notebook_started_at[label] = now

                elapsed_secs = (now - notebook_started_at[label]).total_seconds()
                rate = value / elapsed_secs if elapsed_secs > 1e-6 else 0.0
                remaining_steps = max(total - value, 0)
                remaining_secs = remaining_steps / rate if rate > 0 else -1.0

                progress_html = ipython_display.HTML(
                    _render_progress_bar(
                        label,
                        value,
                        total,
                        elapsed_secs=elapsed_secs,
                        rate=rate,
                        remaining_secs=remaining_secs,
                    )
                )
                if label not in notebook_progress_bars:
                    notebook_progress_bars[label] = ipython_display.display(progress_html, display_id=True)
                else:
                    notebook_progress_bars[label].update(progress_html)
                continue

            if label not in progress_bars:
                progress_bars[label] = tqdm(total=total, desc=label, leave=True)
            progress_bars[label].n = value
            progress_bars[label].refresh()

    return _create_and_update_progress_bars
