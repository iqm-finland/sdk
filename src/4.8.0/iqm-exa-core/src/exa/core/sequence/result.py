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
"""Class for the sequence-level analysis output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exa.common.errors import ValidationError
from exa.core.plotting import Figure
from exa.core.util.report import display_in_notebook, get_template, save_to_file

from .interface import SequenceAnalyzeResultData, SequenceExecuteResultData


class SequenceExecuteResult(SequenceExecuteResultData):
    """Contains the results of a sequence executions, and can analyze the data."""

    def analyze(self, **kwargs) -> SequenceAnalyzeResult:
        """Analyze data in self.

        The keyword arguments are passed to :attr:`.SequenceData.analysis`. Possible observations are tagged with
        tags from the context.

        Returns:
            The output of :attr:`.SequenceData.analysis`.

        """
        result: SequenceAnalyzeResult = self.context.sequence.analysis(self, **kwargs)  # type: ignore[assignment]
        result.observations.validate()  # type: ignore[union-attr]
        result.observations.tag(self.context.tags)  # type: ignore[union-attr]
        return result


@dataclass(repr=False)
class SequenceAnalyzeResult(SequenceAnalyzeResultData):
    """The analyzed end result of a Sequence.

    Analogous to the :class:`AnalyzeResult` of Experiments. Contains the dataset, observations, and plots from the
    sequence-level analysis, as well as the execution result where it was all derived from.

    Can summarize the results as a report.
    """

    def render_report(self, target: Literal["notebook", "str"] | Path = "notebook", **kwargs) -> str | None:
        """Generate a report that displays the observations and plots from the analysis.

        Args:
            target: The target format as string or a file path. Accepted values: "notebook", "str" or :class:`Path`
                instance. For "notebook", the output is a  HTML document that is displayed in a Jupyter
                notebook. Target "str" returns HTML format in a string, and path instances saves
                document to a file.
            kwargs: Extra options passed to the template.

        Returns:
            Rendered string or None if the report is rendered in notebook as a side effect.

        """
        template = get_template("sequence" if target == "notebook" else "sequence_standalone")
        report = template.render(result=self, options=kwargs)

        if target == "notebook":
            return display_in_notebook(report)

        if target == "str":
            return report

        if isinstance(target, Path):
            return save_to_file(target, report)

        raise ValueError(f"Invalid report target {target}")

    def plot(self, key: str = "default", **kwargs) -> Figure:
        """Generate and show a selected plot from :attr:`.SequenceAnalyzeResultData.plots`.

        Args:
            key: Selects which plot to show. If "default" and no "default" key exists, returns the first one.
            kwargs: passed to the selected plot-generating function.

        Returns:
            The generated Figure.

        Raises:
            ValidationError: if no plots exist in :attr:`plots`.
            KeyError: if `key`  does not exist in :attr:`plots`.

        """
        if not self.plots:
            raise ValidationError("No plots defined.")
        if key == "default" and key not in self.plots:
            key = next(iter(self.plots.keys()))
        fig = self.plots[key].generate_from_dataset(self.dataset, **kwargs)
        return fig

    def plot_all(self, keys: list[str] | None = None, **kwargs) -> list[Figure]:
        """Generate and show all plots from :attr:`.SequenceAnalyzeResultData.plots`.

        Args:
            keys: List of plots to show. If None, all the plots will be shown.
            kwargs: passed to the selected plot-generating function.

        Returns:
            The generated Figures.

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
