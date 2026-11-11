# Copyright 2022-2026 IQM
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

from dataclasses import replace

from iqm.error_reduction_tools import workflow as workflow_module
from iqm.error_reduction_tools.cdr import CDR, CDREstimation, plot_cdr_fit, plot_cdr_summary
from iqm.error_reduction_tools.technique import TargetEstimate
from iqm.error_reduction_tools.workflow import ErrorReductionWorkflow
from iqm.error_reduction_tools.zne import ZNE, ZNEEstimation, plot_zne_comparison, plot_zne_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from mockito import mock, unstub, when
import pytest

from .builders import bell_circuit, rotated_circuit
from .exact import ExactExecutor

mpl.use("Agg")


@pytest.fixture(scope="module")
def zne_workflow():
    # Run on exact counts instead of the ResonanceExecutor the workflow builds from its client.
    when(workflow_module).ResonanceExecutor(...).thenReturn(ExactExecutor(depolarizing=0.03))
    try:
        workflow = ErrorReductionWorkflow(
            [bell_circuit()], ["ZZ", "IZ"], ZNE(noise_factors=(1, 3, 5)), client=mock(), shots=2000
        )
        workflow.plan()
        workflow.submit()
        workflow.collect()
    finally:
        unstub()
    return workflow


def test_zne_fit_draws_one_panel_per_pauli_term(zne_workflow):
    """Test that plot_zne_fit draws one panel per Pauli term."""
    figure = plot_zne_fit(zne_workflow.estimate(ZNEEstimation(degree=1), bootstrap_samples=3), ideal=[1.0, 0.0])
    assert len([axes for axes in figure.axes if axes.get_visible()]) == 2
    plt.close(figure)


def test_zne_comparison_draws_every_configuration(zne_workflow):
    """Test that plot_zne_comparison draws every estimation in the legend."""
    results = {
        "linear": zne_workflow.estimate(ZNEEstimation(degree=1)),
        "Richardson": zne_workflow.estimate(ZNEEstimation()),
    }
    figure = plot_zne_comparison(results, pauli_term="ZZ")
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert sum(label.startswith(("linear", "Richardson")) for label in labels) == 2
    plt.close(figure)


def test_plots_refuse_diagnostics_of_the_other_technique(zne_workflow):
    """Test that the CDR plots reject ZNE results."""
    result = zne_workflow.estimate(ZNEEstimation(degree=1))
    with pytest.raises(ValueError, match="not estimated with CDREstimation"):
        plot_cdr_fit(result)
    with pytest.raises(ValueError, match="no target 3"):
        plot_zne_fit(result, target_index=3)
    figure = plot_cdr_summary(result)
    plt.close(figure)


def test_plots_refuse_targets_and_columns_that_do_not_exist(zne_workflow):
    """Test that the plots reject targets, Pauli terms and ideal values that do not match the result."""
    result = zne_workflow.estimate(ZNEEstimation(degree=1))
    with pytest.raises(ValueError, match="no target 3"):
        plot_cdr_fit(result, target_index=3)
    with pytest.raises(ValueError, match="No Pauli term 'XX'"):
        plot_zne_comparison({"linear": result}, pauli_term="XX")
    with pytest.raises(IndexError, match="Pauli term 2 is out of range"):
        plot_zne_comparison({"linear": result}, pauli_term=2)
    with pytest.raises(ValueError, match="Got 1 ideal values for 2 Pauli terms"):
        plot_zne_fit(result, ideal=[1.0])
    with pytest.raises(ValueError, match="not estimated with ZNEEstimation"):
        plot_zne_fit(replace(result, targets=(TargetEstimate(values=result.values[0], diagnostics=None),)))


def test_zne_comparison_needs_one_to_six_configurations(zne_workflow):
    """Test that plot_zne_comparison rejects zero or more than six estimations."""
    result = zne_workflow.estimate(ZNEEstimation(degree=1))
    with pytest.raises(ValueError, match="`results` is empty"):
        plot_zne_comparison({})
    with pytest.raises(ValueError, match="7 configurations share one panel"):
        plot_zne_comparison({str(index): result for index in range(7)})


def test_cdr_fit_refuses_a_column_that_does_not_exist():
    """Test that plot_cdr_fit rejects a column that does not exist."""
    when(workflow_module).ResonanceExecutor(...).thenReturn(ExactExecutor(depolarizing=0.03))
    try:
        workflow = ErrorReductionWorkflow([rotated_circuit(0.7)], "Z", CDR(num_training_circuits=3), client=mock())
        workflow.run(CDREstimation())
    finally:
        unstub()
    with pytest.raises(IndexError, match="Column 1 is out of range. This fit has 1"):
        plot_cdr_fit(workflow.result, columns=1)
