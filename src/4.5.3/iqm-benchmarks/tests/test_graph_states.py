"""Tests for graph state entanglement benchmark."""

from unittest.mock import MagicMock, patch

from iqm.benchmarks.entanglement import graph_states
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo


class TestGraphState:
    """Tests for graph state entanglement benchmark."""

    backend = IQMFakeApollo()

    @patch("matplotlib.pyplot.figure")
    def test_state_tomo(self, mock_fig: MagicMock) -> None:
        """Testing state tomography analysis."""
        minimal_graphstate = graph_states.GraphStateConfiguration(
            qubits=[3, 4, 5],
            shots=2**10,
            tomography=graph_states.TomographyType.STATE,
            num_bootstraps=10,
        )
        benchmark = graph_states.GraphStateBenchmark(self.backend, minimal_graphstate)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

    @patch("matplotlib.pyplot.figure")
    def test_shadows(self, mock_fig: MagicMock) -> None:
        """Testing shadow tomography analysis."""
        minimal_graphstate = graph_states.GraphStateConfiguration(
            qubits=list(range(5)),
            shots=2**6,
            tomography=graph_states.TomographyType.SHADOW,
            num_bootstraps=2,
            n_random_unitaries=10,
            n_median_of_means=2,
        )
        benchmark = graph_states.GraphStateBenchmark(self.backend, minimal_graphstate)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()
