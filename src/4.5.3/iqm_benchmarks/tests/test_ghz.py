"""Tests for GHZ fidelity estimation using the new base class."""

from unittest.mock import MagicMock, patch

from iqm.benchmarks.entanglement.ghz import GHZBenchmark, GHZConfiguration
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb


class TestGHZ:
    """Tests for GHZ fidelity estimation benchmark."""

    backend = IQMFakeApollo()

    @patch("matplotlib.pyplot.figure")
    def test_layouts(self, mock_fig: MagicMock) -> None:
        """Test different qubit layouts for GHZ state generation."""
        minimal_ghz = GHZConfiguration(
            state_generation_routine="tree",
            custom_qubits_array=[
                [0, 1],
                [1, 3, 4],
                [1, 3, 4, 5],
            ],
            shots=3,
            qiskit_optim_level=3,
            optimize_sqg=True,
            fidelity_routine="coherences",
            num_rms=10,
            rem=False,
            mit_shots=10,
            use_dd=True,
        )
        benchmark = GHZBenchmark(self.backend, minimal_ghz)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

    @patch("matplotlib.pyplot.figure")
    def test_state_routine(self, mock_fig: MagicMock) -> None:
        """Test different state generation routines for GHZ state."""
        for gen_routine in ["tree", "naive", "log_depth", "star"]:
            minimal_ghz = GHZConfiguration(
                state_generation_routine=gen_routine,
                custom_qubits_array=[[2, 3, 4]],
                shots=3,
                qiskit_optim_level=3,
                optimize_sqg=True,
                fidelity_routine="coherences",
                num_rms=10,
                rem=False,
                mit_shots=10,
            )
            benchmark = GHZBenchmark(self.backend, minimal_ghz)
            benchmark.run()
            benchmark.analyze()
            mock_fig.assert_called()

    @patch("matplotlib.pyplot.figure")
    def test_rem(self, mock_fig: MagicMock) -> None:
        """Test readout error mitigation on/off for GHZ fidelity estimation."""
        for fidelity_routine in ["coherences", "randomized_measurements"]:
            minimal_ghz = GHZConfiguration(
                state_generation_routine="tree",
                custom_qubits_array=[[2, 3, 4]],
                shots=3,
                qiskit_optim_level=3,
                optimize_sqg=True,
                fidelity_routine=fidelity_routine,
                num_rms=10,
                rem=True,
                mit_shots=10,
            )
            benchmark = GHZBenchmark(self.backend, minimal_ghz)
            benchmark.run()
            benchmark.analyze()
            mock_fig.assert_called()


class TestGHZDeneb(TestGHZ):
    """Tests for GHZ fidelity estimation benchmark on Deneb backend."""

    backend = IQMFakeDeneb()
