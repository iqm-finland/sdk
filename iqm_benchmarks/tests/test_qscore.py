"""Tests for Qscore estimation."""

from iqm.benchmarks.optimization import qscore
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb


class TestQScore:
    """QScore test definitions."""

    backend = IQMFakeApollo()
    custom_qubits_array = [[0, 1, 2, 3], [0, 1, 2, 3, 4]]

    def test_qscore(self) -> None:
        """Test QScore run and analysis."""
        example_qscore = qscore.QScoreConfiguration(
            num_instances=2,
            num_qaoa_layers=1,
            shots=4,
            min_num_nodes=4,
            max_num_nodes=5,
            use_virtual_node=True,
            use_classically_optimized_angles=True,
            choose_qubits_routine="custom",
            custom_qubits_array=self.custom_qubits_array,
            seed=200,
            num_trials=2,
            REM=True,
            mit_shots=10,
        )
        benchmark = qscore.QScoreBenchmark(self.backend, example_qscore)
        benchmark.run()
        benchmark.analyze()


class TestQScoreDeneb(TestQScore):
    """Tests for Qscore on start architecture."""

    backend = IQMFakeDeneb()
    custom_qubits_array = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
