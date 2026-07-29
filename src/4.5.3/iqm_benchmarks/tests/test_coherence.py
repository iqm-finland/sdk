"""Tests for coherence estimation."""

from iqm.benchmarks.coherence.coherence import CoherenceBenchmark, CoherenceConfiguration
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
import numpy as np


class TestCoherence:
    """Tests for coherence estimation benchmark."""

    backend = IQMFakeApollo()

    def test_coherence_t1(self) -> None:
        """Test T1 coherence benchmark."""
        example_coherence = CoherenceConfiguration(
            delays=list(np.linspace(0, 100e-6, 20)),
            qiskit_optim_level=3,
            optimize_sqg=True,
            coherence_exp="t1",
            qubits_to_plot=list(range(self.backend.num_qubits)),
        )
        benchmark = CoherenceBenchmark(self.backend, example_coherence)
        benchmark.run()
        benchmark.analyze()

    def test_coherence_t2_echo(self) -> None:
        """Test T2 Echo coherence benchmark."""
        example_coherence = CoherenceConfiguration(
            delays=list(np.linspace(0, 100e-6, 20)),
            qiskit_optim_level=3,
            optimize_sqg=True,
            coherence_exp="t2_echo",
            qubits_to_plot=list(range(self.backend.num_qubits)),
        )
        benchmark = CoherenceBenchmark(self.backend, example_coherence)
        benchmark.run()
        benchmark.analyze()


class TestCoherenceDeneb(TestCoherence):
    """Tests for coherence estimation benchmark on Deneb backend."""

    backend = IQMFakeDeneb()
