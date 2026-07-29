"""Tests for mirror RB."""

from iqm.benchmarks.randomized_benchmarking.clifford_rb.clifford_rb import (
    CliffordRandomizedBenchmarking,
    CliffordRBConfiguration,
)
from iqm.benchmarks.randomized_benchmarking.eplg.eplg import EPLGBenchmark, EPLGConfiguration
from iqm.benchmarks.randomized_benchmarking.interleaved_rb.interleaved_rb import (
    InterleavedRandomizedBenchmarking,
    InterleavedRBConfiguration,
)
from iqm.benchmarks.randomized_benchmarking.mirror_rb.mirror_rb import (
    MirrorRandomizedBenchmarking,
    MirrorRBConfiguration,
)
from iqm.benchmarks.utils import RoutingMethod
from iqm.qiskit_iqm import IQMFakeBackend
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
import numpy as np


class TestRB:
    """Test suite for randomized benchmarking methods."""

    backend: IQMFakeBackend = IQMFakeApollo()

    def test_mrb(self) -> None:
        """Test mirror randomized benchmarking."""
        example_mrb = MirrorRBConfiguration(
            qubits_array=[[2, 3], [3, 4]],
            depths_array=[[2**m for m in range(6)]],
            num_circuit_samples=2,
            num_pauli_samples=2,
            shots=2**8,
            qiskit_optim_level=1,
            routing_method=RoutingMethod.SABRE,
            two_qubit_gate_ensemble={"CZGate": 0.8, "iSwapGate": 0.2},
            density_2q_gates=0.25,
        )
        benchmark = MirrorRandomizedBenchmarking(self.backend, example_mrb)
        benchmark.run()
        benchmark.analyze()

    def test_irb(self) -> None:
        """Test interleaved randomized benchmarking."""
        example_irb_1q = InterleavedRBConfiguration(
            qubits_array=[[1]],
            sequence_lengths=[2 ** (m + 1) - 1 for m in range(6)],
            num_circuit_samples=5,
            shots=2**8,
            parallel_execution=False,
            interleaved_gate="RGate",
            interleaved_gate_params=[np.pi, 0],
            simultaneous_fit=["amplitude", "offset"],
        )
        benchmark = InterleavedRandomizedBenchmarking(self.backend, example_irb_1q)
        benchmark.run()
        benchmark.analyze()

    def test_crb(self) -> None:
        """Test Clifford randomized benchmarking."""
        example_crb_1q = CliffordRBConfiguration(
            qubits_array=[[3]],
            sequence_lengths=[2 ** (m + 1) - 1 for m in range(6)],
            num_circuit_samples=5,
            shots=2**8,
            parallel_execution=False,
        )
        benchmark = CliffordRandomizedBenchmarking(self.backend, example_crb_1q)
        benchmark.run()
        benchmark.analyze()

    def test_eplg(self) -> None:
        """Test EPLG benchmarking."""
        example_eplg = EPLGConfiguration(
            custom_qubits_array=((0, 1), (1, 4), (4, 5)),
            drb_depths=sorted(set(np.geomspace(1, 100, num=6, endpoint=True, dtype=int).tolist()), reverse=True),
            drb_circuit_samples=5,
            shots=2**8,
            chain_path_samples=1,
            num_disjoint_layers=2,
        )
        benchmark = EPLGBenchmark(self.backend, example_eplg)
        benchmark.run()
        benchmark.analyze()


class TestRBDeneb(TestRB):
    """Test suite for randomized benchmarking methods using Deneb backend."""

    backend: IQMFakeBackend = IQMFakeDeneb()
