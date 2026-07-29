"""Tests for compressive GST benchmark."""

import multiprocessing as mp
from unittest.mock import MagicMock, patch

from iqm.benchmarks.compressive_gst.compressive_gst import CompressiveGST, GSTConfiguration
from iqm.qiskit_iqm import IQMFakeBackend
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
import numpy as np
from qiskit.circuit import QuantumCircuit


class TestGST:
    """Test suite for compressive GST benchmark."""

    backend: IQMFakeBackend = IQMFakeApollo()

    @staticmethod
    def is_positive(gates: np.ndarray, povm: np.ndarray, rho: np.ndarray) -> None:
        """Print the results for checks whether a gate set is physical.

        This includes all positivity and normalization constraints.

        Parameters
        ----------
        gates : numpy array
            Gate set
        povm : numpy array
            POVM
        rho : numpy array
            Initial state

        """
        d, r, _ = gates.shape
        pdim = int(np.sqrt(r))
        n_povm = povm.shape[0]

        gates_choi = gates.reshape(d, pdim, pdim, pdim, pdim)
        gates_choi = np.einsum("ijklm->iljmk", gates_choi).reshape((d, r, r))

        eigvals = np.array([np.linalg.eigvals(gates_choi[i]) for i in range(d)])
        partial_traces = np.einsum("aiikl -> akl", gates.reshape(d, pdim, pdim, pdim, pdim))
        povm_eigvals = np.array([np.linalg.eigvals(povm[i].reshape(pdim, pdim)) for i in range(n_povm)])

        tolerance = 1e-10
        povm_tolerance = 1e-6

        # Check if gates are hermitian
        assert np.all(np.imag(eigvals.reshape(-1)) < tolerance)

        # Check if gates are positive and trace preserving
        for i in range(d):
            assert np.all(eigvals[i, :] > -tolerance)
            assert np.linalg.norm(partial_traces[i] - np.eye(pdim)) < tolerance

        # Initial state positivity and normalization
        assert np.all(np.linalg.eigvals(rho.reshape(pdim, pdim)) > -tolerance)
        assert np.abs(np.trace(rho.reshape(pdim, pdim))) - 1 < tolerance

        # POVM positivity and normalization
        assert np.linalg.norm(np.sum(povm, axis=0).reshape(pdim, pdim) - np.eye(pdim)) < povm_tolerance
        assert np.all(povm_eigvals.reshape(-1) > -tolerance)

    @patch("matplotlib.pyplot.figure")
    def test_1q(self, mock_fig: MagicMock) -> None:
        """Test 1-qubit GST benchmark."""
        mp.set_start_method("spawn", force=True)
        # Testing minimal gate context
        gate_context = [QuantumCircuit(self.backend.num_qubits) for _ in range(3)]
        gate_context[0].h(0)
        gate_context[1].s(0)
        # config
        qubit_layouts = [[4], [1]]
        minimal_1q_config = GSTConfiguration(
            qubit_layouts=[[4], [1]],
            gate_set="1QXYI",
            gate_context=gate_context,
            num_circuits=5,
            shots=100,
            rank=4,
            bootstrap_samples=4,
            max_iterations=[10, 10],
            parallel_execution=True,
            fixed_elements=["G1"],
        )
        benchmark = CompressiveGST(self.backend, minimal_1q_config)
        benchmark.run()
        result = benchmark.analyze()

        for layout in qubit_layouts:
            # Check if all metrics are between 0 and 1 (as currently all metrics are normalized)
            assert all(0 < metric.value < 1 for metric in result.observations)

            # Check if outcomes satisfy physicality constraints
            gates = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_state"]
            povm = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_POVM"]
            self.is_positive(gates, povm, rho)  # raises error if any gate set element is not physical

            gates = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_state"]
            povm = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_POVM"]
            self.is_positive(gates, povm, rho)  # raises error if any gate set element is not physical

            # Check if gates in the Pauli basis have real entries
            gauge_opt_gates_pp = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_gates_Pauli_basis"]
            assert np.all(np.imag(gauge_opt_gates_pp.reshape(-1)) < 1e-10)

            # Check if GST estimate fits data better than target gate set
            tvd_estimate = float(
                result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"][
                    "mean_tvd_estimate_data"
                ][""].split(" ")[0]
            )
            tvd_target = float(
                result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"][
                    "mean_tvd_target_data"
                ][""].split(" ")[0]
            )
            assert tvd_estimate < tvd_target

        mock_fig.assert_called()

    @patch("matplotlib.pyplot.figure")
    def test_2q(self, mock_fig: MagicMock) -> None:
        """Test 2-qubit GST benchmark."""
        mp.set_start_method("spawn", force=True)
        qubit_layouts = [[2, 3]]
        minimal_2q_gst = GSTConfiguration(
            qubit_layouts=qubit_layouts,
            gate_set="2QXYICZ",
            num_circuits=100,
            shots=100,
            rank=1,
            bootstrap_samples=2,
            max_iterations=[10, 10],
        )
        benchmark = CompressiveGST(self.backend, minimal_2q_gst)
        benchmark.run()
        result = benchmark.analyze()

        for layout in qubit_layouts:
            # Check if all metrics are between 0 and 1 (as currently all metrics are normalized)
            assert all(0 < metric.value < 1 for metric in result.observations)

            # Check if outcomes satisfy physicality constraints
            gates = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_state"]
            povm = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_POVM"]
            self.is_positive(gates, povm, rho)  # raises error if any gate set element is not physical

            gates = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_state"]
            povm = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_POVM"]
            self.is_positive(gates, povm, rho)  # raises error if any gate set element is not physical

            # Check if gates in the Pauli basis have real entries
            gauge_opt_gates_pp = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_gates_Pauli_basis"]
            assert np.all(np.imag(gauge_opt_gates_pp.reshape(-1)) < 1e-10)

            # Check if GST estimate fits data better than target gate set
            tvd_estimate = float(
                result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"][
                    "mean_tvd_estimate_data"
                ][""].split(" ")[0]
            )
            tvd_target = float(
                result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"][
                    "mean_tvd_target_data"
                ][""].split(" ")[0]
            )
            assert tvd_estimate < tvd_target

        mock_fig.assert_called()


class TestGSTDeneb(TestGST):
    """Test suite for compressive GST benchmark using Deneb backend."""

    backend: IQMFakeBackend = IQMFakeDeneb()
