"""Unit tests for GST post-processing functions."""

from copy import deepcopy
import unittest

from iqm.benchmarks.compressive_gst.mgst.additional_fns import random_gs_Haar, randU, randU_Haar
from iqm.benchmarks.compressive_gst.mgst.low_level_jit import objf_gauge
from iqm.benchmarks.compressive_gst.mgst.optimization import dU_gauge
from iqm.benchmarks.compressive_gst.mgst.utils_gst import average_gate_fidelities, basis_transform
import numpy as np


class TestGSTProcessing(unittest.TestCase):
    """Test cases for functions needed in GST post-processing."""

    def setUp(self) -> None:
        """Set up test environment."""
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        u_rand = randU_Haar(4)
        x_cnot = np.kron(cnot, cnot.T.conj())
        x_u_rand = np.kron(u_rand, u_rand.T.conj())
        _, x_rand, _, _ = random_gs_Haar(2, 16, 16, 2)
        self.x_cnot = np.array([x_cnot, x_cnot])
        self.x_u_rand = np.array([x_u_rand, x_u_rand])
        self.x_rand = x_rand

    def test_basis_transform(self) -> None:
        """Test basis transformation functions."""
        x_rand_pp = basis_transform(self.x_rand[0], "std", "pp")
        x_rand_ = basis_transform(x_rand_pp, "pp", "std")  # Back and forth transformed needs to be original
        assert np.linalg.norm(x_rand_ - self.x_rand[0]) < 1e-10
        assert x_rand_pp[0, 0] - 1 < 1e-10  # Test for trace preservation
        assert np.linalg.norm(x_rand_pp[0, 1:] - np.zeros(15)) < 1e-10  # Test for trace preservation

    def test_agf(self) -> None:
        """Test average gate fidelity calculation."""
        rand_agfs = average_gate_fidelities(self.x_rand, self.x_cnot)
        cnot_agfs = average_gate_fidelities(self.x_cnot, self.x_cnot)
        u_rand_agfs = average_gate_fidelities(self.x_u_rand, self.x_u_rand)
        mixed_agfs = average_gate_fidelities(self.x_cnot, self.x_u_rand)

        assert all(np.array(mixed_agfs) < 1) and all(np.array(mixed_agfs) >= 0)
        assert all(np.array(rand_agfs) < 1) and all(np.array(rand_agfs) > 0)
        assert cnot_agfs[0] > 1 - 1e-10
        assert u_rand_agfs[0] > 1 - 1e-10

    def test_gauge_derivative(self) -> None:
        """Test gauge derivative calculation via finite differences."""
        r = 16
        d = 3
        pdim = int(np.sqrt(r))
        weights = [1.0] * (d + 1)

        u = deepcopy(randU(pdim, a=1))
        u_channel = np.kron(u, u.T.conj())
        _, x_t, povm_t, rho_t = deepcopy(random_gs_Haar(d, r, r, 2))
        x = np.array([u_channel.T.conj() @ x_t[i] @ u_channel for i in range(x_t.shape[0])])
        povm = np.array([(povm_t[i].conj() @ u_channel).conj() for i in range(povm_t.shape[0])])
        rho = u_channel.T.conj() @ rho_t

        du = dU_gauge(x, povm, rho, x_t, povm_t, rho_t, u, weights=weights)

        trial_a = [-1e-3, 0, 1e-3]
        f_vals = [objf_gauge(x, povm, rho, x_t, povm_t, rho_t, u + a_ * du, weights=weights) for a_ in trial_a]
        derivative = 2 * np.trace(du @ du.T).real
        finite_differences = [(f_vals[1] - f_vals[0]) / 1e-3, (f_vals[2] - f_vals[1]) / 1e-3]
        assert (finite_differences[0] <= derivative <= finite_differences[1]) or (
            finite_differences[0] >= derivative >= finite_differences[1]
        )


if __name__ == "__main__":
    unittest.main()
