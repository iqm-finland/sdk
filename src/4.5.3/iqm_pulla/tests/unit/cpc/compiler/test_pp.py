"""Test collection of utilities for post processing."""

import numpy as np
import pytest
import xarray as xr

from exa.common.data.parameter import Parameter
from iqm.cpc.compiler._utils.post_process import (
    AverageResponse,
    compute_excitation_probability,
    principal_component_analysis,
)


def test_identical_avg():
    with pytest.raises(ValueError, match="not possible because of division by zero"):
        compute_excitation_probability(
            xr.DataArray(data=np.ones((3, 4, 5))), AverageResponse(0, 0.5, 0.5), Parameter("excited_state_propability")
        )


def test_simple():
    arr = compute_excitation_probability(
        xr.DataArray(data=np.ones((3, 4, 5))),
        AverageResponse(0.25 * np.pi, 0.6, 0.4),
        Parameter("excited_state_propability"),
    )
    assert arr.shape == (3, 4, 5)


def test_pca():
    arr = (1 + 1j) * np.arange(1, 6)
    pca = principal_component_analysis(arr)
    assert np.allclose(pca[1], np.sqrt(2) * np.arange(1, 6))
