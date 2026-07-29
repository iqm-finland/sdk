# Copyright 2023 Qiskit on IQM developers
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

"""Testing Bell measure example script."""

import subprocess
import sys

from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm.examples import transpile_example
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMProvider
from mockito import ANY, when
import pytest
import requests

from iqm.station_control.client.station_control import StationControlClient
from tests.conftest import MockJsonResponse

from .conftest import get_mocked_backend


def test_transpile_example_call():
    """Test that example script runs and fails at establishing a connection."""
    with subprocess.Popen(
        (sys.executable, transpile_example.__file__, "--url", "https://not.a.real.domain"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        (_, err) = p.communicate()
        p.wait()
        assert "ConnectionError" in str(err)


def test_transpile_example_disconnected_backend(disconnected_3q_architecture, request):
    """Test that transpile_example works when backend has a disconnected coupling map."""
    dqa = disconnected_3q_architecture
    # Mock the backend
    when(IQMProvider).get_backend().thenReturn(get_mocked_backend(dqa, request))
    when(IQMBackend).run(ANY, shots=ANY).thenRaise(
        Exception("Job submitted to mock in test_transpile_example_disconnected_backend.")
    )

    # Mock the IQMClient connection
    url = "http://some_url"
    calset_id = dqa.calibration_set_id
    when(requests).get(f"{url}/about", headers=ANY).thenReturn(MockJsonResponse(200, {}))
    when(StationControlClient)._check_api_versions().thenReturn(None)
    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(dqa)

    with pytest.raises(Exception, match="Job submitted to mock in test_transpile_example_disconnected_backend."):
        # Will crash because run doesn't do stuff, but transpilation worked at that point.
        transpile_example.transpile_example(url)
