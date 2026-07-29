# Copyright 2025 IQM
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

from mockito import unstub as mockito_unstub
from mockito import verifyNoUnwantedInteractions, verifyStubbedInvocationsAreUsed
import pytest


@pytest.fixture
def unstub_():
    """Guarantee that mockito.unstub() is used on teardown."""
    yield mockito_unstub
    mockito_unstub()


@pytest.fixture
def unstub(unstub_):
    """Additionally to mockito.unstub() ensures that stubs are actually used."""
    yield unstub_

    verifyStubbedInvocationsAreUsed()
    verifyNoUnwantedInteractions()


@pytest.fixture(scope="package")
def monkeypatch_package():
    """Same as monkeypatch, except package-scoped."""
    with pytest.MonkeyPatch.context() as patch_context:
        yield patch_context


@pytest.fixture(autouse=True, scope="package")
def silence_progressbar(monkeypatch_package):
    """disable the terminal progress bar."""
    monkeypatch_package.setattr(
        "iqm.station_control.client.utils.get_progress_bar_callback", lambda *args, **kwargs: None
    )
