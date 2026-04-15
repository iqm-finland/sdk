# Copyright 2024-2025 IQM
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

"""Check that custom implementations are identified."""

import pytest

from iqm.cpc.core.config import _is_custom_implementation


@pytest.mark.parametrize(
    "name,result",
    [
        ("", False),
        ("CZ_GaussianSmoothedSquare", False),
        ("my_module.py::MyClass", True),
        ("directory/my_module.py::MyClass", True),
    ],
)
def test_custom_implementation(name, result):
    """Custom implementation names are recognized correctly."""
    assert _is_custom_implementation(name) is result
