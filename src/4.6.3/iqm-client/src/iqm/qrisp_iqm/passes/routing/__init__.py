# Copyright 2026 IQM
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

"""Routing and placement passes.

This submodule contains passes for qubit layout and routing operations,
including automatic placement algorithms (VF2++, SABRE) and manual layout options.
"""

from iqm.qrisp_iqm.passes.routing.plasma_layout import plasma_layout
from iqm.qrisp_iqm.passes.routing.plasma_route import plasma_route
from iqm.qrisp_iqm.passes.routing.vf2pp_layout import vf2pp_layout

__all__ = [
    "plasma_route",
    "vf2pp_layout",
    "plasma_layout",
]
