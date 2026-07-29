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

"""Transpilation passes for IQM-targeted circuit compilation.

Provides a composable :class:`~qrisp.PassManager`-based transpilation pipeline with:

**Default pipeline**

* :func:`create_iqm_pass_manager` — factory for the complete plasma-sabre pipeline.
* :func:`transpile_to_iqm` — one-liner convenience wrapper.

**Individual passes**

* Layout: :func:`.plasma_layout`, :func:`vf2pp_layout`
* Routing: :func:`.plasma_route`
* Optimisation: :func:`commute_phases`, :func:`measurement_parallelization`
"""

from iqm.qrisp_iqm.passes.commute_phases import commute_phases
from iqm.qrisp_iqm.passes.default_pass_manager import (
    create_iqm_pass_manager,
    transpile_to_iqm,
)
from iqm.qrisp_iqm.passes.measurement_parallelization import measurement_parallelization

# Routing and placement passes
from iqm.qrisp_iqm.passes.routing import (
    plasma_layout,
    plasma_route,
    vf2pp_layout,
)

__all__ = [
    "commute_phases",
    "plasma_route",
    "vf2pp_layout",
    "plasma_layout",
    "measurement_parallelization",
    "create_iqm_pass_manager",
    "transpile_to_iqm",
]
