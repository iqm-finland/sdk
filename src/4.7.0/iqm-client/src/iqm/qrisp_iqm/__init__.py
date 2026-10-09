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
"""Qrisp adapter for IQM's quantum computers.

Provides the full Qrisp → IQM toolchain:

**Backend & execution**

* :class:`~iqm.qrisp_iqm.backends.backend.IQMBackend` — run circuits on IQM hardware.
* :class:`.IQMCircuitJob`, :class:`.IQMPulseJob` — track asynchronous execution.

**Transpilation**

* :func:`.transpile_to_iqm` — one-liner transpilation to native IQM gates.
* :func:`.create_iqm_pass_manager` — factory for the full plasma-sabre pipeline.
* :func:`.plasma_layout`, :func:`.plasma_route`, :func:`.vf2pp_layout` — layout & routing passes.
* :func:`~iqm.qrisp_iqm.passes.commute_phases.commute_phases`, :func:`.measurement_parallelization`
  — optimisation passes.

**Pulse-level integration**

* :func:`.extract_iqm_pulse` — Jasp-trace and compile to pulse-level circuits.
* :class:`.IQMPulseOperation` — embed native IQM QuantumOps in Qrisp circuits.
* :func:`.quantum_op_to_qrisp_func` — turn an IQM QuantumOp into a Qrisp gate function.

**Circuit conversion**

* :func:`.qrisp_to_iqm_converter` — convert transpiled Qrisp circuits to IQM format.

**QEC**

* :class:`.DetectorExperiment` — decorator for QEC experiments with Stim + PyMatching.
"""

from __future__ import annotations

from .backends import IQMBackend, IQMCircuitJob, IQMPulseJob

# Re-export custom pulse operations
from .custom_pulse_operations import delay
from .extract_iqm_pulse_decorator import extract_iqm_pulse
from .iqm_converter import (
    qrisp_to_iqm_converter,
)
from .passes import (
    commute_phases,
    create_iqm_pass_manager,
    measurement_parallelization,
    transpile_to_iqm,
)
from .passes.routing import plasma_layout, plasma_route, vf2pp_layout
from .pulse_operation import IQMPulseOperation, quantum_op_to_qrisp_func
from .qec import DetectorExperiment

__all__ = [
    # Backend
    "IQMBackend",
    "IQMCircuitJob",
    "IQMPulseJob",
    # Circuit conversion
    "qrisp_to_iqm_converter",
    # Pulse operations
    "IQMPulseOperation",
    "quantum_op_to_qrisp_func",
    "extract_iqm_pulse",
    # Transpilation — default pipeline
    "create_iqm_pass_manager",
    "transpile_to_iqm",
    # Transpilation — layout & routing
    "plasma_layout",
    "plasma_route",
    "vf2pp_layout",
    # Transpilation — optimization
    "commute_phases",
    "measurement_parallelization",
    # QEC
    "DetectorExperiment",
    # Custom operations
    "delay",
]
