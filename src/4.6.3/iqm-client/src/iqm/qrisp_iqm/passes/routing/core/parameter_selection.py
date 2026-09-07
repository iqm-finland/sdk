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

"""Parameter selection for the SABRE layout-and-route transpiler.

Maps a single ``effort`` integer to the internal tuning parameters
(init_layout_attempts, C, layout_iterations) that control how much
classical compute is invested in finding a good mapping.

Design rationale
----------------
The transpiler has three independent knobs:

* **init_layout_attempts (ILA)** — how many random starting layouts to
  evaluate.  Up to ``cpu_count`` layouts are evaluated in parallel;
  each additional batch of ``cpu_count`` adds one parallel-wave.
* **layout_iterations** — how many back-and-forth refinement passes each
  layout candidate receives.
* **C** — routing diversity multiplier.  ``C * cpu_count`` independent
  routing runs are performed for the best layout.

The relative importance of ILA vs C depends on the number of 2-qubit
gates in the circuit:

* **Shallow circuits** (few 2-qubit gates): layout choice dominates —
  invest more in ILA.
* **Deep circuits** (many 2-qubit gates): routing stochasticity
  dominates — invest more in C.

The crossover follows the empirically fitted relationship
``ILA_importance% = -11.14 * log2(num_2qb_gates) + 114.5`` (R² = 0.69).

Layout iterations saturate at 3 for 14/16 tested regimes.

Fitting methodology
-------------------
The formula was calibrated on an 11,520-point parameter sweep covering
grid sizes 3×3 to 11×11 and circuits with 30 to 1000 random 2-qubit
gates.  Per-regime response surfaces ``depth ~ 1/(log ILA + 1) +
1/(log C + 1)`` achieved R² = 0.72–0.90.

Validation (Formula C in ``fit_parameters_v3.py``):

* Median efficiency vs model-optimal = 100%
* Mean efficiency = 77%
* Worst-case absolute depth penalty < 0.7 gates (< 0.5%)
"""

from __future__ import annotations

import math

import psutil

cpu_count: int = psutil.cpu_count() or 1


def compute_parameters(
    effort: int,
    num_2qb_gates: int,
    cpu_count: int = cpu_count,
) -> dict[str, int]:
    """Derive transpiler parameters from a single effort level.

    Parameters
    ----------
    effort : int
        Classical compute investment level (≥ 1).
        - effort=1 : fast — one parallel wave, C=1-2
        - effort=3 : moderate — 1-2 layout waves, C=2-6
        - effort=5 : thorough — several layout waves, C=4-10
        Wall-clock time scales roughly linearly with effort.
    cpu_count : int, optional
        Number of available CPU cores (auto-detected by default).
    num_2qb_gates : int, optional
        Number of 2-qubit gates in the circuit.  Controls the
        budget split between layout attempts (ILA) and routing
        diversity (C).  Shallow circuits (< ~100 gates) invest
        more in ILA; deep circuits invest more in C.  When *None*,
        a balanced 50/50 split is used.

    Returns
    -------
    dict
        Keys: ``init_layout_attempts``, ``C``, ``layout_iterations``.

    Examples
    --------
    >>> params = compute_parameters(1, cpu_count=16)
    >>> params['init_layout_attempts'] >= 16
    True
    >>> params['C'] >= 1
    True
    >>> params['layout_iterations']
    3

    >>> # Shallow circuit → heavy ILA, light C
    >>> p = compute_parameters(5, cpu_count=16, num_2qb_gates=30)
    >>> p['init_layout_attempts'] >= 48
    True
    >>> p['C'] <= 5
    True

    >>> # Deep circuit → light ILA, heavy C
    >>> p = compute_parameters(5, cpu_count=16, num_2qb_gates=1000)
    >>> p['init_layout_attempts'] <= 32
    True
    >>> p['C'] >= 8
    True

    """
    effort = max(1, effort)

    # Layout iterations: saturates at 3 (validated — 14/16 regimes)
    layout_iterations = 3

    # ──────────────────────────────────────────────────────────────────
    # Budget split: ILA vs C
    #
    # Fitted from 11,520-point sweep:
    #   ILA_importance% = -11.14 * log2(num_2qb_gates) + 114.5
    #   R² = 0.69
    # ──────────────────────────────────────────────────────────────────
    if num_2qb_gates is not None and num_2qb_gates > 0:
        ila_frac = (-11.14 * math.log2(max(num_2qb_gates, 4)) + 114.5) / 100.0
        ila_frac = max(0.05, min(0.80, ila_frac))
    else:
        ila_frac = 0.50  # balanced when circuit size unknown

    # Floor at 0.20 so neither axis is fully neglected
    ila_weight = max(ila_frac, 0.20)
    c_weight = max(1.0 - ila_frac, 0.20)

    init_layout_attempts = max(cpu_count, round(effort * cpu_count * ila_weight))
    C = max(1, round(effort * c_weight * 2))

    return {
        "init_layout_attempts": init_layout_attempts,
        "C": C,
        "layout_iterations": layout_iterations,
    }
