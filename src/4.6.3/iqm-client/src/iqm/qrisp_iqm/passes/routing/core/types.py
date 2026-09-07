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

"""Dynamic index-type selection for plasma-sabre routing passes.

Provides a single source of truth for choosing the narrowest numpy integer
dtype that can safely represent a given index range.  All data structures
(coupling topology, instruction DAGs, layout permutations, depth trackers,
etc.) should derive their dtype from the functions in this module rather
than hard-coding ``np.int32``.

Rationale
---------
Plasma-sabre targets NISQ devices (< 32 768 qubits).  Quantum circuits are
expected to contain fewer than 32 768 two-qubit gates before decoherence
destroys any meaningful entanglement.  Under these assumptions **all** index
values fit into ``np.int16``, which doubles cache-line density compared to
``np.int32`` and substantially improves ALU throughput in the hot SABRE
metric loop.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Conservative threshold: use int16 when the value range fits, otherwise
# fall back to int32.  int64 is deliberately excluded from the hot path.
# ---------------------------------------------------------------------------
_INT16_MAX = np.iinfo(np.int16).max  # 32767


def index_dtype(max_value: int) -> np.dtype:
    """Return the narrowest numpy signed-integer dtype for the given range.

    Can represent every value from ``0`` to ``max_value`` (inclusive).

    Parameters
    ----------
    max_value : int
        Largest expected index value.

    Returns
    -------
    np.dtype
        ``np.int16`` when ``max_value <= 32767``, otherwise ``np.int32``.

    """
    if max_value <= _INT16_MAX:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def qubit_dtype(n_physical: int) -> np.dtype:
    """Return dtype suitable for per-qubit data structures.

    Convenience wrapper for c2a, a2c, phys_depth, etc.
    """
    return index_dtype(n_physical - 1)


def node_dtype(n_nodes: int) -> np.dtype:
    """Return dtype suitable for per-DAG-node data structures.

    Convenience wrapper for in_degree, reachability_counts, depth_array, etc.
    """
    return index_dtype(n_nodes - 1)


def data_dtype(n_physical: int, n_nodes: int) -> np.dtype:
    """Return dtype for ``instruction_data`` (Mx3) arrays.

    Stores both qubit indices (0..N-1) and instruction indices (0..M-1).
    """
    return index_dtype(max(n_physical - 1, n_nodes - 1))
