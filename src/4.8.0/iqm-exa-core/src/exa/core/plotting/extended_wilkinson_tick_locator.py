# Copyright 2019-2026 IQM Finland Oy
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
"""Tick locator implementation for Matplotlib based on the extended Wilkinson’s Algorithm.

See http://vis.stanford.edu/papers/tick-labels
"""

import math

from matplotlib.axis import XAxis
from matplotlib.ticker import Locator
import numpy as np


def _coverage(dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    return 1.0 - 0.5 * ((dmax - lmax) ** 2 + (dmin - lmin) ** 2) / (0.1 * (dmax - dmin)) ** 2


def _coverage_max(dmin: float, dmax: float, span: float) -> float:
    drange = dmax - dmin
    if span > drange:
        return 1.0 - (0.5 * (span - drange)) ** 2 / (0.1 * drange) ** 2
    return 1.0


def _density(k: float, m: float, dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    r = (k - 1.0) / (lmax - lmin)
    rt = (m - 1.0) / (max(lmax, dmax) - min(lmin, dmin))
    return 2.0 - max(r / rt, rt / r)


def _density_max(k: float, m: float) -> float:
    if k >= m:
        return 2.0 - (k - 1.0) / (m - 1.0)
    return 1.0


def _simplicity(q: float, Q: list[float], j: float, lmin: float, lmax: float, lstep: float) -> float:
    eps = 1e-10
    n = len(Q)
    i = Q.index(q) + 1
    v = 1 if lmin % lstep < eps or (lstep - lmin) % lstep < eps and lmin <= 0 <= lmax else 0

    return (n - i) / (n - 1.0) + v - j


def _simplicity_max(q: float, Q: list[float], j: float) -> float:
    n = len(Q)
    i = Q.index(q) + 1
    v = 1
    return (n - i) / (n - 1.0) + v - j


def _legibility() -> float:
    return 1.0


def _score(weights: list[float], simplicity: float, coverage: float, density: float, legibility: float):  # noqa: ANN202
    return weights[0] * simplicity + weights[1] * coverage + weights[2] * density + weights[3] * legibility


def wilk_ext(  # noqa: ANN201, D103
    dmin: float,
    dmax: float,
    m: float,
    only_inside: float = 0,
    Q: list[float] | None = None,
    w: list[float] | None = None,
):
    if w is None:
        w = [0.2, 0.25, 0.5, 0.05]
    if Q is None:
        Q = [1, 5, 2, 2.5, 4, 3]
    if (dmin >= dmax) or (m < 1):
        return dmin, dmax, dmax - dmin, 1, 0, 2, 0

    best_score = -1.0
    result = None

    j = 1.0
    while j < np.inf:
        for q in map(float, Q):
            sm = _simplicity_max(q, Q, j)

            if _score(w, sm, 1, 1, 1) < best_score:
                j = np.inf
                break

            k = 2.0
            while k < np.inf:
                dm = _density_max(k, m)

                if _score(w, sm, 1, dm, 1) < best_score:
                    break

                delta = (dmax - dmin) / (k + 1.0) / j / q
                z = math.ceil(math.log10(delta))

                while z < np.inf:
                    step = j * q * 10.0**z
                    cm = _coverage_max(dmin, dmax, step * (k - 1.0))

                    if _score(w, sm, cm, dm, 1) < best_score:
                        break

                    min_start = math.floor(dmax / step) * j - (k - 1.0) * j
                    max_start = math.ceil(dmin / step) * j

                    if min_start > max_start:
                        z += 1
                        break

                    for start in np.arange(min_start, max_start + 1):
                        lmin = float(start) * (step / j)
                        lmax = lmin + step * (k - 1.0)
                        lstep = step

                        s = _simplicity(q, Q, j, lmin, lmax, lstep)
                        c = _coverage(dmin, dmax, lmin, lmax)
                        d = _density(k, m, dmin, dmax, lmin, lmax)
                        l = _legibility()  # noqa: E741
                        scr = _score(w, s, c, d, l)

                        if (
                            scr > best_score
                            and (only_inside <= 0 or (lmin >= dmin and lmax <= dmax))
                            and (only_inside >= 0 or (lmin <= dmin and lmax >= dmax))
                        ):
                            best_score = scr
                            result = (lmin, lmax, lstep, j, q, k, scr)

                    z += 1
                # end of z-while-loop
                k += 1
            # end of k-while-loop
        j += 1
    # end of j-while-loop
    return result


def get_ticks(dmin: float, dmax: float, m: float, **kwargs) -> np.ndarray:  # noqa: D103
    lmin, lmax, lstep, _, _, _, _ = wilk_ext(dmin, dmax, m, **kwargs)
    return np.arange(lmin, lmax + lstep, lstep)


class ExtendedWilkinsonTickLocator(Locator):
    """Places the ticks according to the extended Wilkinson algorithm.

    For details see http://vis.stanford.edu/files/2010-TickLabels-InfoVis.pdf

    Args:
        target_density:
            controls the number of ticks. The algorithm will try
            to put as many ticks on the axis but deviations are
            allowed if another criterion is considered more important.
        only_inside:
            controls if the first and last label include the data range.

            - 0  : doesn't matter
            - >0 : label range must include data range
            - <0 : data range must be larger than label range
        per_inch:
            if per_inch=True then it specifies the number of ticks per
            inch instead of a fixed number, so the actual number of
            ticks depends on the size of the axis

        Q:
            numbers that are considered as 'nice'. Ticks will be
            multiples of these

        w:
            list of weights that control the importance of the four
            criteria: simplicity, coverage, density and legibility

    """

    def __init__(
        self,
        target_density: float,
        only_inside: int = 0,
        per_inch: bool = False,
        Q: list[float] | None = None,
        w: list[float] | None = None,
    ):
        if w is None:
            w = [0.2, 0.25, 0.5, 0.05]
        if Q is None:
            Q = [1, 5, 2, 2.5, 4, 3]
        self.target_density = target_density
        self.only_inside = only_inside
        self.Q = Q
        self.w = w
        self.per_inch = per_inch

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # determine "physical" size of the axis and according # of ticks
        if self.per_inch:
            i = 0 if isinstance(self.axis, XAxis) else 1
            axes = self.axis.axes
            pos = axes.get_position()
            size = axes.get_figure().get_size_inches()
            inches = size[i] * pos.bounds[i + 2]
            nticks = self.target_density * inches
        else:
            nticks = self.target_density

        return get_ticks(vmin, vmax, nticks, only_inside=self.only_inside, Q=self.Q, w=self.w)

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        return get_ticks(vmin, vmax, self.target_density, only_inside=self.only_inside, Q=self.Q, w=self.w).tolist()
