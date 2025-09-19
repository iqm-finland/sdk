#  ********************************************************************************
#
# Copyright 2024 IQM
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
"""Utility functions."""

from __future__ import annotations

from typing import get_args, get_origin

import numpy as np

from exa.common.data.parameter import CollectionType, DataType


def map_waveform_param_types(type_hint: type) -> tuple[DataType, CollectionType]:
    """Map a python typehint into EXA Parameter's `(DataType, CollectionType)` tuple.

    Args:
        type: python typehint.

    Returns:
        A `(DataType, CollectionType)` tuple

    Raises:
        ValueError: for a non-supported type.

    """
    value_error = ValueError(f"Nonsupported datatype for a waveform parameter: {type_hint}")
    if hasattr(type_hint, "__iter__") and type_hint is not str:
        if type_hint == np.ndarray:
            data_type = DataType.COMPLEX  # due to np.ndarray not being generic we assume complex numbers
            collection_type = CollectionType.NDARRAY
            return (data_type, collection_type)
        if get_origin(type_hint) is list:
            collection_type = CollectionType.LIST
            type_hint = get_args(type_hint)[0]
        else:
            raise value_error
    else:
        collection_type = CollectionType.SCALAR

    if type_hint is float:
        data_type = DataType.FLOAT
    elif type_hint is int:
        data_type = DataType.INT
    elif type_hint is str:
        data_type = DataType.STRING
    elif type_hint is complex:
        data_type = DataType.COMPLEX
    elif type_hint is bool:
        data_type = DataType.BOOLEAN
    else:
        raise value_error
    return (data_type, collection_type)


def normalize_angle(angle: float) -> float:
    """Normalize the given angle to (-pi, pi].

    Args:
        angle: angle to normalize (in radians)

    Returns:
        ``angle`` normalized to (-pi, pi]

    """
    half_turn = np.pi
    full_turn = 2 * half_turn
    return (angle - half_turn) % -full_turn + half_turn


def phase_transformation(psi_1: float = 0.0, psi_2: float = 0.0) -> tuple[float, float]:
    r"""Implement an arbitrary (RZ, PRX, RZ) gate sequence by modifying the parameters of the
    IQ pulse implementing the PRX.

    By commutation rules we have

    .. math::
       RZ(\psi_2) \: PRX(\theta, \phi) \: RZ(\psi_1) = PRX(\theta, \phi+\psi_2) \: RZ(\psi_1 + \psi_2).

    Hence an arbitrary (RZ, PRX, RZ) gate sequence is equivalent to (RZ, PRX) with adjusted angles.

    Use case: with resonant driving, the PRX gate can be implemented using an :class:`IQPulse` instance,
    and the preceding RZ can be handled by decrementing the local oscillator phase beforehand (something
    the IQPulse instruction can also do), which is equivalent to rotating the local computational frame
    around the z axis in the opposite direction of the required quantum state rotation.

    Args:
        psi_1: RZ angle before the PRX (in rad)
        psi_2: RZ angle after the PRX (in rad)

    Returns:
        change to the PRX phase angle (in rad),
        phase increment for the IQ pulse that implements the remaining RZ (in rad)

    """
    return psi_2, -(psi_1 + psi_2)
