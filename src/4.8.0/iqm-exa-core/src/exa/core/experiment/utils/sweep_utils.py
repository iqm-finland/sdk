#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Utility functions for handling and defining sweeps in Experiment."""

from collections.abc import Callable

import numpy as np
from typing_extensions import deprecated

from exa.common.control import SweepOptions
from exa.common.data import Parameter, SettingNode, Sweep
from exa.common.helpers import format_deprecated
from exa.common.sweep import NdSweep, Sweeps

# BW COMPATIBILITY IMPORTS
from iqm.cpc.compiler._utils.stages import (  # noqa: F401
    CIRCUIT_SWEEP_PARAMETER,
    COUNTER_INDEX,
    RAGGED_DATA_INDEX,
    READOUT_SLOT_SWEEP_PARAMETER,
    REPETITIONS,
    TIME_TRACE_INDEX,
)

from .setting_utils import (
    filter_components_with_drive,
)

ListSweepOptions = list[None | (SweepOptions | np.ndarray) | tuple[SweepOptions | np.ndarray]]
DictSweepOptions = dict[str | Parameter, SweepOptions | np.ndarray]
DefaultSweepOptions = ListSweepOptions | DictSweepOptions


@deprecated(format_deprecated(old="get_sweep_parameters", new=None, since="2026-05-13"))
def get_sweep_parameters(sweeps: Sweeps, only_first_in_tuple: bool = False) -> list[Parameter]:
    """Flattens a tuple of sweeps and returns the swept parameters.

    Args:
        sweeps: a list of tuples of sweeps and sweeps to be flattened.
        only_first_in_tuple: If True, only first sweep in tuple of sweeps will be included.

    Returns:
        A flattened list of the parameters in the given sweeps.

    """
    # TODO unused, should be removed
    parameters_flattened = []
    for sweep_or_tuple in sweeps:
        if isinstance(sweep_or_tuple, tuple):
            for sweep in sweep_or_tuple:
                parameters_flattened.append(sweep.parameter)
                if only_first_in_tuple:
                    break
        else:
            parameters_flattened.append(sweep_or_tuple.parameter)
    return parameters_flattened


def _get_sweep_with_new_options(sweep: Sweep, options: SweepOptions | np.ndarray | None) -> Sweep:
    """Return a sweep with modified options."""
    if options is None:
        return sweep
    if isinstance(options, np.ndarray):
        return Sweep(parameter=sweep.parameter, data=options.tolist())
    if isinstance(options, list):
        return Sweep(parameter=sweep.parameter, data=options)
    return Sweep(parameter=sweep.parameter, data=options.data)


def _update_options_with_dict(default_sweeps: NdSweep, default_options: DefaultSweepOptions) -> NdSweep:
    """Update sweep options with a dict of new options."""
    sweeps = []
    for sweep_or_tuple in default_sweeps:
        if isinstance(sweep_or_tuple, tuple):
            sweeps_in_tuple = []
            for sweep in sweep_or_tuple:
                if sweep.parameter in default_options:
                    sweeps_in_tuple.append(_get_sweep_with_new_options(sweep, default_options[sweep.parameter]))  # type: ignore[call-overload]
                elif sweep.parameter.name in default_options:
                    sweeps_in_tuple.append(_get_sweep_with_new_options(sweep, default_options[sweep.parameter.name]))  # type: ignore[call-overload]
                else:
                    sweeps_in_tuple.append(sweep)
            sweeps.append(tuple(sweeps_in_tuple))
        elif sweep_or_tuple.parameter in default_options:
            sweeps.append(_get_sweep_with_new_options(sweep_or_tuple, default_options[sweep_or_tuple.parameter]))
        elif sweep_or_tuple.parameter.name in default_options:
            sweeps.append(_get_sweep_with_new_options(sweep_or_tuple, default_options[sweep_or_tuple.parameter.name]))
        else:
            sweeps.append(sweep_or_tuple)
    return sweeps


def update_options_for_default_sweeps(default_sweeps: NdSweep, default_options: DefaultSweepOptions) -> NdSweep:
    """Change the default sweep options but otherwise keeping the sweep definitions.

    Args:
        default_sweeps: the default sweeps.
        default_options: the new default options to be applied to the sweeps. The new options can be given either
            as list of the same length as the default sweeps and containing the same tuple structure or as a dict
            where the keys are the sweep parameters or the sweep parameter names, and the values of the new options.
            If a dict is given, only the sweeps or parts of parallel sweeps found in the keys, will be altered.

    Returns:
        The default sweeps with the modified options.

    Raises:
        ValueError: if the default_sweeps and default_options lists are of different lengths or a tuple of
            options is to be applied to a non-tuple sweep.

    """
    if isinstance(default_options, dict):
        return _update_options_with_dict(default_sweeps, default_options)

    if len(default_sweeps) != len(default_options):
        raise ValueError(
            f"Default sweeps list of length {len(default_sweeps)} cannot be updated with a default options list "
            f"of length {len(default_options)}"
        )
    sweeps = []
    for sweep_or_tuple, options_or_tuple in zip(default_sweeps, default_options):
        if isinstance(sweep_or_tuple, tuple):
            sweeps_in_tuple = []
            if isinstance(options_or_tuple, tuple):
                for sweep, options in zip(sweep_or_tuple, options_or_tuple):
                    sweeps_in_tuple.append(_get_sweep_with_new_options(sweep, options))
            else:
                sweeps_in_tuple.extend(
                    [_get_sweep_with_new_options(sweep, options_or_tuple) for sweep in sweep_or_tuple]
                )
            sweeps.append(tuple(sweeps_in_tuple))
        else:
            if isinstance(options_or_tuple, tuple):
                raise ValueError(
                    f"Sweep that is not tuple {sweep_or_tuple} cannot have a tuple valued options {options_or_tuple}"
                )
            sweeps.append(_get_sweep_with_new_options(sweep_or_tuple, options_or_tuple))
    return sweeps


@deprecated(format_deprecated(old="get_1qb_default_sweeps", new=None, since="2026-05-13"))
def get_1qb_default_sweeps(
    components: list[str],
    settings: SettingNode,
    get_single_element_sweeps: Callable[..., list[Sweep | tuple[Sweep, ...]]],
    **kwargs,
) -> NdSweep:
    """Define default sweeps over qubits. Similar to get_2qb_default_sweeps from parallel_2QB_util.py.

    Args:
        chad: The chip architecture definition (CHAD).
        components: Components with drive.
        settings: The settings tree.
        get_single_element_sweeps: The function for getting single-element sweeps. The input arguments of this function
            should be ``qubit: str, settings: SettingNode``, and additional arguments (=kwargs).
        kwargs: Additional arguments to ``get_single_element_sweeps``.

    Returns:
        The default sweeps over single elements.

    """
    sweeps: NdSweep = []
    components = filter_components_with_drive(components, settings)

    for i, component in enumerate(components):
        new_sweeps = get_single_element_sweeps(component, settings, **kwargs)
        new_sweep_tuples: NdSweep = []
        for new_sweep_or_tuple in new_sweeps:
            if isinstance(new_sweep_or_tuple, tuple):
                new_sweep_tuples.append(new_sweep_or_tuple)
            elif isinstance(new_sweep_or_tuple, Sweep):
                new_sweep_tuples.append((new_sweep_or_tuple,))
        if i == 0:
            for new_sweep_tuple in new_sweep_tuples:
                sweeps.append(new_sweep_tuple)
        else:
            for j, new_sweep_tuple in enumerate(new_sweep_tuples):
                if new_sweep_tuple[0].parameter != sweeps[j][0].parameter:
                    sweeps[j] = sweeps[j] + new_sweep_tuple
    return sweeps
