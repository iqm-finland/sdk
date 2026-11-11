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
"""Core utilities for Experiment."""

from copy import copy

from exa.common.data import CollectionType, DataType, Parameter, Setting, SettingNode, Sweep
from exa.common.qcm_data import ChipTopology
from exa.common.sweep import NdSweep
from exa.core.experiment_definitions import ExperimentOptions, ReadoutMode


def merge_sweeps(user_sweeps: NdSweep, default_sweeps: NdSweep) -> NdSweep:
    """Merges default sweeps with user sweeps.

    User sweeps take precedence.
    They can introduce sweeps over entirely new Parameters, and also replace individual ``Sweep``
    instances within a default ``ParallelSweep``. For the latter use case the replacing
    sweeps must be given...

    .. note::

       This function is not a very good method for overriding default sweeps.
       :meth:`.Experiment.create` parameter ``default_sweep_ranges`` should be preferred for this,
       and the ``sweeps`` parameter should be be used only for performing additional sweeps.

    Args:
        user_sweeps: User-given sweeps. Sweeps over parameters not found in ``default_sweeps``
            will be at the end of the returned ``NdSweep``.
        default_sweeps: Default sweeps for an Experiment.

    Returns:
        Merged sweeps.

    Raises:
        ValueError: ``user_sweeps`` is invalid.

    """
    user_sweep_parameters = {
        user_sweep.parameter.name: user_sweep
        for user_parallel_sweep in user_sweeps
        for user_sweep in user_parallel_sweep
    }
    default_sweep_parameters = set()
    ret = []
    # replace default sweeps
    for default_parallel_sweep in default_sweeps:
        new_parallel_sweep = []
        for default_sweep in default_parallel_sweep:
            name = default_sweep.parameter.name
            default_sweep_parameters.add(name)
            # User wants to override this sweep
            new_sweep = user_sweep if (user_sweep := user_sweep_parameters.get(name)) is not None else default_sweep
            # Make a copy since Sweep is not immutable
            new_parallel_sweep.append(Sweep(parameter=new_sweep.parameter, data=new_sweep.data))
        ret.append(tuple(new_parallel_sweep))

    # add new sweeps
    for user_parallel_sweep in user_sweeps:
        is_old = [user_sweep.parameter.name in default_sweep_parameters for user_sweep in user_parallel_sweep]
        if all(is_old):
            continue
        elif any(is_old):
            raise ValueError("Cannot mix new and default sweeps in a single ParallelSweep.")
        # all-new sweep, append a copy
        ret.append(
            tuple(Sweep(parameter=new_sweep.parameter, data=new_sweep.data) for new_sweep in user_parallel_sweep)
        )
    return ret


def get_sweep_parameters(sweeps: NdSweep, only_first_in_tuple: bool = False) -> list[Parameter]:
    """Flattens a tuple of sweeps and returns the swept parameters.

    Args:
        sweeps: a list of tuples of sweeps and sweeps to be flattened.
        only_first_in_tuple: If True, only first sweep in tuple of sweeps will be included.

    Returns:
        A flattened list of the parameters in the given sweeps.

    """
    parameters_flattened = []
    for parallel_sweep in sweeps:
        for sweep in parallel_sweep:
            parameters_flattened.append(sweep.parameter)
            if only_first_in_tuple:
                break
    return parameters_flattened


def finalize_settings(
    settings: SettingNode,
    options: ExperimentOptions,
    chip_topology: ChipTopology,
    subscribed_components: list[str] | dict[str, list[str]],
) -> None:
    """Finalize settings for a run based on :class:``.ExperimentOptions``.

    Sets the additionally measured (subscribed or just probed) components.

    Args:
        settings: The settings tree.
        options: The options for the run.
        chip_topology: The chip topology.
        subscribed_components: The subscribed components.

    """
    if options.readout_mode == ReadoutMode.COUNTER.value and not settings.stages.subscribed_groups.settings:
        if isinstance(subscribed_components, dict):
            for group, group_components in subscribed_components.items():
                settings.stages.subscribed_groups[group] = Setting(
                    Parameter(group, data_type=DataType.STRING, collection_type=CollectionType.LIST),
                    copy(group_components),  # type: ignore[arg-type]
                )
        else:
            settings.stages.subscribed_groups["all_components"] = Setting(
                Parameter("all_components", data_type=DataType.STRING, collection_type=CollectionType.LIST),
                copy(subscribed_components),  # type: ignore[arg-type]
            )
    elif (
        options.readout_mode != ReadoutMode.COUNTER.value
        and not settings.stages.additionally_subscribed_components.value
    ):
        settings.stages.additionally_subscribed_components = copy(subscribed_components)
    if not settings.stages.additionally_probed_components.value:  # get all probeable components
        probed_components = []
        terminal_measure = options.used_measure_operations["terminal"]
        for component in chip_topology.qubits_sorted + chip_topology.couplers_sorted:
            default_measure = settings.get_default_implementation_name(terminal_measure, component)
            if (
                component in settings.gates[terminal_measure][default_measure].children
                and component in chip_topology.component_to_probe_line
                and (
                    options.exclude_operations_for is None
                    or component not in options.exclude_operations_for.get("readout", {})
                )
            ):
                probed_components.append(component)
        settings.stages.additionally_probed_components = probed_components
