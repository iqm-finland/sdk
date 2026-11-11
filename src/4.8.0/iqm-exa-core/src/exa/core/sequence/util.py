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
"""Utils for sequences."""

from typing import Any, Protocol

from exa.core.experiment import Experiment

from .experiment_tracker import ExperimentTracker


class _Sequence(Protocol):
    """A Sequence.

    This is just a dummy class to overcome circular imports.
    The actual class is :class:`exa.sequence.core.sequence.Sequence`.
    """

    options: dict[str, Any] = {}


def update_options(node: Experiment | ExperimentTracker | _Sequence, options: dict[str, Any]) -> None:
    """Updates the options of an experiment or a sequence from the sequence-level options.

    The relevant sequence level options are chosen and their values applied to the node's options.
    For the usage of this function, the sequence options should be named
    ``f"{experiment_class_name}_{experiment_option_name}"``.
    For example, to set the ``repetitions`` of :class:`.CPhaseDiag`
    one should have the sequence level option named as ``CPhaseDiag_repetitions``.

    The convenience prefix ``GLOBAL`` can be used in place of an experiment class name to set the option for any
    Experiment or ExperimentTracker regardless or class name. In case node is a Sequence the option is set for all
    contained experiments. Class name options take precedence over ``GLOBAL`` options if both are specified for the same
    option.

    Options can also be preceded with ``RECURSIVE``, which will cause the option to be recursively applied
    also to all subsequences in case node is a Sequence. For example, ``RECURSIVE_CPhaseDiag_repetitions`` or
    ``RECURSIVE_GLOBAL_repetitions``. Omitting ``RECURSIVE`` sets the option only on the top level sequence.
    The ``RECURSIVE`` prefix has no effect on Experiment or ExperimentTracker node objects.
    Non-recursive options take precedence over recursive ones if both are specified for the same option.
    The order of prefixes has no effect on execution.


    Args:
        node: The experiment instance or an experiment tracker or a subsequence.
        options: The sequence-level options dict.


    Raises:
        ValueError: if both ``GLOBAL`` and node name prefix name are specified; e.g ``GLOBAL_CPhaseDiag_repetitions``

    """

    def _parse_options():  # noqa: ANN202
        recursive_global_options = {}
        recursive_node_options = {}
        local_global_options = {}
        local_node_options = {}

        for key, value in options.items():
            if node_prefix in key and global_prefix in key:
                raise ValueError(f"Option can not be marked both {global_prefix} and {node_prefix}")
            if recursion_prefix in key:
                if global_prefix in key and not subsequence:
                    recursive_global_options[key.replace(recursion_prefix, "").replace(global_prefix, "")] = value
                elif node_prefix in key:
                    recursive_node_options[key.replace(recursion_prefix, "").replace(node_prefix, "")] = value
            elif global_prefix in key and not subsequence:
                local_global_options[key.replace(global_prefix, "")] = value
            elif node_prefix in key:
                local_node_options[key.replace(node_prefix, "")] = value

        # The merge order is important here as it guarantees the precedence from right to left.
        parsed_options = recursive_global_options | recursive_node_options | local_global_options | local_node_options
        if subsequence:
            parsed_options = parsed_options | {key: value for key, value in options.items() if recursion_prefix in key}
        return parsed_options

    global_prefix = "GLOBAL_"
    recursion_prefix = "RECURSIVE_"

    if isinstance(node, ExperimentTracker):
        node_prefix = f"{node.experiment_class_name}_"
        subsequence = False
    elif isinstance(node, Experiment):
        node_prefix = f"{node.__class__.__name__}_"
        subsequence = False
    else:
        node_prefix = f"{node.__class__.__name__}_"
        subsequence = True

    relevant_options = _parse_options()

    for option_name, option_value in relevant_options.items():
        if isinstance(node, (Experiment, ExperimentTracker)):
            setattr(node.options, option_name, option_value)
        else:
            node.options[option_name] = option_value
