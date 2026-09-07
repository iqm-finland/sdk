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

"""Generic utilities for converting sweep definitions from
user-friendly format to canonic ones.
"""

from typing import TypeAlias

from exa.common.control.sweep.option import StartStopOptions
from exa.common.data.parameter import Parameter, Sweep
from exa.common.errors.iqm_error import ValidationError

ParallelSweep: TypeAlias = tuple[Sweep, ...]
"""One or more single-parameter sweeps executed in parallel, like the Python zip."""

NdSweep: TypeAlias = list[ParallelSweep]
"""Cartesian product of N ParallelSweeps."""

Sweeps: TypeAlias = list[Sweep | ParallelSweep]
"""List of single or parallel Sweeps.

Convenience type used only in Experiments, and converted to NdSweep ASAP."""


def convert_sweeps_to_list_of_tuples(sweeps: Sweeps) -> NdSweep:
    """Validate sweeps and convert it to format accepted by the station control.

    Converts a more lax sweep definition to a strict NdSweep.
    The sweeps themselves are the same, except single Sweep instances are turned
    into a ParallelSweep containing a single Sweep.

    Verify that:

    * ``sweeps`` elements are either Sweep or ParallelSweep
    * ParallelSweep elements are Sweep
    * ParallelSweep contains at least one element
    * ParallelSweep component Sweeps all have equal-length data

    Args:
        sweeps: More user-friendly definition of a list of sweeps.

    Returns:
        Converted ``sweeps``.

    Raises:
        ValidationError: ``sweeps`` does not follow the contract.

    """
    new_list = []
    for tuple_or_sweep in sweeps:
        if isinstance(tuple_or_sweep, tuple):
            # ParallelSweep
            if not tuple_or_sweep:
                raise ValidationError("ParallelSweeps must have at least one element")
            for sweep in tuple_or_sweep:
                if not isinstance(sweep, Sweep):
                    raise ValidationError(f"ParallelSweeps must contain the Sweep type, got {type(sweep)}")
                expeced_data_len = len(tuple_or_sweep[0].data)
                data_len = len(sweep.data)
                if data_len != expeced_data_len:
                    raise ValidationError(
                        f"Data length {data_len} of {sweep} did not match expected length {expeced_data_len}"
                    )
            new_list.append(tuple_or_sweep)
        elif isinstance(tuple_or_sweep, Sweep):
            new_list.append((tuple_or_sweep,))
        else:
            raise ValidationError(
                f"Elements in sweeps must be either tuple[Sweep, ...] or Sweep, got {type(tuple_or_sweep)}"
            )
    return new_list


def linear_index_sweep(parameter: Parameter, length: int) -> NdSweep:
    """Produce an NdSweep over a dummy index.

    Can be used in places where a "hardware sweep" is needed but not really meaningful.

    Args:
        parameter: Data parameter this index is for.
        length: Number of integers in the dummy sweep.

    Returns:
        Linear sweep over a parameter whose name is ``parameter.name + _index``
        and whose data ranges from 0 to ``length - 1`` with steps of 1.

    """
    return [
        (
            Sweep(
                parameter=Parameter(name=parameter.name + "_index", label=parameter.label + " index"),
                data=StartStopOptions(0, length - 1, count=length).data,
            ),
        )
    ]
