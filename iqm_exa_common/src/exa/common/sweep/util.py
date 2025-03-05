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

from exa.common.control.sweep.option import StartStopOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import Parameter

# One sweep where many sweeps are executed together, like a zip. Type hint alias.
ParallelSweep = tuple[Sweep, ...]

# N-dimensional sweep, represented as list of N parallel Sweeps. Type hint alias.
NdSweep = list[ParallelSweep]

# List of sweeps or parallel Sweeps. Type hint alias.
Sweeps = list[Sweep | ParallelSweep]


def convert_sweeps_to_list_of_tuples(sweeps: Sweeps) -> NdSweep:
    """Validate sweeps and convert it to format accepted by the station control.

    Converts a more convenient sweep definition list to a strict list of tuples of sweeps.
    The sweep instances themselves are the same, except single sweep instances are turned
    into a tuple containing a single sweep.

    Verify that:
    * sweeps list element is either Sweep or ParallelSweep
    * tuple_of_sweep element is a Sweep
    * tuple of sweeps contains at least one element
    * length of a data is identical in all tuples of sweeps

    Args:
        sweeps: More user-friendly definition of a list of sweeps.

    Returns:
        List of tuples of sweeps.

    Raises:
        ValueError if sweeps parameter does not follow the contract.

    """
    new_list = []
    for tuple_or_sweep in sweeps:
        if isinstance(tuple_or_sweep, tuple):
            if not len(tuple_or_sweep) > 0:
                raise ValueError("Tuples of sweeps must have at least one element")
            for sweep in tuple_or_sweep:
                if not isinstance(sweep, Sweep):
                    raise ValueError(f"Tuples of sweeps must contain tuple type, got {type(sweep)}")
                if len(sweep.data) != len(tuple_or_sweep[0].data):
                    raise ValueError(
                        f"Length {len(sweep.data)} of a data in {sweep} did not match to expected "
                        + f" length {len(tuple_or_sweep[0].data)}"
                    )
            new_list.append(tuple_or_sweep)
        elif isinstance(tuple_or_sweep, Sweep):
            new_list.append((tuple_or_sweep,))
        else:
            raise ValueError(f"Elements in sweeps must be either tuples of Sweeps or Sweep, got {type(tuple_or_sweep)}")
    return new_list


def linear_index_sweep(parameter: Parameter, length: int) -> list[tuple[Sweep]]:
    """Produce an NdSweep over a dummy index.

    Can be used in places where a "hardware sweep" is needed but not really meaningful.

    Args:
        parameter: Data parameter this index is for.
        length: Number of integers in the dummy sweep.

    Returns:
        A linear sweep over a parameter whose name is ``parameter.name + _index`` and whose data ranges from 0 to
        `length` with steps of 1.

    """
    return [
        (
            Sweep(
                parameter=Parameter(name=parameter.name + "_index", label=parameter.label + " index"),
                data=StartStopOptions(0, length - 1, count=length).data,
            ),
        )
    ]
