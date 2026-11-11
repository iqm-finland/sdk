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
"""Options for controlling experiment behavior."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ExperimentOptions(BaseModel):
    """Options for controlling experiment behavior.

    Options change the behavior of the experiment execution in a way that is specific to the experiment type.
    Unlike instrument settings, these are not propagated to the instruments.
    """

    readout_mode: str = "complex"
    """Defines what type of final readout the experiment uses, see :class:`ReadoutMode` for the
    possible values.
    """
    readout_programming_mode: str = "classic"
    """How the experiment handles final readout, heralding and reset, see :class:`ReadoutProgrammingMode`
    for the possible values.
    """
    active_reset_cycles: int | None = None
    """Number of active ``reset`` operations inserted at the beginning of each segment for each active qubit, in their
    own :class:`TimeBox`. Uses the default implementation of ``reset``. ``None`` means active reset is not used
    but instead reset is done by waiting (relaxation), using the default implementation of the ``reset_wait`` operation.
    Integer values smaller than 1 result in neither active nor reset by wait being used, in which case any reset
    operations must be explicitly added in the circuit generation function.
    """
    used_measure_operations: dict[str, str] = {
        "terminal": "measure_fidelity",
        "heralding": "measure",
        "reset_at_end": "measure",
    }
    """Names of the measurement QuantumOp the various :attr:`readout_programming_mode` s use when building
    the circuits. Changing the defaults (e.g. making ``"heralding"`` use ``"measure_fidelity"``) may result in
    sub-optimal behavior.

    * ``"terminal"`` is used for terminal measurement when :attr:`readout_programming_mode` is
      either ``"classic"`` or ``heralding``.
    * ``"heralding"`` is used for the heralding measurement when :attr:`readout_programming_mode` is ``"heralding"``.
    * ``"reset_at_end"`` is used when :attr:`readout_programming_mode` is ``"reset_at_end"``.

    Note: this attribute applies only to the measure operations added automatically by Experiment.
    If the experiment adds measure operations explicitly in its circuit,
    you must write the logic for choosing the right operation to the circuit generation function itself.
    """
    repetitions: int = 256
    """Averaging used in the experiment. This value will be used for ``controllers.options.playlist_repeats``,
    and possibly for ``controllers.options.averaging_bins``, depending on :attr:`readout_mode`."""
    subscribed_components: list[str] | dict[str, list[str]] | None = None
    """Components that are read out in the experiment. Can be a list of component names, or a dict
    of lists of component names where the keys are the readout group names (this option is only available
    with :attr:`readout_mode` ``== "counter"``)."""
    exclude_operations_for: dict[str, list[str] | str] | None = None
    """Dictionary containing operations that are excluded from the experiment for given components. Operations
    are given as keys and component or components as values e.g. exclude_operations_for = {"readout": [QB1, QB2]}.
    The operations are removed as part of :meth:`get_settings`."""
    data_size_safety_switch: bool = True
    """If ``True`` does not allow runs that would result in excessive amounts of data, possibly straining the DB.
    An error message will be thrown that approximates the data size and asks the user to set the option ``False`` if
    they still want to proceed."""
    force_ragged_data: bool = False
    """If ``False`` the result datasets will be reshaped to an N-dimensional box based on the segment
    (hard sweeps) dimensions, whenever that is possible. The ragged data format, where the result data is
    indexed along the readout trigger index, is used only when necessary, i.e. if the number of readout triggers per
    segment is not constant. Turn this option to ``True`` if you want the experiment to always produce ragged
    data. This can be useful e.g. if the number of readouts per segment is not deterministic or known, and the
    experiment may under some circumstances return ND-data and in some ragged data (forcing the data to be always ragged
    makes the result handling consistent)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def to_dict(self) -> dict[str, Any]:
        """Extract state of the current experiment options as a dictionary.

        Any arbitrary type present is also converted to dict, and a ``__type__`` attribute is added to the root
        of that dict, so that :meth:`.from_dict` is able to convert them back.
        The arbitrary types need to be specified in :func:`_class_name_to_class` and have the method ``to_dict``
        implemented.

        Returns:
            A dict[str, Any] representing the state of the current experiment options.

        Raises:
            ValueError: if an arbitrary type does not have the method "to_dict" implemented.

        """

        def convert_value(obj):  # noqa: ANN001, ANN202
            if obj is None or isinstance(obj, (str, bool, int, float)):
                return obj
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, (tuple, list)):
                return [convert_value(o) for o in obj]
            if isinstance(obj, dict):
                return {convert_value(key): convert_value(value) for key, value in obj.items()}
            to_dict_method = getattr(obj, "to_dict", None)
            if callable(to_dict_method):
                return obj.to_dict() | {"__type__": type(obj).__name__}
            raise ValueError(
                f'To convert a non-primitive type "{type(obj)}" to dict, it has to have'
                ' the method "to_dict" implemented.'
            )

        return {k: convert_value(v) for k, v in self.model_dump().items()}

    @classmethod
    def from_dict(cls: type[ExperimentOptions], values: dict[str, Any]) -> ExperimentOptions:
        """Construct an ExperimentOptions instance from specified state provided as a dictionary.

        Construction can be based on the type of the class, or just this base class if the type is not relevant.
        If :attr:`values` contains ``'__type__'`` in a nested dictionary (as is typically inserted by
        :meth:`.to_dict`), it will be silently removed.
        The intended usage is that the ``'__type__'`` information is used to determine which arbitrary type was
        converted to a dict and now needs to be converted back to the type.

        Args:
            values: Dictionary to create the instance from.

        Returns:
            An instance of :class:`.ExperimentOptions`.

        Raises:
            ValueError: either if an arbitrary type has been used and it does not have a mapping in
                :func:`._class_name_to_class`, or if the arbitrary type does not have the method
                "from_dict" implemented.

        """

        def convert_value(val):  # noqa: ANN001, ANN202
            if val is None or isinstance(val, (str, bool, int, float)):
                return val
            if isinstance(val, (tuple, list)):
                return type(val)(convert_value(v) for v in val)
            if isinstance(val, dict):
                if "__type__" in val:
                    class_name_to_class = _class_name_to_class()
                    class_name = val.pop("__type__")
                    if class_name not in class_name_to_class:
                        raise ValueError(f'No class definition for arbitrary type "{class_name}"')
                    clazz = class_name_to_class[class_name]
                    from_dict_method = getattr(clazz, "from_dict", None)
                    if callable(from_dict_method):
                        return clazz.from_dict(val)
                    raise ValueError(
                        f'To convert a non-primitive type "{class_name}" from dict, it has to have'
                        ' the method "from_dict" implemented.'
                    )
                return {convert_value(key): convert_value(value) for key, value in val.items()}
            raise ValueError(f"Unrecognized type in dict: {type(val)}")

        return cls(**{k: convert_value(v) for k, v in values.items()})

    def __contains__(self, item: str) -> bool:
        """Return ``True`` if ``item`` is a declared option field, ``False`` otherwise."""
        return item in self.__dict__

    def __getitem__(self, item: str) -> Any:
        """Return the value of option field ``item``; raise ``KeyError`` when not declared."""
        if item in self.__dict__:
            return self.__dict__[item]
        raise KeyError(item)


def _class_name_to_class() -> dict[str, Any]:
    """Returns mapping from an arbitrary type name to a class.

    If an arbitrary type is used in a child instance of :class:`ExperimentOptions`, it needs to be mapped here and
    it has to have the methods ``to_dict`` and ``from_dict`` implemented.
    """
    return {}
