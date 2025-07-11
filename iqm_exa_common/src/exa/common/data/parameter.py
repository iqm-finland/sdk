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

# mypy: ignore-errors

"""Physical quantities and instrument settings.

A basic data structure in EXA is the :class:`~exa.common.data.parameter.Parameter`, which represents
a single variable. The variable can be a high-level or low-level control knob of an instrument such as
the amplitude of a pulse or a control voltage; a physical quantity such as resonance frequency; or an abstract concept
like the number of averages in a measurement.

The Parameter is a simple structure with a name, label, unit and a datatype without much functionality.
The :class:`.Setting` combines a Parameter and a *value* of the corresponding type.
Like Parameters, Settings are lightweight objects that contain information but don't do anything by themselves.


.. doctest::

    >>> from exa.common.data.parameter import Parameter, Setting
    >>> p = Parameter('qubit_1.drive.pi_pulse.duration', 'Duration', 's')
    >>> s = Setting(p, 1e-6)
    >>> print(s)
    Setting(Duration = 1e-06 s)

The Settings are immutable, which means that the value can't be changed, we can only make a copy with another value.
When assigning a new value to a Setting, the datatype of the value is validated against the expected datatype of the
parameter.

.. testsetup:: imports

    from exa.common.data.parameter import Parameter, Setting
    from exa.common.data.setting_node import SettingNode
    p = Parameter('qubit_1.drive.pi_pulse.duration', 'Duration', 's')

.. doctest:: imports

    >>> setting1 = p.set(500)  # an alternative way to transform a parameter into a setting
    >>> setting2 = setting1.update(300)
    >>> setting1.value
    500
    >>> setting2.value
    300
    >>> setting1.parameter == setting2.parameter
    True

"""

from __future__ import annotations

import ast
from collections.abc import Hashable
import copy
from enum import IntEnum
from typing import Any, Self, TypeAlias
import warnings

import numpy as np
from pydantic import model_validator
import xarray as xr

from exa.common.control.sweep.sweep_values import SweepValues
from exa.common.data.base_model import BaseModel
from exa.common.data.value import ObservationValue
from exa.common.errors.station_control_errors import ValidationError

CastType: TypeAlias = str | list["CastType"] | None
SourceType: TypeAlias = None | BaseModel | dict[str, Any]
"""Type for Setting sources."""


class DataType(IntEnum):
    """Parameter data type."""

    ANYTHING = 0
    FLOAT = 1
    COMPLEX = 2
    STRING = 3
    BOOLEAN = 4
    INT = 5

    NUMBER = 101  # Deprecated

    def cast(self, value: CastType) -> Any:
        if isinstance(value, list):
            return [self.cast(item) for item in value]
        else:
            return self._cast(value)

    def validate(self, value: Any) -> bool:
        if value is None or self is DataType.ANYTHING:
            return True
        elif isinstance(value, np.generic):
            return self.validate(value.item())
        elif self in [DataType.FLOAT, DataType.NUMBER]:
            # Accept int as well, i.e. 1 == 1.0
            return type(value) in (int, float)
        elif self is DataType.INT:
            # Accept float as well, i.e. 1.0 == 1
            if type(value) in (int, float):
                return value % 1 == 0
            return False
        elif self is DataType.COMPLEX:
            return type(value) in (int, float, complex)
        elif self is DataType.STRING:
            return isinstance(value, str)
        elif self is DataType.BOOLEAN:
            return isinstance(value, bool)
        else:
            return False

    def _cast(self, value: str | None) -> Any:  # noqa: PLR0911
        if value is None:
            return None
        elif self in [DataType.FLOAT, DataType.NUMBER]:
            return float(value)
        elif self is DataType.INT:
            return int(value)
        elif self is DataType.COMPLEX:
            if isinstance(value, complex):
                return value
            return complex("".join(value.split()))
        elif self is DataType.BOOLEAN:
            if value.lower() == "true" or value == "1":
                return True
            if value.lower() == "false" or value == "0":
                return False
            raise TypeError("Boolean data types can only be 'false', 'true, '0' or '1' (case-insensitive)")
        elif self is DataType.STRING:
            return value
        else:  # TODO: can this be removed?
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):  # if the value can not be evaluated, return the original value
                return value


class CollectionType(IntEnum):
    """Parameter collection type."""

    SCALAR = 0
    """Scalar, not a list of any kind."""
    LIST = 1
    """Python list."""
    NDARRAY = 2
    """Numpy ndarray."""

    def cast(self, value: Any) -> Any:
        """Cast the given value to this collection type."""
        if self is CollectionType.NDARRAY and isinstance(value, list):
            return np.asarray(value)
        if self is CollectionType.LIST:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if not isinstance(value, list) and value is not None:
                return [value]
        return value


class Parameter(BaseModel):
    """A basic data structure that represents a single variable.

    The variable can be a high-level or low-level control knob of an instrument such as the amplitude of a pulse
    or a control voltage; a physical quantity such as resonance frequency; or an abstract concept
    like the number of averages in a measurement.

    :class:`.Setting` combines Parameter with a numerical, boolean, or string value to represent a quantity.
    """

    name: str
    """Parameter name used as identifier"""
    label: str = ""
    """Parameter label used as pretty identifier for display purposes. Default: `name`"""
    unit: str = ""
    """SI unit of the quantity, if applicable."""
    data_type: DataType | tuple[DataType, ...] = DataType.FLOAT
    """Data type or a tuple of datatypes that this parameter accepts and validates. One of :class:`.DataType`.
    Default: FLOAT."""
    collection_type: CollectionType = CollectionType.SCALAR
    """Data format that this parameter accepts and validates. One of :class:`.CollectionType`.
    Default: SCALAR."""
    element_indices: int | list[int] | None = None
    """For parameters representing a single value in a collection-valued parent parameter, this field gives the indices
    of that value. If populated, the ``self.name`` and ``self.label`` will be updated in post init to include
    the indices (becoming ``"<parent name>__<index0>__<index1>__...__<indexN>"`` and ``"<parent label> <indices>"``
    , respectively). The parent name can then be retrieved with ``self.parent_name`` and the parent label with
    ``self.parent_label``."""

    _parent_name: str | None = None
    _parent_label: str | None = None

    def __init__(
        self,
        name: str,
        label: str = "",
        unit: str = "",
        data_type: DataType | tuple[DataType, ...] = DataType.FLOAT,
        collection_type: CollectionType = CollectionType.SCALAR,
        element_indices: int | list[int] | None = None,
        **kwargs,
    ) -> None:
        if not label:
            label = name
        if data_type == DataType.NUMBER:
            warnings.warn(
                "data_type 'DataType.NUMBER' is deprecated, use 'DataType.FLOAT' instead.",
                DeprecationWarning,
            )
            # Consider DataType.NUMBER as a deprecated alias for DataType.FLOAT
            data_type = DataType.FLOAT
        super().__init__(
            name=name,
            label=label,
            unit=unit,
            data_type=data_type,
            collection_type=collection_type,
            element_indices=element_indices,
            **kwargs,
        )
        if self.element_indices is not None:
            if self.collection_type is not CollectionType.SCALAR:
                raise ValidationError("Element-wise parameter must have 'CollectionType.SCALAR' collection type.")

            match self.element_indices:
                case [_i, *_more_is] as idxs:
                    # matches anything non-empty Sequence-ish, as a list, (but not dicts or str/bytes and such)
                    idxs = list(idxs)
                case int(idx):
                    idxs = [idx]
                case _:
                    raise ValidationError("Parameter 'element_indices' must be one or more ints.")
            object.__setattr__(self, "element_indices", idxs)
            # there may be len(idxs) num of "__" separated indices at the end, remove those to get the parent name
            seperated_indices = "__".join([str(idx) for idx in idxs])
            parent_name = self.name.replace("__" + seperated_indices, "")
            object.__setattr__(self, "_parent_name", parent_name)
            object.__setattr__(self, "_parent_label", self.label.replace(f" {idxs}", ""))
            object.__setattr__(self, "label", f"{self._parent_label} {idxs}")
            name = parent_name + "__" + seperated_indices
            object.__setattr__(self, "name", name)

    @property
    def parent_name(self) -> str | None:
        """Returns the parent name.

        This `None` except in element-wise parameters where gives the name of the parent parameter.
        """
        return self._parent_name

    @property
    def parent_label(self) -> str | None:
        """Returns the parent label.

        This `None` except in element-wise parameters where gives the label of the parent parameter.
        """
        return self._parent_label

    def set(self, value: Any) -> Setting:
        """Create a Setting object with given `value`."""
        self.validate(value)
        return Setting(self, value)

    @staticmethod
    def build_data_set(
        variables: list[tuple[Parameter, list[Any]]],
        data: tuple[Parameter, SweepValues],
        attributes: dict[str, Any] | None = None,
        extra_variables: list[tuple[str, int]] | None = None,
    ):
        """Build an xarray Dataset, where the only DataArray is given by `results` and coordinates are given by
        `variables`. The data is reshaped to correspond to the sizes of the variables. For example,
        ``variables = [(par_x, [1,2,3]), (par_y: [-1, -2])]`` will shape the data to 3-by-2 array. If there are not
        enough `variables` to reshape the data, remaining dimensions can be given by `extra_variables`. For example,
        ``variables = [(par_x: [1,2,3])], extra_variables=[('y', 2)]`` yields the same 3-by-2 data. ``'y'`` will then be
        a "dimension without coordinate" in xarray terms.

        Args:
             variables: Coordinates of the set.
             data: Data Parameter and associated data as a possible nested list.
             attributes: metadata to attach to the whole Dataset.
             extra_variables: Valueless dimensions and their sizes.

        """
        variable_names: list[str] = []
        variable_sizes: list[int] = []
        variable_data_arrays: dict[str, xr.DataArray] = {}
        for variable in variables:
            variable_names.append(variable[0].name)
            variable_sizes.append(len(variable[1]))
            variable_data_arrays[variable[0].name] = variable[0].build_data_array(np.asarray(variable[1]))
        variable_names.extend(variable[0] for variable in (extra_variables or []))
        variable_sizes.extend(variable[1] for variable in (extra_variables or []))

        observed_data = np.asarray(data[1]).reshape(tuple(variable_sizes), order="F")
        observed_data_array = {data[0].name: data[0].build_data_array(observed_data, variable_names)}
        if attributes is None:
            return xr.Dataset(data_vars=observed_data_array, coords=variable_data_arrays)
        else:
            return xr.Dataset(data_vars=observed_data_array, coords=variable_data_arrays, attrs=attributes)

    def validate(self, value: Any) -> bool:
        """Validate that given `value` matches the :attr:`data_type` and :attr:`collection_type`."""
        return self._validate(value, self.data_type)

    def _validate(self, value: Any, data_type: DataType | tuple[DataType, ...]) -> bool:
        # on the first iteration the `data_type` type is checked, in case it is a tuple,
        # current method is called once again with each `data_type` from tuple with further
        # checking of `collection_type`
        if isinstance(data_type, tuple):
            return any(self._validate(value, dt) for dt in data_type)
        if self.collection_type is CollectionType.LIST:
            return isinstance(value, list) and all(data_type.validate(v) for v in value)
        elif self.collection_type is CollectionType.NDARRAY:
            return isinstance(value, np.ndarray) and all(data_type.validate(v) for v in value.flat)
        else:
            return data_type.validate(value)

    def build_data_array(
        self,
        data: np.ndarray,
        dimensions: list[str] | list[Hashable] | None = None,
        coords: dict[Hashable, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Attach Parameter information to a numerical array.

        Given an array of numerical values, returns a corresponding :class:`xr.DataArray` instance
        that gets its name, units, and dimension names (unless explicitly given) from the
        :class:`Parameter` instance.

        Args:
            data: numerical values
            dimensions: names of the dimensions of ``data``
            coords: coordinates labeling the dimensions of ``data``
            metadata: additional :attr:`xr.DataArray.attrs`

        Returns:
            corresponding DataArray

        """
        if dimensions is None:
            if len(data.shape) == 1:
                dimensions = [self.name]
            else:
                dimensions = [f"{self.name}_{i}" for i in range(len(data.shape))]
        attrs = {
            "parameter": self,
            "standard_name": self.name,
            "long_name": self.label,
            "units": self.unit,
        }
        if metadata:
            intersection = set(attrs.keys()).intersection(set(metadata.keys()))
            if intersection:
                raise ValueError(f"Can not have keys {intersection} in metadata dictionary")
            attrs = {**attrs, **metadata}
        da = xr.DataArray(name=self.name, data=data, attrs=attrs, dims=dimensions, coords=coords)
        # copying the coordinate metadata, if present, to the new DataArray coordinates
        if coords:
            for key in [k for k in coords if isinstance(coords[k], xr.DataArray)]:
                da[key].attrs = coords[key].attrs
        return da

    def create_element_parameter_for(self, indices: int | list[int]) -> Parameter:
        """Utility for creating an element-wise parameter for a single value in a collection valued parameter.

        Args:
            indices: The indices in the collection for which to create the element-wise parameter.

        Returns:
            The element-wise parameter.

        Raises:
            UnprocessableEntityError: If ``self`` is not collection-valued.

        """
        if self.collection_type is CollectionType.SCALAR:
            raise ValidationError(
                "Cannot create an element-wise parameter from a parameter with 'CollectionType.SCALAR'."
            )
        return Parameter(
            name=self.name,
            label=self.label,
            unit=self.unit,
            data_type=self.data_type,
            collection_type=CollectionType.SCALAR,
            element_indices=indices,
        )


class Setting(BaseModel):
    """Physical quantity represented as a Parameter attached to a numerical value."""

    parameter: Parameter
    """The parameter this Setting represents."""
    value: ObservationValue
    """Data value attached to the parameter."""
    read_only: bool = False
    """Indicates if the attribute is read-only."""
    path: str = ""
    """Path in the settings tree (starting from the root ``SettingNode``) for this setting."""

    _source: SourceType = None
    """The source for this Setting value. May contain an observation (ObservationDefinition or ObservationData)
    or a source-dict (e.g. ``{"type": "configuration_source", "configurator": "defaults_from_yml"}``). By default,
    ``None``, which denotes the source not being specified (e.g. hardcoded defaults). The source is stored in a private
    attribute and thus is never serialized (the source field can contain non-serializable data such as Callables)."""

    def __init__(
        self,
        parameter: Parameter | None = None,
        value: ObservationValue | None = None,
        read_only: bool = False,
        path: str = "",
        source: SourceType = None,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter=parameter,
            value=value,
            read_only=read_only,
            path=path,
            **kwargs,
        )
        self._source = source

    @model_validator(mode="after")
    def validate_parameter_value_after(self) -> Self:
        if self.parameter.collection_type in (CollectionType.LIST, CollectionType.NDARRAY):
            # Use __setattr__ to set the value, since the instance is frozen.
            # Ideally, this value would be set in "before" validation.
            # However, "before" validation has to deal with raw data, which could be any arbitrary object.
            # To avoid extra complexity, let Pydantic deal with the raw data and change the value in "after" validation.
            object.__setattr__(self, "value", self.parameter.collection_type.cast(self.value))
        if self.value is not None and not self.parameter.validate(self.value):
            raise ValidationError(f"Invalid value '{self.value}' for parameter '{self.parameter}'.")
        return self

    def update(self, value: ObservationValue, source: SourceType = None) -> Setting:
        """Create a new setting object with updated value and source.

        Args:
            value: New value for the setting.
            source: New source for the setting.

        Returns:
            Copy of ``self`` with modified properties.

        """
        if self.read_only:
            raise ValueError(
                f"Can't update the value of {self.parameter.name} to {value} since the setting is read-only."
            )
        if isinstance(value, list) and self.parameter.collection_type == CollectionType.NDARRAY:
            value = np.array(value)
        # Need to create a new Setting here instead of using Pydantic model_copy().
        # model_copy() can't handle backdoor settings without errors, i.e. values with a list of 2 elements.
        return Setting(self.parameter, value, self.read_only, self.path, source=source)

    @property
    def name(self):
        """Name used as identifier, same as name of :attr:`parameter`."""
        return self.parameter.name

    @property
    def parent_name(self):
        """Parent name of the parameter of ``self``."""
        return self.parameter.parent_name

    @property
    def label(self):
        """Label used as pretty identifier for display purposes, same as label of :attr:`parameter`."""
        return self.parameter.label

    @property
    def parent_label(self):
        """Parent label of the parameter of ``self``."""
        return self.parameter.parent_label

    @property
    def unit(self):
        """SI unit of the :attr:`value`, if applicable, same as unit of :attr:`parameter`."""
        return self.parameter.unit

    @property
    def element_indices(self) -> int | list[int] | None:
        """Element-wise indices of the parameter in ``self``."""
        return self.parameter.element_indices

    @property
    def source(self) -> SourceType:
        """Return the source for this Setting's value."""
        return self._source

    @staticmethod
    def get_by_name(name: str, values: set[Setting]) -> Setting | None:
        return next((setting for setting in values if setting.parameter.name == name), None)

    @staticmethod
    def remove_by_name(name: str, values: set[Setting]) -> set[Setting]:
        removed = copy.deepcopy(values)
        if setting := Setting.get_by_name(name, values):
            removed.discard(setting)
        return removed

    @staticmethod
    def replace(settings: Setting | list[Setting], values: set[Setting] | None = None) -> set[Setting]:
        if values is None:
            values = set()
        if not isinstance(settings, list):
            settings = [settings]
        removed = values
        for setting in settings:
            removed = Setting.remove_by_name(setting.parameter.name, removed)
            removed.add(setting)
        return removed

    @staticmethod
    def diff_sets(first: set[Setting], second: set[Setting]) -> set[Setting]:
        """Return a one-sided difference between two sets of Settings, prioritising values in `first`.

        Args:
             first: Set whose values will be in the resulting diff.
             second: Set that is compared to `first`.

        Returns:
            A new set of Settings whose parameters are only found in `first`, and Settings in `first` whose
            values differ from their counterparts in `second`.

        """
        diff = first.difference(second)
        for s in first.intersection(second):
            a, b = [Setting.get_by_name(s.parameter.name, group) for group in [first, second]]
            if a is not None and b is not None and a.value != b.value:
                diff.add(a)
        return diff

    @staticmethod
    def merge(settings1: set[Setting], settings2: set[Setting]) -> set[Setting]:
        if settings1 is None:
            settings1 = set()
        if settings2 is None:
            settings2 = set()
        merged = copy.deepcopy(settings1)
        for setting in settings2:
            found = Setting.get_by_name(setting.parameter.name, merged)
            if found:
                Setting.replace(setting, merged)
            else:
                merged.add(setting)
        return merged

    def create_element_parameter_for(self, indices: int | list[int]) -> Parameter:
        """Utility for creating an element-wise parameter for a single value in a collection valued parameter.

        Args:
            indices: The indices in the collection for which to create the element-wise parameter.

        Returns:
            The element-wise parameter.

        Raises:
            ValueError: If ``self`` is not collection-valued.

        """
        return self.parameter.create_element_parameter_for(indices)

    def __hash__(self):
        return hash(self.parameter)

    def __eq__(self, other: Any) -> bool:
        if not (isinstance(other, Setting) and self.parameter == other.parameter):
            return False
        if isinstance(self.value, np.ndarray):
            if isinstance(other.value, np.ndarray):
                return np.array_equal(self.value, other.value)
            return False
        else:
            if isinstance(other.value, np.ndarray):
                return False
            return self.value == other.value

    def __lt__(self, other: Setting):
        return self.parameter.__lt__(other.parameter)

    def __str__(self):
        return f"Setting({self.parameter.label} = {self.value} {self.parameter.unit} {self.read_only=})"

    def with_path_name(self) -> Setting:
        """Copy of self with the parameter name replaced by the path name."""
        return self.model_copy(update={"parameter": self.parameter.model_copy(update={"name": self.path})})
