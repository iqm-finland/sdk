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

"""Functions that encode and decode all sweep arguments(different types of
sweeps, return parameters, settings, etc.). The module provides functions for
serializing and deserializing sweep arguments before saving them to database.
"""

import json
from typing import Any, Optional, Union, cast

from exa.common.api.model.parameter_model import ParameterModel
from exa.common.api.model.setting_node_model import SettingNodeModel
from exa.common.api.model.sweep_model import SweepModel
from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.helpers.json_helper import decode_json, get_json_encoder
from exa.common.sweep.util import NdSweep, Sweeps, linear_index_sweep


def encode_nd_sweeps(sweeps: NdSweep, **kwargs) -> str:
    """Encode sweeps to a JSON string.

    Args:
        sweeps: sweeps to be serialized.
        kwargs: keyword arguments passed to json.dumps

    Returns:
        json as a string

    """
    kwargs.setdefault("cls", _SweepDataEncoder)
    return json.dumps(sweeps, **kwargs)


def encode_return_parameters_legacy(return_parameters: dict[Parameter, Optional[NdSweep]], **kwargs) -> str:
    """Encode sweeps to a JSON string.

    Args:
        return_parameters: Return parameters as specified by :meth:`~.Backend.sweep`.
        kwargs: keyword arguments passed to json.dumps

    Returns:
        json as a string

    """
    kwargs.setdefault("cls", _SweepDataEncoder)
    return json.dumps(return_parameters, **kwargs)


def encode_return_parameters(return_parameters: dict[Parameter, Optional[NdSweep]], **kwargs) -> str:
    """Encode sweeps to a JSON string.

    Args:
        return_parameters: Return parameters as specified by :meth:`~.Backend.sweep`.
        kwargs: keyword arguments passed to json.dumps

    Returns:
        json as a string

    """
    kwargs.setdefault("cls", _SweepDataEncoder)
    listed_params = [{"parameter": param, "hard_sweeps": sweeps} for param, sweeps in return_parameters.items()]
    return json.dumps(listed_params, **kwargs)


def decode_and_validate_sweeps(sweeps_json: str) -> Sweeps:
    """Decodes and validates json string of list of Sweeps and tuples of Sweeps.

    Args:
        sweeps_json: json string of nested structure of Sweep objects

    Returns:
        list of sweeps and tuples of sweeps

    Raises:
        ValueError if decoded result is not expected return type

    """
    decoded = _loads(sweeps_json)
    if not isinstance(decoded, list):
        raise ValueError(f"Outer type is not list type when decoding {sweeps_json}")

    for elem in decoded:
        if isinstance(elem, tuple):
            if any([not isinstance(t, Sweep) for t in elem]):
                raise ValueError("list of Sweeps must contain tuples of `Sweep` objects")
        elif not isinstance(elem, Sweep):
            raise ValueError(f"list must contain either Sweep or Tuple elements. Got {type(elem)}")
    return cast(Sweeps, decoded)


def decode_return_parameters_legacy(json_str: str) -> dict[Parameter, Optional[NdSweep]]:
    """Deserialize return parameters.

    For backwards compatibility, changes values of the return parameters dict to a new,
    more general format: NdSweeps, which is a list of tuples of Sweeps.

    A key in the dict may be a Parameter or a Setting; both will be converted to a Parameter.
    The values in the dict are converted with the following rules:

    * 1 is converted to an empty NdSweep, i.e., a scalar.
    * Other integers are converted to a :func:`.linear_index_sweep`
    * `Sweep` is converted to an NdSweep that contains only the sweep
    * NdSweep and None are not converted.

    Args:
        json_str: JSON representation of the ``return_parameters`` loaded
            from e.g. persistence

    Return:
        a reconstituted, typed ``return_parameters`` structure

    """

    def decode_key(key: str) -> Parameter:
        parameter_or_setting: dict[str, Any] = json.loads(key)
        parameter_dict = parameter_or_setting.get("parameter", parameter_or_setting)
        return ParameterModel(**parameter_dict).decode()

    raw: dict[str, Any] = _loads(json_str)
    decoded: dict[Setting, Union[NdSweep, Sweep, int]] = {decode_key(key): value for key, value in raw.items()}
    return _legacy_return_parameters_to_new_format(decoded)


def decode_return_parameters(json_str: str) -> dict[Parameter, Optional[NdSweep]]:
    """Deserialize return parameters.

    For backwards compatibility, changes values of the return parameters dict to a new,
    more general format: NdSweeps, which is a list of tuples of Sweeps.

    Return parameters JSON syntax: ``[{"parameter": readout_parameter_json, "hard_sweeps": [hard_sweep,...]}, ...]``.

    ``readout_parameter_json`` may be a Parameter or a Setting; both will be converted to a Parameter.
    ``hard_sweeps`` are converted with the following rules:

    * 1 is converted to an empty NdSweep, i.e., a scalar.
    * Other integers are converted to a :func:`.linear_index_sweep`
    * `Sweep` is converted to an NdSweep that contains only the sweep
    * NdSweep and None are not converted.

    Args:
        json_str: JSON representation of the ``return_parameters`` loaded
            from e.g. persistence

    Return:
        a reconstituted, typed ``return_parameters`` structure

    """
    decoded = _loads(json_str)
    if not isinstance(decoded, list):
        raise ValueError(f"Outer type is not list type when decoding: {json_str}")

    return_parameters = {
        ParameterModel(**param_and_sweeps["parameter"]).decode(): param_and_sweeps["hard_sweeps"]
        for param_and_sweeps in decoded
    }
    return return_parameters


def decode_settings(json_str: str) -> SettingNode:
    """Deserialize settings from json string

    Args:
        json_str: JSON representation of ``settings`` loaded
            from e.g. persistence

    Returns:
        deserialized settings

    """
    return SettingNodeModel(**json.loads(json_str)).decode()


def _legacy_return_parameters_to_new_format(
    old: dict[Union[Parameter, Setting], Union[NdSweep, Sweep, int, None]],
) -> dict[Parameter, NdSweep]:
    """For backwards compatibility, changes values of the return parameters dict to a new,
    more general format: NdSweeps, which is a list of tuples of Sweeps.

    Args:
        old: return parameters in old format.

    Returns:
        `old` coerced to the new format.

    """
    new = {}
    for key, value in old.items():
        parameter = key.parameter if isinstance(key, Setting) else key

        # Previously, int meant "I don't know the numerical coordinates of this hard sweep, so
        # just give me a dummy sweep with integers from 0 to <coord>"
        if isinstance(value, int):
            # Length 1 means scalar, so no hard sweeps at all
            new_value = linear_index_sweep(parameter, value) if value != 1 else []
        elif isinstance(value, Sweep):
            new_value = [(value,)]
        else:
            new_value = value
        new[parameter] = new_value
    return new


def _loads(*args, **kwargs) -> Any:
    """Extend json.loads with tuple and Sweep decoding support

    Args:
        args: positional arguments passed to json.dumps
        kwargs: keyword arguments passed to json.dumps

    Returns:
        python data structure that recovers encoded structure of
        nested tuples and Sweep -objects

    """

    def _decode_json(obj: Any) -> Any:
        if isinstance(obj, dict) and {"parameter", "data"}.issubset(obj):
            return SweepModel(**obj).decode()
        return decode_json(obj)

    kwargs.setdefault("object_hook", _decode_json)
    return json.loads(*args, **kwargs)


class _SweepDataEncoder(json.JSONEncoder):
    """Extension of json encoder to support handling of Sweep, Parameter, and tuples."""

    def default(self, obj: Any) -> Any:
        """Default encoder

        Args:
            obj: object to be encoded

        Returns:
            encoded object

        """
        if isinstance(obj, Sweep):
            return json.loads(SweepModel.encode(obj).model_dump_json())
        if isinstance(obj, Parameter):
            return json.loads(ParameterModel.encode(obj).model_dump_json())
        if isinstance(obj, tuple):
            return get_json_encoder()[tuple](obj)
        return super().default(obj)

    # JSONEncoder doesn't call `default` method for tuple, because
    # it knows how to serialize it. In order to use custom encoder
    # we need to override `encode` method
    def encode(self, obj: Any) -> Any:
        """Override JSONEncoder encode in order to change the
        default tuple behaviour.

        Args:
            obj: object to be encoded

        Returns:
            encoded object

        """

        def _encode_tuples(item):
            if isinstance(item, tuple):
                return get_json_encoder()[tuple](item)
            if isinstance(item, list):
                return [_encode_tuples(e) for e in item]
            if isinstance(item, dict):
                return {
                    key if isinstance(key, str) else self.encode(key): _encode_tuples(value)
                    for key, value in item.items()
                }
            return item

        return super().encode(_encode_tuples(obj))
