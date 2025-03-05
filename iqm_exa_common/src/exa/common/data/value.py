import base64
from typing import Annotated, Any

import numpy as np
from pydantic import PlainSerializer, PlainValidator, WithJsonSchema
from pydantic_core import core_schema


def validate_value(value: Any) -> Any:
    """Validate (i.e. deserialize) JSON serializable value to Python type, to support complex and ndarray types."""
    if isinstance(value, dict):
        if "__complex__" in value:
            value = complex(value["real"], value["imag"])
        elif "__ndarray__" in value:
            data = base64.b64decode(value["data"])
            value = np.frombuffer(data, value["dtype"]).reshape(value["shape"])
    return value


def serialize_value(value: Any) -> Any:
    """Serialize value type to JSON serializable type, to support complex and ndarray types."""
    if isinstance(value, complex):
        value = {"__complex__": "true", "real": value.real, "imag": value.imag}
    elif isinstance(value, np.ndarray):
        # ensure array buffer is contiguous and in C order
        value = np.require(value, requirements=["A", "C"])
        data = base64.b64encode(value.data)
        value = {"__ndarray__": "true", "data": data, "dtype": str(value.dtype), "shape": value.shape}
    return value


# TODO: We might want to rename these to ObservationValue and ObservationUncertainty, respectively.
Value = Annotated[
    bool | str | int | float | complex | np.ndarray,
    PlainValidator(validate_value),
    PlainSerializer(serialize_value),
    WithJsonSchema(core_schema.any_schema()),
]
Uncertainty = Annotated[
    int | float | complex | np.ndarray,
    PlainValidator(validate_value),
    PlainSerializer(serialize_value),
    WithJsonSchema(core_schema.any_schema()),
]

# TODO: Consider if we want to rename these permanently to avoid unnecessary "as" imports
ObservationValue = Value
ObservationUncertainty = Uncertainty
