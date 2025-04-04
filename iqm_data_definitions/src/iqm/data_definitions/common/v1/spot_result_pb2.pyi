"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file

Spot result declarations
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import iqm.data_definitions.common.v1.observation_pb2
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class NumcodecsConfig(google.protobuf.message.Message):
    """Describes a numcodec codec:
    https://numcodecs.readthedocs.io/en/stable/abc.html
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CONFIG_JSON_FIELD_NUMBER: builtins.int
    config_json: builtins.str
    """more elegant formats go here, eventually"""
    def __init__(
        self,
        *,
        config_json: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["config", b"config", "config_json", b"config_json"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["config", b"config", "config_json", b"config_json"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["config", b"config"]) -> typing.Literal["config_json"] | None: ...

global___NumcodecsConfig = NumcodecsConfig

@typing.final
class NpyArray(google.protobuf.message.Message):
    """A numeric array as raw bytes buffer, with "array header" to
    identify the data type; analogous to numpy's own '.npy' format (see numpy.lib.format).
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SHAPE_FIELD_NUMBER: builtins.int
    DTYPE_STR_FIELD_NUMBER: builtins.int
    FORTRAN_ORDER_FIELD_NUMBER: builtins.int
    ENCODE_NUMCODECS_FIELD_NUMBER: builtins.int
    ARRAY_BYTES_FIELD_NUMBER: builtins.int
    dtype_str: builtins.str
    """encodes dtype, as np.dtype.str. cf. numpy.lib.dtype_to_descr.
    note: encoding of record/named array dtypes currently undefined.
    """
    fortran_order: builtins.bool
    """whether array_bytes is serialized in Fortran order."""
    array_bytes: builtins.bytes
    """contents of array buffer in memory, encoded as described by `encode_numcodecs`."""
    @property
    def shape(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """np.ndarray.shape; it is permissible to elide a leading shape[0] > 1,
        which will be implied by length(array_bytes)
        """

    @property
    def encode_numcodecs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___NumcodecsConfig]:
        """Sequence of numcodecs [1] Codecs configuration JSON used to encode array_bytes.
        Order is that in which codecs were applied to encode; for decoding, use codecs in reverse order.
        (Empty list indicates `array_bytes` are the unencoded array buffer.)

        [1] https://numcodecs.readthedocs.io/en/stable/abc.html
        """

    def __init__(
        self,
        *,
        shape: collections.abc.Iterable[builtins.int] | None = ...,
        dtype_str: builtins.str = ...,
        fortran_order: builtins.bool = ...,
        encode_numcodecs: collections.abc.Iterable[global___NumcodecsConfig] | None = ...,
        array_bytes: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["array_bytes", b"array_bytes", "dtype_str", b"dtype_str", "encode_numcodecs", b"encode_numcodecs", "fortran_order", b"fortran_order", "shape", b"shape"]) -> None: ...

global___NpyArray = NpyArray

@typing.final
class SpotResultValue(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INT64_ARRAY_FIELD_NUMBER: builtins.int
    FLOAT64_ARRAY_FIELD_NUMBER: builtins.int
    COMPLEX128_ARRAY_FIELD_NUMBER: builtins.int
    NPY_ARRAY_FIELD_NUMBER: builtins.int
    @property
    def int64_array(self) -> iqm.data_definitions.common.v1.observation_pb2.Int64Array: ...
    @property
    def float64_array(self) -> iqm.data_definitions.common.v1.observation_pb2.Float64Array: ...
    @property
    def complex128_array(self) -> iqm.data_definitions.common.v1.observation_pb2.Complex128Array: ...
    @property
    def npy_array(self) -> global___NpyArray: ...
    def __init__(
        self,
        *,
        int64_array: iqm.data_definitions.common.v1.observation_pb2.Int64Array | None = ...,
        float64_array: iqm.data_definitions.common.v1.observation_pb2.Float64Array | None = ...,
        complex128_array: iqm.data_definitions.common.v1.observation_pb2.Complex128Array | None = ...,
        npy_array: global___NpyArray | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["complex128_array", b"complex128_array", "float64_array", b"float64_array", "int64_array", b"int64_array", "npy_array", b"npy_array", "value", b"value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["complex128_array", b"complex128_array", "float64_array", b"float64_array", "int64_array", b"int64_array", "npy_array", b"npy_array", "value", b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["value", b"value"]) -> typing.Literal["int64_array", "float64_array", "complex128_array", "npy_array"] | None: ...

global___SpotResultValue = SpotResultValue
