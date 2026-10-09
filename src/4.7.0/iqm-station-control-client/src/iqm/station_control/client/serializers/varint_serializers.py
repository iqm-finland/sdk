# Copyright 2026 IQM
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
"""Helpers for protobuf-style varint serialization.

Protobuf automatically varint-encodes integer values inside a message body.
These functions are needed when a varint must appear outside any message -
for example, as a length prefix in a hand-assembled byte stream - where the
protobuf library has no opportunity to perform the encoding automatically.
"""


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer using protobuf varint encoding."""
    if value < 0:
        raise ValueError("Varint encoding only supports non-negative integers")

    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def decode_varint(buffer: bytes, offset: int) -> tuple[int, int]:
    """Decode a protobuf varint from ``buffer`` starting at ``offset``."""
    value = 0
    shift = 0

    while offset < len(buffer):
        byte = buffer[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 64:
            raise ValueError("Varint is too long")

    raise ValueError("Unexpected end of input while decoding varint")
