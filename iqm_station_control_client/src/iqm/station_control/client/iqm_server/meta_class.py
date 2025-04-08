# Copyright 2025 IQM
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
import inspect


class IqmServerClientMeta(type):
    """Custom metaclass that automatically adds '<feature> not implemented'
    stub implementations for all StationControlClient functions that are
    not overridden by IqmServerClient.
    """

    def __new__(cls, name, bases, dct):
        for f_name, _ in inspect.getmembers(
            bases[0], predicate=lambda m: inspect.isfunction(m) and not m.__name__.startswith("__")
        ):
            if f_name not in dct:
                dct[f_name] = _not_implemented_stub(f_name)
        return super().__new__(cls, name, bases, dct)


def _not_implemented_stub(feature: str):
    """Generate a function that raises NotImplementedError."""

    def stub(*args, **kwargs):
        raise NotImplementedError(f"'{feature}' is not implemented for this backend")

    return stub
