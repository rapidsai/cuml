#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#
import functools
import typing

import cupy as cp

CUPY_SPARSE_DTYPES = [cp.float32, cp.float64, cp.complex64, cp.complex128]

# Use _DecoratorType as a type variable for decorators. See:
# https://github.com/python/mypy/pull/8336/files#diff-eb668b35b7c0c4f88822160f3ca4c111f444c88a38a3b9df9bb8427131538f9cR260
_DecoratorType = typing.TypeVar(
    "_DecoratorType", bound=typing.Callable[..., typing.Any]
)


def wraps_typed(
    wrapped: _DecoratorType,
    assigned=("__doc__", "__annotations__"),
    updated=functools.WRAPPER_UPDATES,
) -> typing.Callable[[_DecoratorType], _DecoratorType]:
    """
    Typed version of `functools.wraps`. Allows decorators to retain their
    return type.
    """
    return functools.wraps(wrapped=wrapped, assigned=assigned, updated=updated)
