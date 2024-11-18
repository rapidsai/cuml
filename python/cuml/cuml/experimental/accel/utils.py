#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

from typing import Any


class _Unusable:
    """
    A totally unusable type. When a "fast" object is not available,
    it's useful to set it to _Unusable() so that any operations
    on it fail, and ensure fallback to the corresponding
    "slow" object.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError(
            "Fast implementation not available. "
            "Falling back to the slow implementation"
        )

    def __getattribute__(self, name: str) -> Any:
        if name in {"__class__"}:  # needed for type introspection
            return super().__getattribute__(name)
        raise TypeError("Unusable type. Falling back to the slow object")

    def __repr__(self) -> str:
        raise AttributeError("Unusable type. Falling back to the slow object")


@functools.lru_cache(maxsize=None)
def get_final_type_map():
    """
    Return the mapping of all known fast and slow final types to their
    corresponding proxy types.
    """
    return dict()


@functools.lru_cache(maxsize=None)
def get_intermediate_type_map():
    """
    Return a mapping of all known fast and slow intermediate types to their
    corresponding proxy types.
    """
    return dict()


@functools.lru_cache(maxsize=None)
def get_registered_functions():
    return dict()