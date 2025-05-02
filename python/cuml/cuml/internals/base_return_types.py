#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import typing

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray


def _process_generic(gen_type):

    # Check if the type is not a generic. If not, must return "generic" if
    # subtype is CumlArray otherwise None
    if not isinstance(gen_type, typing._GenericAlias):
        if issubclass(gen_type, CumlArray):
            return "generic"

        # We don't handle SparseCumlArray at this time
        if issubclass(gen_type, SparseCumlArray):
            raise NotImplementedError(
                "Generic return types with SparseCumlArray are not supported "
                "at this time"
            )

        # Otherwise None (keep processing)
        return None

    # Its a generic type by this point. Support Union, Tuple, Dict and List
    supported_gen_types = [
        tuple,
        dict,
        list,
        typing.Union,
    ]

    if gen_type.__origin__ in supported_gen_types:
        # Check for a CumlArray type in the args
        for arg in gen_type.__args__:
            inner_type = _process_generic(arg)

            if inner_type is not None:
                return inner_type
    else:
        raise NotImplementedError("Unknow generic type: {}".format(gen_type))

    return None


def _get_base_return_type(class_name, attr):

    if (
        not hasattr(attr, "__annotations__")
        or "return" not in attr.__annotations__
    ):
        return None

    try:
        type_hints = typing.get_type_hints(attr)

        if "return" in type_hints:

            ret_type = type_hints["return"]

            is_generic = isinstance(ret_type, typing._GenericAlias)

            if is_generic:
                return _process_generic(ret_type)
            elif issubclass(ret_type, CumlArray):
                return "array"
            elif issubclass(ret_type, SparseCumlArray):
                return "sparsearray"
            elif issubclass(ret_type, cuml.internals.base.Base):
                return "base"
            else:
                return None
    except NameError:
        # A NameError is raised if the return type is the same as the
        # type being defined (which is incomplete). Check that here and
        # return base if the name matches
        # Cython 3 changed to preferring types rather than strings for
        # annotations. Strings end up wrapped in an extra layer of quotes,
        # which we have to replace here.
        if attr.__annotations__["return"].replace("'", "") == class_name:
            return "base"
    except Exception:
        raise AssertionError(
            f"Failed to determine return type for {attr} (class = '${class_name}'). This is a bug in cuML, please report it."
        )

    return None
