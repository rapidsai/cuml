#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import cuml.internals
import typing

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray


def _get_base_return_type(class_name, attr):

    if (not hasattr(attr, "__annotations__")
            or "return" not in attr.__annotations__):
        return None

    try:
        type_hints = typing.get_type_hints(attr)

        if ("return" in type_hints):

            ret_type = type_hints["return"]

            is_generic = isinstance(ret_type, typing._GenericAlias)

            if (is_generic):
                return _process_generic(ret_type)
            elif (issubclass(ret_type, CumlArray)):
                return "array"
            elif (issubclass(ret_type, SparseCumlArray)):
                return "sparsearray"
            elif (issubclass(ret_type, cuml.internals.base.Base)):
                return "base"
            else:
                return None
    except NameError:
        # A NameError is raised if the return type is the same as the
        # type being defined (which is incomplete). Check that here and
        # return base if the name matches
        if (attr.__annotations__["return"] == class_name):
            return "base"
    except Exception:
        assert False, "Shouldnt get here"
        return None

    return None
