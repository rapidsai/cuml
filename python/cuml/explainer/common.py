#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cupy_array
from pylibraft.common.handle import Handle
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")


def get_tag_from_model_func(func, tag, default=None):
    """
    Function returns the tags from the model that function `func` is bound to.

    Parameters
    ----------
    func: object
        Function to check whether the object it is bound to has a _get_tags
        attribute, and return tags from it.
    tag: str
        Tag that will be returned if exists
    default: object  (default = None)
        Value that will be returned if tags cannot be fetched.
    """
    tags_fn = getattr(getattr(func, "__self__", None), "_get_tags", None)

    if tags_fn is not None:
        tag_value = tags_fn().get(tag)
        result = tag_value if tag_value is not None else default

        return result

    return default


def get_handle_from_cuml_model_func(func, create_new=False):
    """
    Function to obtain a RAFT handle from the object that `func` is bound to
    if possible.

    Parameters
    ----------
    func: object
        Function to check whether the object it is bound to has a _get_tags
        attribute, and return tags from it.
    create_new: boolean (default = False)
        Whether to return a new RAFT handle if none could be fetched. Otherwise
        the function will return None.
    """
    owner = getattr(func, "__self__", None)

    if owner is not None and isinstance(owner, Base):
        if owner.handle is not None:
            return owner.handle

    handle = Handle() if create_new else None
    return handle


def get_dtype_from_model_func(func, default=None):
    """
    Function detect if model that `func` is bound to prefers data of certain
    data type. It checks the attribute model.dtype.

    Parameters
    ----------
    func: object
        Function to check whether the object it is bound to has a _get_tags
        attribute, and return tags from it.
    create_new: boolean (default = False)
        Whether to return a new RAFT handle if none could be fetched. Otherwise
        the function will return None.
    """
    dtype = getattr(getattr(func, "__self__", None), "dtype", None)

    dtype = default if dtype is None else dtype

    return dtype


def model_func_call(X, model_func, gpu_model=False):
    """
    Function to call `model_func(X)` using either `NumPy` arrays if
    gpu_model is False or X directly if model_gpu based is True.
    Returns the results as CuPy arrays.
    """
    if gpu_model:
        y = input_to_cupy_array(X=model_func(X), order="K").array
    else:
        try:
            y = input_to_cupy_array(model_func(cp.asnumpy(X))).array
        except TypeError:
            raise TypeError(
                "Explainer can only explain models that can "
                "take GPU data or NumPy arrays as input."
            )

    return y


def get_cai_ptr(X):
    """
    Function gets the pointer from an object that supports the
    __cuda_array_interface__. Raises TypeError if `X` does not support it.
    """
    if hasattr(X, "__cuda_array_interface__"):
        return X.__cuda_array_interface__["data"][0]
    else:
        raise TypeError("X must support `__cuda_array_interface__`")


def get_link_fn_from_str_or_fn(link):
    if isinstance(link, str):
        if link in link_dict:
            link_fn = link_dict[link]
        else:
            raise ValueError(
                "'link' string does not identify any known" " link functions. "
            )
    elif callable(link):
        if callable(getattr(link, "inverse", None)):
            link_fn = link
        else:
            raise TypeError("'link' function {} is not valid.".format(link))

    return link_fn


def output_list_shap_values(X, dimensions, output_type):
    if output_type == "cupy":
        if dimensions == 1:
            return X[0]
        else:
            res = []
            for x in X:
                res.append(x)
            return res
    else:
        if dimensions == 1:
            return cp.asnumpy(X[0])
        else:
            res = []
            for x in X:
                res.append(cp.asnumpy(x))
            return res


# link functions


def identity(x):
    return x


def _identity_inverse(x):
    return x


def logit(x):
    return cp.log(x / (1 - x))


def _logit_inverse(x):
    return 1 / (1 + cp.exp(-x))


identity.inverse = _identity_inverse
logit.inverse = _logit_inverse


link_dict = {"identity": identity, "logit": logit}
