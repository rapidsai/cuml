#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.internals.validation import check_array


def model_func_call(X, model_func, gpu_model=False):
    """
    Function to call `model_func(X)` using either `NumPy` arrays if
    gpu_model is False or X directly if model_gpu based is True.
    Returns the results as CuPy arrays.
    """
    if gpu_model:
        y = model_func(X)
    else:
        try:
            y = model_func(cp.asnumpy(X))
        except TypeError:
            raise TypeError(
                "Explainer can only explain models that can "
                "take GPU data or NumPy arrays as input."
            )

    return check_array(y, ensure_2d=False, ensure_all_finite=False)


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
                "'link' string does not identify any known link functions. "
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
