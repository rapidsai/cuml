#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cupy as cp


def get_tag_from_model_func(func, tag, default=None):
    ""
    tags_fn = getattr(
        getattr(func, '__self__', None),
        '_get_tags',
        None
    )

    if tags_fn is not None:
        tag_value = tags_fn.get(tag)
        result = tag_value if tag_value is not None else default

        return result

    return default


def get_dtype_from_model_func(func, default=None):
    dtype = getattr(
        getattr(func, '__self__', None),
        'dtype',
        None
    )

    dtype = default if dtype is None else dtype

    return dtype


def get_link_fn_from_str(link):
    if isinstance(link, str):
        if link in link_dict:
            link_fn = link_dict[link]
        else:
            return ValueError("'link' string does not identify any known"
                              " link functions. ")
    elif callable(link):
        if callable(getattr(link, "inverse", None)):
            link_fn = link
        else:
            raise TypeError("'link' function {} is not valid.".format(link))

    return link_fn


def model_call(X, model, model_gpu_based=False):
    if model_gpu_based:
        y = model(X)
    else:
        try:
            y = cp.array(model(
                X.to_output('numpy'))
            )
        except TypeError:
            raise TypeError('Explainer can only explain models that can '
                            'take GPU data or NumPy arrays as input.')

    return y


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


link_dict = {
    'identity': identity,
    'logit': logit
}
