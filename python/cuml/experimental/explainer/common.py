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


def get_model_order_from_tags(model,
                              default='F'):
    tags_fn = getattr(
        getattr(model.predict, '__self__', None),
        '_get_tags',
        None
    )

    if tags_fn is not None:
        order = tags_fn.get('preferred_input_order')
        result = order if order is not None else default

    return result


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
