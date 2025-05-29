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

import warnings

from cuml.internals.global_settings import GlobalSettings


def _warn_deprecated(method):
    warnings.warn(
        f"`{method}` is deprecated and will be removed in 25.08.\n\n"
        "Execution of cuML models (except FIL) will always run on GPU now, please use the "
        "corresponding CPU-based projects (sklearn, umap-learn, ...) for handling any CPU execution.\n\n"
        "If you're using FIL please use `cuml.fil.set_fil_device_type`/`cuml.fil.get_fil_device_type` "
        "instead.\n\n"
        "For all other users please use either:\n"
        "- `model.as_sklearn()`/`model_class.from_sklearn()` to coerce cuML models to/from their CPU counterparts\n"
        "- `cuml.accel` for zero-code-change execution of the same code on CPU or GPU.",
        FutureWarning,
        stacklevel=3,
    )


def set_global_device_type(device_type):
    from cuml.fil import set_fil_device_type

    _warn_deprecated("set_global_device_type")
    set_fil_device_type(device_type)


def get_global_device_type():
    _warn_deprecated("get_global_device_type")
    return GlobalSettings().device_type


class using_device_type:
    def __init__(self, device_type):
        _warn_deprecated("using_device_type")
        self.device_type = device_type
        self.prev_device_type = None

    def __enter__(self):
        from cuml.fil import set_fil_device_type

        self.prev_device_type = GlobalSettings().fil_device_type
        set_fil_device_type(self.device_type)

    def __exit__(self, *_):
        GlobalSettings().fil_device_type = self.prev_device_type
