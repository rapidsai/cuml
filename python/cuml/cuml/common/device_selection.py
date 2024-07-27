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


from cuml.internals.global_settings import GlobalSettings
from cuml.internals.device_type import DeviceType


def set_global_device_type(device_type):
    GlobalSettings().device_type = DeviceType.from_str(device_type)


def get_global_device_type():
    return GlobalSettings().device_type


class using_device_type:
    def __init__(self, device_type):
        self.device_type = device_type
        self.prev_device_type = None

    def __enter__(self):
        self.prev_device_type = GlobalSettings().device_type
        set_global_device_type(self.device_type)

    def __exit__(self, *_):
        set_global_device_type(self.prev_device_type)
