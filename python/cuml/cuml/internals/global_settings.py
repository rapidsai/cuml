#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import threading

from cuml.internals.device_type import DeviceType
from cuml.internals.mem_type import MemoryType


class _GlobalSettingsData(threading.local):  # pylint: disable=R0903
    """Thread-local storage class with per-thread initialization of default
    values for global settings"""

    def __init__(self):
        super().__init__()
        self.shared_state = {
            "root_cm": None,
            "_output_type": None,
            "_fil_device_type": DeviceType.device,
            "_fil_memory_type": MemoryType.device,
            "_memory_type": MemoryType.device,
        }


_global_settings_data = _GlobalSettingsData()


class GlobalSettings:
    """A thread-local borg class for tracking cuML global settings

    Because cuML makes use of internal context managers which try to minimize
    the number of conversions among various array types during internal calls,
    it is necessary to track certain settings globally. For instance, users can
    set a global output type, and cuML will ensure that the output is converted
    to the requested type *only* when a given API call returns to an external
    caller. Tracking when this happens requires globally-managed state.

    This class serves as a thread-local data store for any required global
    state. It is a thread-local borg, so updating an attribute on any instance
    of this class will update that attribute on *all* instances in the same
    thread. This additional layer of indirection on top of an ordinary
    `threading.local` object is to facilitate debugging of global settings
    changes. New global setting attributes can be added as properties to this
    object, and breakpoints or debugging statements can be added to a
    property's method to track when and how those properties change.

    In general, cuML developers should simply access `cuml.global_settings`
    rather than re-instantiating separate instances of this class in order to
    avoid the overhead of re-instantiation, but using a separate instance
    should not cause any logical errors.
    """

    def __init__(self):
        self.__dict__ = _global_settings_data.shared_state

    @property
    def memory_type(self):
        return self._memory_type

    @memory_type.setter
    def memory_type(self, value):
        self._memory_type = value

    @property
    def output_type(self):
        """The globally-defined default output type for cuML API calls"""
        return self._output_type  # pylint: disable=no-member

    @output_type.setter
    def output_type(self, value):
        self._output_type = value

    @property
    def xpy(self):
        return self.memory_type.xpy

    @property
    def fil_device_type(self):
        """The device type used for FIL"""
        return self._fil_device_type

    @fil_device_type.setter
    def fil_device_type(self, value):
        self._fil_device_type = value
        # Only change the memory type if current value is incompatible with new
        # device
        if not self._fil_device_type.is_compatible(self.fil_memory_type):
            self.fil_memory_type = self._fil_device_type.default_memory_type

    @property
    def fil_memory_type(self):
        """The memory type used for FIL"""
        return self._fil_memory_type

    @fil_memory_type.setter
    def fil_memory_type(self, value):
        self._fil_memory_type = value

    @property
    def fil_xpy(self):
        return self.fil_memory_type.xpy
