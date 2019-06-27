# Copyright (c) 2019, NVIDIA CORPORATION.
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

import logging
import numba.cuda
import time

from .utils import get_device_id, select_device

from threading import Lock, Thread


class IPCThread(Thread):
    """
    This mechanism gets around Numba's restriction of CUDA contexts being
    thread-local by creating a thread that can select its own device.
    This allows the user of IPC handles to open them up directly on the
    same device as the owner (bypassing the need for peer access.)
    """

    def __init__(self, ipcs, device):

        Thread.__init__(self)

        self.lock = Lock()
        self.ipcs = ipcs

        # Use canonical device id
        self.device = get_device_id(device)

        print("Starting new IPC thread on device %i for ipcs %s" %
              (self.device, str(list(ipcs))))
        self.running = False

    def run(self):

        select_device(self.device)

        print("Opening: " + str(self.device) + " "
              + str(numba.cuda.get_current_device()))

        self.lock.acquire()

        try:
            self.arrs = [ipc.open() for ipc in self.ipcs]
            self.ptr_info = [x.__cuda_array_interface__ for x in self.arrs]

            self.running = True
        except Exception as e:
            logging.error("Error opening ipc_handle on device " +
                          str(self.device) + ": " + str(e))

        self.lock.release()

        while (self.running):
            time.sleep(0.0001)

        try:
            logging.warn("Closing: " + str(self.device) +
                         str(numba.cuda.get_current_device()))
            self.lock.acquire()
            [ipc.close() for ipc in self.ipcs]
            self.lock.release()

        except Exception as e:
            logging.error("Error closing ipc_handle on device " +
                          str(self.device) + ": " + str(e))

    def close(self):

        """
        This should be called before calling join(). Otherwise, IPC handles
        may not be properly cleaned up.
        """
        self.lock.acquire()
        self.running = False
        self.lock.release()

    def info(self):
        """
        Warning: this method is invoked from the calling thread. Make
        sure the context in the thread reading the memory is tied to
        self.device, otherwise an expensive peer access might take
        place underneath.
        """
        while (not self.running):
            time.sleep(0.0001)

        return self.ptr_info


def new_ipc_thread(ipcs, dev):
    t = IPCThread(ipcs, dev)
    t.start()
    return t
