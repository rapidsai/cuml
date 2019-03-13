#
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


import cuml._common.handle


cdef class Base:
    """
    Base class for all the ML algos. It handles some of the common operations
    across all algos. Every ML algo class exposed at cython level must inherit
    from this class.
    """

    def __init__(self, handle=None, verbose=False):
        """
        Constructor. All children must call init method of this base class!

        Parameters
        ----------
        handle : cuml.handle.Handle
               If it is None, a new one is created just for this class
        verbose : bool
                Whether to print debug spews
        """
        if handle is None:
            self.__handle = cuml._common.handle.Handle()
        else:
            self.__handle = handle
        self.verbose = verbose


    @property
    def handle(self):
        """
        Sets the underlying cumlHandle structure

        Return
        ----------
        handle : cuml.handle.Handle
               The current cumlHandle used by this class
        """
        return self.__handle


    @handle.setter
    def handle(self, h):
        """
        Sets the underlying cumlHandle structure

        Parameters
        ----------
        h : cuml.handle.Handle
          cumlHandle which is used to manage cuda workload being scheduled
          from hereafterwards
        """
        self.__handle = h


    # TODO: implement a base definition for get_params and set_params?
