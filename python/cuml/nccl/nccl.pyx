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

from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

from libcpp cimport bool
from libc.stdlib cimport malloc, free

cdef extern from "cuML_comms_py.hpp" namespace "ML":
    void get_unique_id(char *uid, int size)
    void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId, int size)

cdef extern from "nccl.h":

    cdef struct ncclComm

    ctypedef struct ncclUniqueId:
        char *internal[128]

    ctypedef ncclComm *ncclComm_t

    ctypedef enum ncclResult_t:
        ncclSuccess
        ncclUnhandledCudaError
        ncclSystemError
        ncclInternalError
        ncclInvalidArgument
        ncclInvalidUsage
        ncclNumResults

    ncclResult_t ncclCommInitRank(ncclComm_t *comm,
                                  int nranks,
                                  ncclUniqueId commId,
                                  int rank)

    ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId)

    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank)

    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *count)

    const char *ncclGetErrorString(ncclResult_t result)

    ncclResult_t ncclCommAbort(ncclComm_t comm)

    ncclResult_t ncclCommDestroy(ncclComm_t comm)

NCCL_UNIQUE_ID_BYTES = 128

def unique_id():
    """
    Returns a new ncclUniqueId converted to a
    character array that can be safely serialized
    and shared to a remote worker.
    :return: string a 128-byte unique id string
    """
    cdef char *uid = <char *> malloc(NCCL_UNIQUE_ID_BYTES * sizeof(char))
    get_unique_id(uid, NCCL_UNIQUE_ID_BYTES)
    c_str = uid[:NCCL_UNIQUE_ID_BYTES-1]
    free(uid)
    return c_str


cdef class nccl:
    """
    A NCCL wrapper for initializing and closing NCCL comms
    in Python.
    """
    cdef ncclComm_t *comm

    cdef int size
    cdef int rank

    def __cinit__(self):
        self.comm = <ncclComm_t*>malloc(sizeof(ncclComm_t))

    def __dealloc__(self):

        comm_ = <ncclComm_t*>self.comm

        if comm_ != NULL:
            free(self.comm)
            self.comm = NULL

    @staticmethod
    def get_unique_id():
        """
        Returns a new nccl unique id
        :return: string nccl unique id
        """
        return unique_id()

    def init(self, nranks, commId, rank):
        """
        Construct a nccl-py object
        :param nranks: int size of clique
        :param commId: string unique id from client
        :param rank: int rank of current worker
        """

        self.size = nranks
        self.rank = rank

        cdef ncclUniqueId *ident = <ncclUniqueId*>malloc(sizeof(ncclUniqueId))
        ncclUniqueIdFromChar(ident, commId, NCCL_UNIQUE_ID_BYTES)

        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result = ncclCommInitRank(comm_, nranks,
                                                    deref(ident), rank)

        if result != ncclSuccess:
            err_str = ncclGetErrorString(result)
            print("NCCL_ERROR: %s" % err_str)

    def destroy(self):
        """
        Call destroy on the underlying NCCL comm
        """
        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result
        if comm_ != NULL:
            result = ncclCommDestroy(deref(comm_))

            if result != ncclSuccess:
                err_str = ncclGetErrorString(result)
                print("NCCL_ERROR: %s" % err_str)

            free(self.comm)
            self.comm = NULL

    def abort(self):
        """
        Call abort on the underlying nccl comm
        """
        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result
        if comm_ != NULL:
            result = ncclCommAbort(deref(comm_))

            if result != ncclSuccess:
                err_str = ncclGetErrorString(result)
                print("NCCL_ERROR: %s" % err_str)
            free(comm_)
            self.comm = NULL

    def cu_device(self):
        """
        Get the device backing the underlying comm
        :returns int device id
        """
        cdef int *dev = <int*>malloc(sizeof(int))

        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result = ncclCommCuDevice(deref(comm_), dev)

        if result != ncclSuccess:
            err_str = ncclGetErrorString(result)
            print("NCCL_ERROR: %s" % err_str)

        ret = dev[0]
        free(dev)
        return ret

    def user_rank(self):
        """
        Get the rank id of the current comm
        :return: int rank
        """

        cdef int *rank = <int*>malloc(sizeof(int))

        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result = ncclCommUserRank(deref(comm_), rank)

        if result != ncclSuccess:
            err_str = ncclGetErrorString(result)
            print("NCCL_ERROR: %s" % err_str)

        ret = rank[0]
        free(rank)
        return ret

    def get_comm(self):
        """
        Returns the underlying nccl comm in a size_t (similar to void*).
        This can be safely typecasted from size_t into ncclComm_t*
        :return: size_t ncclComm_t instance
        """
        return <size_t>self.comm
