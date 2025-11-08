#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# The following is taken from
# https://github.com/cython/cython/blob/master/Cython/Includes/libcpp/optional.pxd,
# which provides a binding for std::optional in Cython 3.0

from libcpp cimport bool


cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass nullopt_t:
        nullopt_t()

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        ctypedef T value_type
        optional()
        optional(nullopt_t)
        optional(optional&) except +
        optional(T&) except +
        bool has_value()
        T& value()
        T& value_or[U](U& default_value)
        void swap(optional&)
        void reset()
        T& emplace(...)
        T& operator*()
        # T* operator->() # Not Supported
        optional& operator=(optional&)
        optional& operator=[U](U&)
        bool operator bool()
        bool operator!()
        bool operator==[U](optional&, U&)
        bool operator!=[U](optional&, U&)
        bool operator<[U](optional&, U&)
        bool operator>[U](optional&, U&)
        bool operator<=[U](optional&, U&)
        bool operator>=[U](optional&, U&)

    optional[T] make_optional[T](...) except +
