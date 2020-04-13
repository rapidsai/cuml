
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.string cimport string


cdef extern from "cuml/common/logger.hpp" namespace "ML" nogil:
    cdef cppclass Logger:
        pass


def set_level(level):
    Logger::get().setLevel(<int>level)


def set_pattern(pattern):
    cdef string s = pattern
    Logger::get().setPattern(s)


def should_log_for(level):
    return Logger::get().shouldLogFor(<int>level)


def get_pattern():
    return Logger::get().getPattern()
