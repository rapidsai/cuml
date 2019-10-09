from libcpp.string cimport string

cdef extern from "common/nvtx.hpp" namespace "ML":

  void PUSH_RANGE(string msg)

  void POP_RANGE()


def pynvtx_range_push(msg):
    cdef string s = msg.encode("UTF-8")
    PUSH_RANGE(s.c_str())

def pynvtx_range_pop():
    POP_RANGE()
