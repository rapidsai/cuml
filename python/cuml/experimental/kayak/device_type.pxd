cdef extern from "cuml/experimental/kayak/device_type.hpp" namespace "kayak" nogil:
    cdef enum device_type:
        cpu "kayak::device_type::cpu",
        gpu "kayak::device_type::gpu"
