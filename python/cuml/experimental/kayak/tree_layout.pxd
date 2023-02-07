cdef extern from "cuml/experimental/kayak/tree_layout.hpp" namespace "kayak" nogil:
    cdef enum tree_layout:
        depth_first "kayak::tree_layout::depth_first",
        breadth_first "kayak::tree_layout::breadth_first"
