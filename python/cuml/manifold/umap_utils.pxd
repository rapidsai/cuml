from libc.stdint cimport uint64_t
from libc.stdint cimport int64_t
from libcpp cimport bool


cdef extern from "cuml/manifold/umapparams.h" namespace "ML::UMAPParams":

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1

cdef extern from "cuml/common/callback.hpp" namespace "ML::Internals":

    cdef cppclass GraphBasedDimRedCallback

cdef extern from "cuml/manifold/umapparams.h" namespace "ML":

    cdef cppclass UMAPParams:
        int n_neighbors,
        int n_components,
        int n_epochs,
        float learning_rate,
        float min_dist,
        float spread,
        float set_op_mix_ratio,
        float local_connectivity,
        float repulsion_strength,
        int negative_sample_rate,
        float transform_queue_size,
        int verbosity,
        float a,
        float b,
        float initial_alpha,
        int init,
        int target_n_neighbors,
        MetricType target_metric,
        float target_weight,
        uint64_t random_state,
        bool deterministic,
        int optim_batch_size,
        GraphBasedDimRedCallback * callback
