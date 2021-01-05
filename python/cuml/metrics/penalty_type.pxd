cdef extern from "cuml/metrics/penalty_type.hpp" namespace "MLCommon::Functions":

    ctypedef enum penalty:
        PenaltyNone "MLCommon::Functions::penalty::NONE"
        PenaltyL1 "MLCommon::Functions::penalty::L1"
        PenaltyL2 "MLCommon::Functions::penalty::L2"
        PenaltyElasticNet "MLCommon::Functions::penalty::ELASTICNET"
