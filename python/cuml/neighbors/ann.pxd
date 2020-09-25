#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# distutils: language = c++


from libc.stdint cimport uintptr_t
from libcpp cimport bool

cimport cuml.common.cuda


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":
    cdef cppclass knnIndexParam:
        bool automated

    ctypedef enum QuantizerType:
        QT_8bit,
        QT_4bit,
        QT_8bit_uniform,
        QT_4bit_uniform,
        QT_fp16,
        QT_8bit_direct,
        QT_6bit

    cdef cppclass IVFParam(knnIndexParam):
        int nlist
        int nprobe

    cdef cppclass IVFFlatParam(IVFParam):
        pass

    cdef cppclass IVFPQParam(IVFParam):
        int M
        int n_bits
        bool usePrecomputedTables

    cdef cppclass IVFSQParam(IVFParam):
        QuantizerType qtype
        bool encodeResidual


cdef inline check_algo_params(algo, params):
    def check_param_list(params, param_list):
        for param in param_list:
            if not hasattr(params, param):
                ValueError('algo_params misconfigured : {} \
                            parameter unset'.format(param))
    if algo == 'ivfflat':
        check_param_list(params, ['nlist', 'nprobe'])
    elif algo == "ivfpq":
        check_param_list(params, ['nlist', 'nprobe', 'M', 'n_bits',
                                  'usePrecomputedTables'])
    elif algo == "ivfsq":
        check_param_list(params, ['nlist', 'nprobe', 'qtype',
                                  'encodeResidual'])


cdef inline build_ivfflat_algo_params(params, automated):
    cdef IVFFlatParam* algo_params = new IVFFlatParam()
    if automated:
        return <uintptr_t>algo_params
    algo_params.nlist = <int> params['nlist']
    algo_params.nprobe = <int> params['nprobe']
    return <uintptr_t>algo_params


cdef inline build_ivfpq_algo_params(params, automated):
    cdef IVFPQParam* algo_params = new IVFPQParam()
    if automated:
        return <uintptr_t>algo_params
    algo_params.nlist = <int> params['nlist']
    algo_params.nprobe = <int> params['nprobe']
    algo_params.M = <int> params['M']
    algo_params.n_bits = <int> params['n_bits']
    algo_params.usePrecomputedTables = \
        <bool> params['usePrecomputedTables']
    return <uintptr_t>algo_params


cdef inline build_ivfsq_algo_params(params, automated):
    cdef IVFSQParam* algo_params = new IVFSQParam()
    if automated:
        return <uintptr_t>algo_params

    quantizer_type = {
        'QT_8bit': <int> QuantizerType.QT_8bit,
        'QT_4bit': <int> QuantizerType.QT_4bit,
        'QT_8bit_uniform': <int> QuantizerType.QT_8bit_uniform,
        'QT_4bit_uniform': <int> QuantizerType.QT_4bit_uniform,
        'QT_fp16': <int> QuantizerType.QT_fp16,
        'QT_8bit_direct': <int> QuantizerType.QT_8bit_direct,
        'QT_6bit': <int> QuantizerType.QT_6bit,
    }

    algo_params.nlist = <int> params['nlist']
    algo_params.nprobe = <int> params['nprobe']
    algo_params.qtype = <QuantizerType> quantizer_type[params['qtype']]
    algo_params.encodeResidual = <bool> params['encodeResidual']
    return <uintptr_t>algo_params


cdef inline build_algo_params(algo, params):
    automated = params is None or params == 'auto'
    if not automated:
        check_algo_params(algo, params)

    automated = params is None or params == 'auto'
    cdef knnIndexParam* algo_params = <knnIndexParam*> 0
    if algo == 'ivfflat':
        algo_params = <knnIndexParam*><uintptr_t> \
            build_ivfflat_algo_params(params, automated)
    if algo == 'ivfpq':
        algo_params = <knnIndexParam*><uintptr_t> \
            build_ivfpq_algo_params(params, automated)
    elif algo == 'ivfsq':
        algo_params = <knnIndexParam*><uintptr_t> \
            build_ivfsq_algo_params(params, automated)

    algo_params.automated = <bool>automated
    return <uintptr_t>algo_params


cdef inline destroy_algo_params(ptr):
    cdef knnIndexParam* algo_params = <knnIndexParam*> <uintptr_t> ptr
    del algo_params
