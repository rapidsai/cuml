#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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


cdef check_algo_params(algo, params):
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


cdef build_ivfflat_algo_params(params, automated):
    cdef IVFFlatParam* algo_params = new IVFFlatParam()
    if automated:
        params = {
            'nlist': 8,
            'nprobe': 2
        }
    algo_params.nlist = <int> params['nlist']
    algo_params.nprobe = <int> params['nprobe']
    return <uintptr_t>algo_params


cdef build_ivfpq_algo_params(params, automated, additional_info):
    cdef IVFPQParam* algo_params = new IVFPQParam()
    if automated:
        allowedSubquantizers = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
        allowedSubDimSize = {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32}
        N = additional_info['n_samples']
        D = additional_info['n_features']

        params = {
            'nlist': 8,
            'nprobe': 3
        }

        for n_subq in allowedSubquantizers:
            if D % n_subq == 0 and (D / n_subq) in allowedSubDimSize:
                params['usePrecomputedTables'] = False
                params['M'] = n_subq
                break

        if 'M' not in params:
            for n_subq in allowedSubquantizers:
                if D % n_subq == 0:
                    params['usePrecomputedTables'] = True
                    params['M'] = n_subq
                    break

        # n_bits should be in set {4, 5, 6, 8} since FAISS 1.7
        params['n_bits'] = 4
        for n_bits in [5, 6, 8]:
            min_train_points = (2 ** n_bits) * 39
            if N >= min_train_points:
                params['n_bits'] = n_bits
                break

    algo_params.nlist = <int> params['nlist']
    algo_params.nprobe = <int> params['nprobe']
    algo_params.M = <int> params['M']
    algo_params.n_bits = <int> params['n_bits']
    algo_params.usePrecomputedTables = \
        <bool> params['usePrecomputedTables']
    return <uintptr_t>algo_params


cdef build_ivfsq_algo_params(params, automated):
    cdef IVFSQParam* algo_params = new IVFSQParam()
    if automated:
        params = {
            'nlist': 8,
            'nprobe': 2,
            'qtype': 'QT_8bit',
            'encodeResidual': True
        }

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


cdef build_algo_params(algo, params, additional_info):
    automated = params is None or params == 'auto'
    if not automated:
        check_algo_params(algo, params)

    cdef knnIndexParam* algo_params = <knnIndexParam*> 0
    if algo == 'ivfflat':
        algo_params = <knnIndexParam*><uintptr_t> \
            build_ivfflat_algo_params(params, automated)
    if algo == 'ivfpq':
        algo_params = <knnIndexParam*><uintptr_t> \
            build_ivfpq_algo_params(params, automated, additional_info)
    elif algo == 'ivfsq':
        algo_params = <knnIndexParam*><uintptr_t> \
            build_ivfsq_algo_params(params, automated)

    return <uintptr_t>algo_params


cdef destroy_algo_params(ptr):
    cdef knnIndexParam* algo_params = <knnIndexParam*> <uintptr_t> ptr
    del algo_params
