#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import typing

import numpy as np
import cupy as cp
import cuml
import cuml.common
import cuml.common.cuda
import cuml.internals.logger as logger
import cuml.internals
import pylibraft.common.handle
import cuml.internals.input_utils
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.input_utils import input_to_host_array
from cuml.internals.mem_type import MemoryType
from cuml.internals.array import CumlArray

from cuml.common.doc_utils import generate_docstring
from cuml.internals.mixins import TagsMixin
from cuml.common.device_selection import DeviceType
from cuml.common.base import Base as originalBase


class Base(originalBase):
    """
    Experimental base class to implement CPU/GPU interoperability.
    """

    def dispatch_func(self, func_name, *args, **kwargs):
        """
        This function will dispatch calls to training and inference according
        to the global configuration. It should work for all estimators
        sufficiently close the scikit-learn implementation as it uses
        it for training and inferences on host.

        Parameters
        ----------
        func_name : string
            name of the function to be dispatched
        args : arguments
            arguments to be passed to the function for the call
        kwargs : keyword arguments
            keyword arguments to be passed to the function for the call
        """
        # look for current device_type
        device_type = cuml.global_settings.device_type
        if device_type == DeviceType.device:
            # call the original cuml method
            cuml_func_name = '_' + func_name
            if hasattr(self, cuml_func_name):
                cuml_func = getattr(self, cuml_func_name)
                return cuml_func(*args, **kwargs)
            else:
                raise ValueError('Function "{}" could not be found in'
                                 ' the cuML estimator'.format(cuml_func_name))

        elif device_type == DeviceType.host:
            # check if the sklean model already set as attribute of the cuml
            # estimator its presence should signify that CPU execution was
            # used previously
            if not hasattr(self, '_cpu_model'):
                filtered_kwargs = {}
                for keyword, arg in self._full_kwargs.items():
                    if keyword in self._cpu_hyperparams:
                        filtered_kwargs[keyword] = arg
                    else:
                        logger.info("Unused keyword parameter: {} "
                                    "during CPU estimator "
                                    "initialization".format(keyword))

                # initialize model
                self._cpu_model = self._cpu_model_class(**filtered_kwargs)

                # transfer attributes trained with cuml
                for attr in self.get_attr_names():
                    # check presence of attribute
                    if hasattr(self, attr) or \
                       isinstance(getattr(type(self), attr, None), property):
                        # get the cuml attribute
                        if hasattr(self, attr):
                            cu_attr = getattr(self, attr)
                        else:
                            cu_attr = getattr(type(self), attr).fget(self)
                        # if the cuml attribute is a CumlArrayDescriptorMeta
                        if hasattr(cu_attr, 'get_input_value'):
                            # extract the actual value from the
                            # CumlArrayDescriptorMeta
                            cu_attr_value = cu_attr.get_input_value()
                            # check if descriptor is empty
                            if cu_attr_value is not None:
                                if cu_attr.input_type == 'cuml':
                                    # transform cumlArray to numpy and set it
                                    # as an attribute in the CPU estimator
                                    setattr(self._cpu_model, attr,
                                            cu_attr_value.to_output('numpy'))
                                else:
                                    # transfer all other types of attributes
                                    # directly
                                    setattr(self._cpu_model, attr,
                                            cu_attr_value)
                        elif isinstance(cu_attr, CumlArray):
                            # transform cumlArray to numpy and set it
                            # as an attribute in the CPU estimator
                            setattr(self._cpu_model, attr,
                                    cu_attr.to_output('numpy'))
                        elif isinstance(cu_attr, cp.ndarray):
                            # transform cupy to numpy and set it
                            # as an attribute in the CPU estimator
                            setattr(self._cpu_model, attr,
                                    cp.asnumpy(cu_attr))
                        else:
                            # transfer all other types of attributes directly
                            setattr(self._cpu_model, attr, cu_attr)

            # converts all the args
            args = tuple(input_to_host_array(arg)[0] for arg in args)
            # converts all the kwarg
            for key, kwarg in kwargs.items():
                kwargs[key] = input_to_host_array(kwarg)[0]

            # call the method from the sklearn model
            cpu_func = getattr(self._cpu_model, func_name)
            res = cpu_func(*args, **kwargs)

            if func_name in ['fit', 'fit_transform', 'fit_predict']:
                # need to do this to mirror input type
                self._set_output_type(args[0])
                self._set_output_mem_type(args[0])
                # always return the cuml estimator while training
                # mirror sk attributes to cuml after training
                for attr in self.get_attr_names():
                    # check presence of attribute
                    if hasattr(self._cpu_model, attr) or \
                       isinstance(getattr(type(self._cpu_model),
                                          attr, None), property):
                        # get the cpu attribute
                        if hasattr(self._cpu_model, attr):
                            cpu_attr = getattr(self._cpu_model, attr)
                        else:
                            cpu_attr = getattr(type(self._cpu_model),
                                               attr).fget(self._cpu_model)
                        # if the cpu attribute is an array
                        if isinstance(cpu_attr, np.ndarray):
                            # get data order wished for by CumlArrayDescriptor
                            if hasattr(self, attr + '_order'):
                                order = getattr(self, attr + '_order')
                            else:
                                order = 'K'
                            # transfer array to gpu and set it as a cuml
                            # attribute
                            cuml_array = input_to_cuml_array(cpu_attr,
                                                             order=order)[0]
                            setattr(self, attr, cuml_array)
                        else:
                            # transfer all other types of attributes directly
                            setattr(self, attr, cpu_attr)
                if func_name == 'fit':
                    return self
            # return method result
            return res

    def fit(self, *args, **kwargs):
        return self.dispatch_func('fit', *args, **kwargs)

    def predict(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('predict', *args, **kwargs)

    def transform(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('transform', *args, **kwargs)

    def kneighbors(self, X, *args, **kwargs) \
            -> typing.Union[CumlArray, typing.Tuple[CumlArray, CumlArray]]:
        return self.dispatch_func('kneighbors', X, *args, **kwargs)

    def fit_transform(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('fit_transform', *args, **kwargs)

    def fit_predict(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('fit_predict', *args, **kwargs)

    def inverse_transform(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('inverse_transform', *args, **kwargs)

    def score(self, *args, **kwargs):
        return self.dispatch_func('score', *args, **kwargs)

    def decision_function(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('decision_function', *args, **kwargs)

    def predict_proba(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('predict_proba', *args, **kwargs)

    def predict_log_proba(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('predict_log_proba', *args, **kwargs)
