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

import functools
from importlib import import_module
import numpy as np

import cuml
from cuml.common.device_selection import DeviceType
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_host_array
from cuml.common.base import Base


def enable_cpu(gpu_func):
    @functools.wraps(gpu_func)
    def dispatch(self, *args, **kwargs):
        if isinstance(self, UniversalBase):
            func_name = gpu_func.func_name
            return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
        else:
            return gpu_func(self, *args, **kwargs)
    return dispatch


class UniversalBase(Base):
    """
    Experimental base class to implement CPU/GPU interoperability.
    """

    def dispatch_func(self, func_name, gpu_func, *args, **kwargs):
        """
        This function will dispatch calls to training and inference according
        to the global configuration. It should work for all estimators
        sufficiently close the scikit-learn implementation as it uses
        it for training and inferences on host.

        Parameters
        ----------
        func_name : string
            name of the function to be dispatched
        gpu_func : function
            original cuML function
        args : arguments
            arguments to be passed to the function for the call
        kwargs : keyword arguments
            keyword arguments to be passed to the function for the call
        """
        # look for current device_type
        device_type = cuml.global_settings.device_type
        if device_type == DeviceType.device:
            # call the original cuml method
            return gpu_func(self, *args, **kwargs)
        elif device_type == DeviceType.host:
            # check if the sklean model already set as attribute of the cuml
            # estimator its presence should signify that CPU execution was
            # used previously
            if not hasattr(self, 'sk_model_'):
                # import model in sklearn
                if hasattr(self, 'sk_import_path_'):
                    # if import path differs from the one of sklearn
                    # look for sk_import_path_
                    model_path = self.sk_import_path_
                else:
                    # import from similar path to the current estimator
                    # class
                    model_path = 'sklearn' + self.__class__.__module__[4:]
                model_name = self.__class__.__name__
                sk_model = getattr(import_module(model_path), model_name)
                # initialize model
                self.sk_model_ = sk_model()
                # transfer params set during cuml estimator initialization
                for param in self.get_param_names():
                    self.sk_model_.__dict__[param] = self.__dict__[param]

                # transfer attributes trained with cuml
                for attr in self.get_attributes_names():
                    # check presence of attribute
                    if hasattr(self, attr):
                        # get the cuml attribute
                        cu_attr = self.__dict__[attr]
                        # if the cuml attribute is a CumlArrayDescriptorMeta
                        if hasattr(cu_attr, 'get_input_value'):
                            # extract the actual value from the
                            # CumlArrayDescriptorMeta
                            cu_attr_value = cu_attr.get_input_value()
                            # check if descriptor is empty
                            if cu_attr_value is not None:
                                if cu_attr.input_type == 'cuml':
                                    # transform cumlArray to numpy and set it
                                    # as an attribute in the sklearn model
                                    self.sk_model_.__dict__[attr] = \
                                        cu_attr_value.to_output('numpy')
                                else:
                                    # transfer all other types of attributes
                                    # directly
                                    self.sk_model_.__dict__[attr] = \
                                        cu_attr_value
                        else:
                            # transfer all other types of attributes directly
                            self.sk_model_.__dict__[attr] = cu_attr
                    else:
                        raise ValueError('Attribute "{}" could not be found in'
                                         ' the cuML estimator'.format(attr))

            # converts all the args
            args = tuple(input_to_host_array(arg)[0] for arg in args)
            # converts all the kwarg
            for key, kwarg in kwargs.items():
                kwargs[key] = input_to_host_array(kwarg)[0]

            # call the method from the sklearn model
            sk_func = getattr(self.sk_model_, func_name)
            res = sk_func(*args, **kwargs)
            if func_name == 'fit':
                # need to do this to mirror input type
                self._set_output_type(args[0])
                # always return the cuml estimator while training
                # mirror sk attributes to cuml after training
                for attribute in self.get_attributes_names():
                    sk_attr = self.sk_model_.__dict__[attribute]
                    # if the sklearn attribute is an array
                    if isinstance(sk_attr, np.ndarray):
                        # transfer array to gpu and set it as a cuml
                        # attribute
                        cuml_array = input_to_cuml_array(sk_attr)[0]
                        setattr(self, attribute, cuml_array)
                    else:
                        # transfer all other types of attributes directly
                        setattr(self, attribute, sk_attr)
                return self
            else:
                # return method result
                return res
