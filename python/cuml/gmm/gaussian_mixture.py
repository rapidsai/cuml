#
# Copyright (c) 2019, NVIDIA CORPORATION.
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


from abc import ABC, abstractmethod
from cuml.gmm.utils.sample_utils import *
from cuml.gmm.utils.utils import *

from cuml.gmm.gaussian_mixture_backend import _GaussianMixtureBackend

RUP_SIZE = 32


class _BaseCUML(ABC):
    def _get_ctype_ptr(self, obj):
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_dtype(self, precision):
        return {
            'single': np.float32,
            'double': np.float64,
        }[precision]

    def __init__(self, precision):
        self.precision = precision
        self.dtype = self._get_dtype(precision)


class GaussianMixture(_BaseCUML, _GaussianMixtureBackend):
    def __init__(self, n_components, tol=1e-03,
                 reg_covar=1e-06, max_iter=100,
                 warm_start=False, precision='single',
                 random_state=None,
                 init_params="mcw"):
        super().__init__(precision=precision)

        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.random_state = random_state
        self.init_params = init_params

        self._isLog = True
        self._isInitialized = False
        self._isAllocated = False

    def _set_dims(self, X=None, nCl=None, nDim=None, nObs=None):
        if X is None :
            self.nCl = nCl
            self.nDim = nDim
            self.nObs = nObs
        else :
            self.nCl = self.n_components
            self.nDim = X.shape[1]
            self.nObs = X.shape[0]

    def _initialize(self):
        if "m" in self.init_params :
            mus = np.zeros((self.n_components, self.nDim))
            self._set_means(mus)
        if "w" in self.init_params :
            pis = np.zeros((self.n_components, 1))
            inv_pis = np.zeros((self.n_components, 1))
            self._set_weights(pis)
            self._set_inv_pis(inv_pis)
        if 'c' in self.init_params :
            covars = np.zeros((self.n_components, self.nDim, self.nDim))
            self._set_covars(covars)

        self.cur_llhd = cuda.to_device(np.zeros(1, dtype=self.dtype))

    def _setup(self, X):
        self.lddx = roundup(self.nDim, RUP_SIZE)
        self.dX = process_parameter(X.T, self.lddx, self.dtype)

        print("data size on device ", to_mb(self.dX.alloc_size), "Mb")

        Llhd = sample_matrix(self.nObs, self.nCl, self.random_state, isRowNorm=True)
        self.lddllhd = roundup(self.nCl, RUP_SIZE)
        self.dLlhd = Llhd.T
        self.dLlhd = process_parameter(self.dLlhd, self.lddllhd, self.dtype)
        print("resp size on device ", to_mb(self.dLlhd.alloc_size), "Mb")


    def fit(self, X):
        if self.warm_start :
            try:
                getattr(self, "nCl")
            except AttributeError:
                print("Please run the model a first time")
        else :
            self._set_dims(X)
            self._initialize()
        self._setup(X)
        self.allocate_ws()
        self.init_step()

        prev_lbow = - np.inf

        for it in range(1, self.max_iter + 1) :
            self.step()
            print("it", it, "lbow ",self.lower_bound_)

            diff = self.lower_bound_ - prev_lbow
            if  diff < self.tol :
                break
            prev_lbow = self.lower_bound_

    def _get_means(self):
        mus = self.dmus.copy_to_host()
        mus = deallign(mus, self.nDim, self.nCl, self.lddmus)
        return mus.T

    def _set_means(self, mus):
        self.nDim = mus.shape[1]
        self.lddmus = roundup(self.nDim, RUP_SIZE)
        self.dmus = process_parameter(mus.T, self.lddmus, self.dtype)

    def _get_covars(self):
        sigmas = self.dsigmas.copy_to_host()
        sigmas = deallign(sigmas, self.nDim, self.nCl * self.nDim,
                          self.lddsigmas)
        sigmas = sigmas.reshape((self.nDim, self.nDim, self.nCl), order="F")
        return np.swapaxes(sigmas, 0, 2)

    def _set_covars(self, covars):
        self.nDim = covars.shape[1]
        self.lddsigmas = roundup(self.nDim, RUP_SIZE)
        # TODO : Check if the reshape works correctly when the user inputs covars
        sigmas = np.swapaxes(covars, 0, 2)
        sigmas = np.reshape(sigmas, (self.nDim, self.n_components * self.nDim))
        # sigmas = sigmas.T
        self.dsigmas = process_parameter(sigmas, self.lddsigmas, self.dtype)

    def _get_weights(self):
        pis = self.dpis.copy_to_host()
        pis = deallign(pis, self.nCl, 1, self.lddpis)
        pis = pis.flatten()
        return pis

    def _set_weights(self, weights):
        self.lddpis = roundup(self.nCl, RUP_SIZE)
        self.dpis = process_parameter(weights, self.lddpis, self.dtype)

    def _set_inv_pis(self, inv_pis):
        self.lddinvpis = roundup(self.nCl, RUP_SIZE)
        self.dinv_pis = process_parameter(inv_pis, self.lddinvpis, self.dtype)

    def _get_resp(self):
        llhd = self.dLlhd.copy_to_host()
        llhd = deallign(llhd, self.nCl, self.nObs, self.lddllhd)
        llhd = llhd.T
        return llhd

    def _set_resp(self, llhd):
        self.lddllhd = roundup(self.nCl, RUP_SIZE)
        self.dLlhd = process_parameter(llhd, self.lddllhd, self.dtype)

    @property
    def lower_bound_(self):
        return self.cur_llhd.copy_to_host() / self.nObs

    means_ = property(_get_means, _set_means)
    covars_ = property(_get_covars, _set_covars)
    weights_ = property(_get_weights, _set_weights)
    resp_ = property(_get_resp, _set_resp)
