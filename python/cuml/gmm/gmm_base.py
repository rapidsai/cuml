from abc import ABC, abstractmethod
from cuml.gmm.sample_utils import *

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


class _BaseGMM(_BaseCUML):
    def __init__(self,
                 n_components,
                 tol,
                 reg_covar,
                 max_iter,
                 init_params,
                 warm_start,
                 precision,
                 random_state):

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

    @abstractmethod
    def fit(self, X):
        pass

    @property
    def means_(self):
        mus = self.dParams["mus"].copy_to_host()
        mus = deallign(mus, self.nDim, self.nCl, self.ldd["mus"])
        return mus.T

    @property
    def covariances_(self):
        sigmas = self.dParams["sigmas"].copy_to_host()
        sigmas = deallign(sigmas, self.nDim, self.nCl * self.nDim,
                          self.ldd["sigmas"])
        sigmas = sigmas.reshape((self.nDim, self.nDim, self.nCl), order="F")
        return np.swapaxes(sigmas, 0, 2)

    @property
    def weights_(self):
        pis = self.dParams["pis"].copy_to_host()
        pis = deallign(pis, self.nCl, 1, self.ldd["pis"])
        pis = pis.flatten()
        return pis

    @property
    def lower_bound_(self):
        return self.cur_llhd.copy_to_host() / self.nObs

    @property
    def resp_(self):
        llhd = self.dParams["llhd"].copy_to_host()
        llhd = deallign(llhd, self.nCl, self.nObs, self.ldd["llhd"])
        llhd = llhd.T
        return llhd
