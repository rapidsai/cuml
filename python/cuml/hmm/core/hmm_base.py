from abc import ABC
from cuml.gmm.sample_utils import *


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

    def __init__(self, precision, random_state, align_size=32):
        self.precision = precision
        self.dtype = self._get_dtype(precision)
        self.random_state = random_state

        self.align_size = align_size


class _DevHMM(ABC):
    def __init__(self):
        pass

    def _get_gamma(self):
        gamma = self.dGamma.copy_to_host()
        gamma = deallign(gamma, self.n_components, self.nObs, self.lddgamma)
        return gamma.T

    def _set_gamma(self, gamma):
        self.lddgamma = roundup(self.n_components, self.align_size)
        self.dGamma = process_parameter(gamma, self.lddgamma, self.dtype)

    def _get_B(self):
        B = self.dB.copy_to_host()
        B = deallign(B, self.n_components, self.nObs, self.lddgamma)
        return B

    def _set_B(self, B):
        self.lddb = roundup(self.n_components, self.align_size)
        self.dB = process_parameter(B, self.lddb, self.dtype)

    def _get_llhd(self):
        B = self.dLlhd.copy_to_host()
        B = B.flatten()
        return B

    def _set_llhd(self, llhd):
        self.dLlhd = process_parameter(llhd[:, None], self.nSeq, self.dtype)

    def _get_dVStates(self):
        dVStates = self.dVStates.copy_to_host()
        dVStates = dVStates.flatten()
        return dVStates

    def _set_dVStates(self, dVStates):
        self.dVStates = process_parameter(dVStates[:, None], self.nObs, self.int_type)

    def _get_logllhd(self):
        return self.dlogllhd.copy_to_host()[0]

    def _set_logllhd(self, logllhd):
        logllhd_array = np.array(logllhd, dtype=self.dtype)
        self.dlogllhd = cuda.to_device(logllhd_array)

    _gammas_ = property(_get_gamma, _set_gamma)
    _B_ = property(_get_B, _set_B)
    _llhd = property(_get_llhd, _set_llhd)
    _dVStates_ = property(_get_dVStates, _set_dVStates)
    _logllhd_ = property(_get_logllhd, _set_logllhd)


class _BaseHMM(_BaseCUML, _DevHMM):
    def __init__(self,
                 n_components,
                 precision,
                 random_state,
                 init_params,
                 n_iter,
                 ):

        _BaseCUML.__init__(self,
                           precision=precision,
                           random_state=random_state)
        self.init_params = init_params
        self.n_iter = n_iter

        self.n_components = n_components

        if self.n_components > 65535 :
            raise Exception('The number of states should not exceed 65,535. The value of n_components was: {}'.format(n_components))

        self.int_type = np.uint16

    def fit(self, X, lengths=None):
        self._set_dims(X, lengths)
        self._initialize()
        self._reset()

        for step in range(self.n_iter):
            # self._forward_backward(X, lengths, True, True, True)
            self._m_step(X, lengths)

    def decode(self, X, lengths=None, algorithm=None):
        self._set_dims(X, lengths)
        self._reset()

        self._viterbi(X, lengths)
        state_sequence = self._dVStates_
        llhd = self._llhd
        return llhd, state_sequence

    # @abstractmethod
    # def predict(self, X, lengths=None):
    #     pass

    def predict_proba(self, X, lengths=None):
        self._set_dims(X, lengths)
        self._initialize()
        self._reset()

        self._forward_backward(X, lengths, True, True, True)
        # self._forward_backward(X, lengths, False, False, False)
        return self._gammas_

    # @abstractmethod
    # def sample(self, n_samples=1, random_state=None):
    #     pass

    def score(self, X, lengths=None):
        self._reset()
        self._forward_backward(X, lengths, True, False, False)
        return self._score()

    def score_samples(self, X, lengths=None):
        posteriors = self.predict_proba(X, lengths)
        logprob = self._score()
        return logprob, posteriors

    def _get_transmat(self):
        T = self.dT.copy_to_host()
        T = deallign(T, self.n_components, self.n_components, self.lddt)
        return T

    def _set_transmat(self, transmat):
        self.lddt = roundup(self.n_components, self.align_size)
        self.dT = process_parameter(transmat, self.lddt, self.dtype)


    def _get_startprob_(self):
        startProb = self.dstartProb.copy_to_host()
        return startProb

    def _set_startprob_(self, startProb):
        self.lddsp = startProb.shape[0]
        self.dstartProb = process_parameter(startProb[:, None], self.lddsp, self.dtype)

    transmat_ = property(_get_transmat, _set_transmat)
    startprob_ = property(_get_startprob_, _set_startprob_)

    def _initialize(self):
        # Align flatten, cast and copy to device
        if 't' in self.init_params :
            init_value = 1 / self.n_components
            T = np.full((self.n_components, self.n_components), init_value)
            self._set_transmat(T)

        if 's' in self.init_params :
            init_value = 1 / self.n_components
            sp = np.full(self.n_components, init_value)
            self._set_startprob_(sp)

        for dist in self.dists:
            dist._initialize()

        self._set_logllhd(0)

    def _set_dims(self, X, lengths):
        self.nObs = X.shape[0]

        if lengths is None:
            self.nSeq = 1
        else:
            self.nSeq = lengths.shape[0]

        for dist in self.dists :
            dist._set_dims(X)

    def _setup(self, X, lengths):
        self.dX = X.T
        self.lddx = 1
        # Align flatten, cast and copy to device
        self.dX = process_parameter(self.dX, self.lddx, self.x_type)

        # Process lengths
        if lengths is None:
            lengths = np.array([self.nObs])
        # Check leading dimension
        lengths = lengths.astype(self.int_type)
        self.dlengths = cuda.to_device(lengths)

        for dist in self.dists :
            dist._setup(X)

    def _reset(self):
        B = sample_matrix(self.n_components,
                          self.nObs,
                          random_state=self.random_state,
                          isColNorm=True)
        self._set_B(B)

        Gamma = np.zeros((self.n_components, self.nObs), dtype=self.dtype)
        self._set_gamma(Gamma)

        dVStates = np.zeros(self.nObs, dtype=self.int_type)
        self._set_dVStates(dVStates)

        Llhd = np.zeros(self.nSeq, dtype=self.dtype)
        self._set_llhd(Llhd)

    def _score(self):
        return self._logllhd_