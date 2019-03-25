from abc import ABC
from cuml.gmm.sample_utils import *

class _DevHMM(ABC):
    def __init__(self, precision):
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


    _gammas_ = property(_get_gamma, _set_gamma)
    _B_ = property(_get_B, _set_B)
    _llhd = property(_get_llhd, _set_llhd)