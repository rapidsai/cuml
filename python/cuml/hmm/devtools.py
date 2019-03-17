from abc import ABC, abstractmethod
from cuml.hmm.sample_utils import *

class _DevHMM(ABC):
    def __init__(self, precision):
        pass

    @property
    def emissions_(self):
        B = self.dB.copy_to_host()
        B = deallign(B, self.nObs, self.n_mix, self.lddb)
        B = B.T
        return B



