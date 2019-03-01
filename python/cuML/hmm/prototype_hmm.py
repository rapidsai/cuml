import numpy as np
import cudf

from hmm import GaussianHMM

data = cudf.read_csv('data/data.csv')

hmm = GaussianHMM()
hmm.fit(data)