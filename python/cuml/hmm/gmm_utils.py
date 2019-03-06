import numpy as np

def roundup(x, ref):
    return  (int) (ref * np.ceil(x / ref))