import numpy as np

def logsumexp(arr:np.ndarray):
    # normalized to avoid overflow issues (maybe run into underflow?)
    c = arr.max()
    lse = c + np.log(np.sum(np.exp(arr-c)))
    return lse

def softmax(arr:np.ndarray):
    # normalized to avoid overflow issues (maybe run into underflow?)
    return np.exp(arr-logsumexp(arr))