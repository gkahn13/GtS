import numpy as np

def weighted_sample(weights, objects):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    """
    # An array of the weights, cumulatively summed.
    cs = np.cumsum(weights)
    # Find the index of the first weight over a random value.
    idx = sum(cs < np.random.rand())
    return objects[min(idx, len(objects) - 1)]

def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret

def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret

def from_onehot(v):
    return np.nonzero(v)[0][0]

def from_onehot_n(v):
    if len(v) == 0:
        return []
    return np.nonzero(v)[1]
