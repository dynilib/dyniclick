import numpy as np

def centroid(x):
    return np.sum(np.arange(len(x)) * x) / np.sum(x)
