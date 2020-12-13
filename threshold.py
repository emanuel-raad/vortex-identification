from numba import stencil
import numpy as np

@stencil(neighborhood=((-1, 1), (-1, 1)), )
def _localDiffStencil(arr):
    return np.nanmin(arr[-1:2, -1:2] - arr[0, 0])

def localDiffStencil(arr):
    return _localDiffStencil(np.pad(arr, 1, mode='constant', constant_values=np.nan))[1:-1,1:-1]