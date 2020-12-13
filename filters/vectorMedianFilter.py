'''
VMF - Vector Median Filter
'''

import numpy as np
from scipy.ndimage import generic_filter
import math

def vectorMedianFilter(u, v):
    def _func(indices, u, v):
        center_idx = math.floor(len(indices)/2)
        center_value = indices[center_idx]

        indices = indices.astype(np.int64)
        # All boundary values are given an index of -1
        # Use a mask to remove them
        mask = indices >= 0
        indices = indices[mask]

        vecs = np.column_stack((u[indices], v[indices]))
        temp = np.zeros_like(indices, dtype=np.float64)

        for i in range(len(indices)):
            temp[i] = np.sum(np.linalg.norm(vecs[i] - vecs, axis=1))

        mins = np.where(temp == temp.min())
        mins = np.asarray(mins).ravel()

        if np.shape(mins)[0] > 1 and (center_value in indices[mins]):
            # Prefer returning the center index if there is more than one min
            return center_value
        else:
            # Otherwise just return the first minimum
            return indices[mins[0]]

    n, m = np.shape(u)
    idx = np.arange(n*m, dtype=np.int64).reshape((n, m))

    a = generic_filter(
        idx, _func, size=(3,3), mode='constant', cval=-1,
        extra_keywords={
            'u':u.ravel(),
            'v':v.ravel()
        }
    )
    
    a = a.ravel()
    u = u.ravel()[a].reshape((n, m))
    v = v.ravel()[a].reshape((n, m))
    
    return u, v