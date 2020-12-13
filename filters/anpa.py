'''
ANPA
'''

from scipy.ndimage import generic_filter
import math
import numpy as np

def anpaFilterSpatial(u, v, u_vmf, v_vmf):
    '''
    Run vector median filter first
    Then pass the outputs to ANPA filter
    '''
    def _func(indices, u, v, u_vmf, v_vmf, u_out, v_out):    
        
        center_idx = math.floor(len(indices)/2)
        center_value = int(indices[center_idx])
        
        yp = np.column_stack((u[center_value], v[center_value]))[0]
        indices = indices.astype(np.int64)
                
        # All boundary values are given an index of -1
        # Use a mask to remove them
        mask = indices >= 0
        indices = indices[mask]

        y = np.column_stack((    u[indices],     v[indices]))
        x = np.column_stack((u_vmf[indices], v_vmf[indices]))
                
        n = len(indices)
        
        weights = np.zeros_like(indices, dtype=np.float64)
        
        for l in range(n):
            hl = np.power(n, -0.33/2.0) * (np.sum(np.linalg.norm(y - y[l], axis=1)) + 1e-10)
            z = (yp - y[l]) / hl
            k = np.exp(-0.5 * np.dot(z.T, z)) 
            
            
            weights[l] = np.power(hl, -2.0) * k
        
        u_out[center_value] = np.dot(x[:,0], weights) / np.sum(weights)
        v_out[center_value] = np.dot(x[:,1], weights) / np.sum(weights)

        return 0

    n, m = np.shape(u)

    u_out = np.zeros_like(u, dtype=np.float64)
    v_out = np.zeros_like(v, dtype=np.float64)
        
    idx  = np.arange(n*m, dtype=np.int64).reshape((n, m))

    a = generic_filter(
        idx, _func, size=(3, 3), mode='constant', cval=-1,
        extra_keywords={
            'u':u.ravel(),
            'v':v.ravel(),
            'u_vmf':u_vmf.ravel(),
            'v_vmf':v_vmf.ravel(),
            'u_out':u_out.ravel(),
            'v_out':v_out.ravel(),
        }
    )

    return u_out.reshape((n, m)), v_out.reshape((n, m))

def anpaFilterTemporal(t, u, v, u_vmf, v_vmf, halfWidth = 4):
    '''
    Run vector median filter first
    Then pass the outputs to ANPA filter
    '''
    def _func(u, v, u_vmf, v_vmf, yp):
        
        n = len(u)        
        yp = np.reshape(yp, (2,))
        
        x = np.column_stack((u_vmf, v_vmf))
        y = np.column_stack((    u,     v))
                
        weights = np.zeros(n, dtype=np.float64)
        
        for l in range(n):
            hl = np.power(n, -0.33/2.0) * (np.sum(np.linalg.norm(y - y[l], axis=1)) + 1e-10)
            z = (yp - y[l]) / hl
            k = np.exp(-0.5 * np.dot(z.T, z))
            
            weights[l] = np.power(hl, -2.0) * k
                    
        u_out = np.dot(x[:,0], weights) / np.sum(weights)
        v_out = np.dot(x[:,1], weights) / np.sum(weights)
        
        half = int(len(u_vmf) / 2)
        
        if np.isnan(u_out):
            u_out = u_vmf[half]
        if np.isnan(v_out):
            v_out = v_vmf[half]
        
        return u_out, v_out
    
    n, m, z = np.shape(u)
        
    u_out = np.zeros((n, m), dtype=np.float64)
    v_out = np.zeros((n, m), dtype=np.float64)
        
    for i in range(n):
        for j in range(m):
            
            back = halfWidth
            if t - back < 0:
                back = t
            
            front = halfWidth + 1
            if t + front > z:
                front = z - t
            
            yp = np.array([ u[i,j,t], v[i,j,t] ])
            
            u_out[i, j], v_out[i, j] = _func(
                u[i,j,t-back:t+front],
                v[i,j,t-back:t+front],
                u_vmf[i,j,t-back:t+front],
                v_vmf[i,j,t-back:t+front],
                yp
            )
                                    
    return u_out, v_out