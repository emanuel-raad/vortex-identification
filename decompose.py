import numpy as np

def velocity_decomposition(u, v, delX, delY):
        
    du = np.gradient(u, delX, delY)
    dv = np.gradient(v, delX, delY)
        
    du_dx = du[0]
    du_dy = du[1]
    
    dv_dx = dv[0]
    dv_dy = dv[1]
    
    zero  = np.zeros_like(du_dx)
        
    L = np.array([
        [du_dx, dv_dx, zero],
        [du_dy, dv_dy, zero],
        [ zero,  zero, zero],
    ])
        
    strain    = np.zeros_like(L)
    vorticity = np.zeros_like(L)
    
    for i in range(0, 3):
        for j in range(0, 3):
            strain[i][j]    = 0.5 * (L[i][j] + L[j][i])
            vorticity[i][j] = 0.5 * (L[i][j] - L[j][i])
        
    return L, strain, vorticity