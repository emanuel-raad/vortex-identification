import numpy as np

def q_criterion(strain, vorticity):
    
    q     = np.zeros_like(strain[0][0])
    s_mag = np.zeros_like(strain[0][0])
    v_mag = np.zeros_like(strain[0][0])
    
    for i in range(0, 3):
        for j in range(0, 3):
            s_mag = s_mag + np.multiply(strain[i][j], strain[i][j])
            v_mag = v_mag + np.multiply(vorticity[i][j], vorticity[i][j])
    
    # s_mag = np.sqrt(s_mag)
    # v_mag = np.sqrt(v_mag)
    
    q = 0.5 * (v_mag - s_mag)
    
    return q