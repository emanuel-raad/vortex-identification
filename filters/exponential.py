import numpy as np

def exponentialSmoothing(arr, alpha):
    n, m, t = np.shape(arr)
    filtered_left = np.zeros_like(arr)
    filtered_right = np.zeros_like(arr)
    
    # forward pass
    for i in range(t):        
        if i == 0:
            filtered_right[:,:,i] = arr[:,:,i]
        else:
            filtered_right[:,:,i] = alpha * arr[:,:,i] + (1-alpha) * filtered_right[:,:,i-1]
    
    # backward pass
    for i in reversed(range(t)):        
        if i == t-1:
            filtered_left[:,:,i] = arr[:,:,i]
        else:
            filtered_left[:,:,i] = alpha * arr[:,:,i] + (1-alpha) * filtered_left[:,:,i+1]

    # return average of both passes
    return (filtered_right + filtered_left) / 2