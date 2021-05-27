import numpy as np


 
def pca(dataset, num_pc = -1):
    
    # L 
    if(num_pc == -1):
        num_pc = min(dataset.shape)        
    
    # perform data centering
    pca_mean = [np.mean(dataset.T[i]) for i in range(dataset.shape[1])]
    pca_centered = np.array([dataset[i]-pca_mean for i in range(dataset.shape[0])])
    
    # perform svd
    u, s_values, vh = np.linalg.svd(pca_centered)
    
    # construct sigma matrix
    s_fullrank= np.diag(s_values)
    s = np.zeros((u.shape[1],vh.shape[0]))
    s[:s_fullrank.shape[0],:s_fullrank.shape[1]] = s_fullrank
    
    # truncate sigma according to L by setting 0
    s_truncated = s.copy()
    s_truncated[num_pc:,] = 0
      
    return u,s,vh,s_truncated

def pca_energy(dataset, num_pc = -1):
    
    if(num_pc == -1):
        num_pc = min(dataset.shape)  
        
    _, s, _, s_truncated, = pca(dataset, num_pc)
    
    # square of 1 to L ith singular values
    s_squared = np.square(s_truncated)
    # sum of square of all singular values
    trace = np.sum(np.square(s))
    # matrix of varinces
    energy_matrix = s_squared/trace
    # list of variances
    energy = energy_matrix.diagonal()   
    
    return energy





