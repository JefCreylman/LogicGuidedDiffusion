import torch
import numpy as np

def find_gamma(D, tolerance=1e-6, max_iter=100):
    '''
    Calculates gamma following NCSNv2
    Args:
        D (int):
            Dimensionality of the input data
        tolerance (float):
            Tolerance for error on C = 0.5, stops before max_iter if the error is lower than the tolerance
        max_iter (int):
            Maximum iterations for gamma calculation if tolerance is not reached

    Returns:
        gamma (float):
            The gamma that approximates C = 0.5 (see NCSNv2)
        c (Number):
            The actual C for the found gamma
    '''
    sqrt_2D = np.sqrt(2 * D)
    
    # Search boundaries for gamma
    low = 1.0001
    high = 2.0
    
    for _ in range(max_iter):
        gamma = (low + high) / 2.0
        
        x_plus = sqrt_2D * (gamma - 1) + 3 * gamma
        x_minus = sqrt_2D * (gamma - 1) - 3 * gamma
        
        #Phi is CDF of a standard Gaussian, calculation follows https://en.wikipedia.org/wiki/Error_function
        phi_plus = 0.5 * (1.0 + torch.erf(torch.tensor(x_plus) / np.sqrt(2.0)))
        phi_minus = 0.5 * (1.0 + torch.erf(torch.tensor(x_minus) / np.sqrt(2.0)))
        
        c = (phi_plus - phi_minus).item()
        
        if abs(c - 0.5) < tolerance:
            return gamma, c
        
        if c > 0.5:
            low = gamma
        else:
            high = gamma

    gamma = (low + high) / 2.0
        
    x_plus = sqrt_2D * (gamma - 1) + 3 * gamma
    x_minus = sqrt_2D * (gamma - 1) - 3 * gamma
    
    #Phi is CDF of a standard Gaussian, calculation follows https://en.wikipedia.org/wiki/Error_function
    phi_plus = 0.5 * (1.0 + torch.erf(torch.tensor(x_plus) / np.sqrt(2.0)))
    phi_minus = 0.5 * (1.0 + torch.erf(torch.tensor(x_minus) / np.sqrt(2.0)))
    
    c = (phi_plus - phi_minus).item()
    
    return gamma, c

def solve_num_timesteps(d, ls1, lsL):
    '''
    Solves the needed number of timesteps for given D, LOGSIGMA1, and LOGSIGMAL following NCSNv2
    Args:
        D (int):
            Dimensionality of the input data
        ls1 (float):
            Logarithm of the largest noise scale
        lsL (float):
            Logarithm of the smallest noise scale
    '''
    gamma, c = find_gamma(D=d)

    num_timesteps = int(np.ceil((ls1 - lsL)/np.log10(gamma)))
    sigmas = torch.logspace(start=ls1, end=lsL, steps=num_timesteps).tolist()

    print("gamma: ", gamma)
    print("C: ", c)
    print("num_timesteps: ", num_timesteps)
    print("sigmas: ", sigmas)

def solve_ld_eta():
    pass

if __name__ == "__main__":
    D = 56*56
    LOGSIGMA1 = 0
    LOGSIGMAL = -2.
    solve_num_timesteps(D, LOGSIGMA1, LOGSIGMAL)
    solve_ld_eta()