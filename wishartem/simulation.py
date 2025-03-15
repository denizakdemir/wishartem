import numpy as np
from typing import List, Tuple, Optional
from scipy import stats
from wishartem.em_covariance_combiner import EMCovarianceCombiner

def generate_triangle_data(n_samples: int, direct_corr: float, 
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate three variables with a triangular correlation structure.
    
    This function generates data from a multivariate normal distribution with 
    a specific correlation structure: X₁ -- X₂ -- X₃, where X₂ is correlated with
    both X₁ and X₃, but X₁ and X₃ are not directly correlated.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    direct_corr : float
        Direct correlation coefficient between adjacent variables.
    seed : Optional[int], default=None
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the three variables as separate arrays.
        
    Raises
    ------
    ValueError
        If the generated correlation matrix is not positive definite.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Create correlation matrix
    corr_matrix = np.array([
        [1.0, direct_corr, direct_corr**2],
        [direct_corr, 1.0, direct_corr],
        [direct_corr**2, direct_corr, 1.0]
    ])
    
    # Validate correlation matrix is positive definite
    eigenvals = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvals <= 0):
        raise ValueError(f"Generated correlation matrix is not positive definite. Eigenvalues: {eigenvals}")
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(mean=[0, 0, 0], cov=corr_matrix, size=n_samples)
    
    return data[:, 0], data[:, 1], data[:, 2]

def create_partial_datasets(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray,
                          n_samples: List[int]) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Create partial datasets with selected variables.
    
    This function creates partial covariance matrices from the generated data,
    where variables X₁ and X₃ are never directly observed together.
    
    Parameters
    ----------
    x1 : np.ndarray
        First variable.
    x2 : np.ndarray
        Second variable.
    x3 : np.ndarray
        Third variable.
    n_samples : List[int]
        List of sample sizes for each partial dataset.
        
    Returns
    -------
    Tuple[List[np.ndarray], List[List[int]]]
        Tuple containing the list of partial covariance matrices and their corresponding variable indices.
    """
    # Create partial covariance matrices
    cov1 = np.cov(np.vstack([x1[:n_samples[0]], x2[:n_samples[0]]]))
    cov2 = np.cov(np.vstack([x2[n_samples[0]:n_samples[0]+n_samples[1]], 
                            x3[n_samples[0]:n_samples[0]+n_samples[1]]]))
    
    return [cov1, cov2], [[0, 1], [1, 2]]

def matrix_to_vector_idx(i: int, j: int, n_vars: int) -> int:
    """
    Convert matrix indices to vectorized index for sampling covariance.
    
    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    n_vars : int
        Number of variables.
        
    Returns
    -------
    int
        Vectorized index.
    """
    if i > j:
        i, j = j, i
    return i * n_vars - (i * (i + 1)) // 2 + j

def run_single_simulation(direct_corr: float, n_samples: List[int], 
                         seed: Optional[int] = None, verbose: bool = False) -> Tuple[float, float, float]:
    """
    Run a single simulation with given correlation and sample sizes.
    
    Parameters
    ----------
    direct_corr : float
        Direct correlation coefficient between adjacent variables.
    n_samples : List[int]
        List of sample sizes for each partial dataset.
    seed : Optional[int], default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print detailed information.
        
    Returns
    -------
    Tuple[float, float, float]
        Tuple containing the estimated correlation, standard error, and p-value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    total_samples = sum(n_samples)
    x1, x2, x3 = generate_triangle_data(total_samples, direct_corr, seed)
    
    # Create partial datasets
    partial_covs, var_indices = create_partial_datasets(x1, x2, x3, n_samples)
    
    # Fit EM algorithm
    combiner = EMCovarianceCombiner(max_iter=100, tol=1e-6)
    combined_cov, sampling_cov = combiner.fit(
        partial_covs, 
        var_indices,
        degrees_freedom=n_samples
    )
    
    if verbose:
        print("\nCombined covariance matrix:")
        print(combined_cov)
        print("\nSampling covariance matrix:")
        print(sampling_cov)
    
    # Extract correlation
    var1 = combined_cov[0, 0]
    var3 = combined_cov[2, 2]
    cov13 = combined_cov[0, 2]
    est_corr = cov13 / np.sqrt(var1 * var3)
    
    # Get correct index for cov13 in sampling covariance
    n_vars = combined_cov.shape[0]
    idx13 = matrix_to_vector_idx(0, 2, n_vars)
    
    # Calculate standard error for correlation using delta method
    var_cov13 = sampling_cov[idx13, idx13]
    se = np.sqrt(var_cov13) / np.sqrt(var1 * var3)
    
    # Compute z-statistic and p-value
    z_stat = est_corr / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return est_corr, se, p_value

def run_power_analysis(direct_corr: float, n_samples: List[int], 
                      n_sims: int = 1000, alpha: float = 0.05) -> dict:
    """
    Run multiple simulations to estimate power or Type I error rate.
    
    Parameters
    ----------
    direct_corr : float
        Direct correlation coefficient between adjacent variables.
    n_samples : List[int]
        List of sample sizes for each partial dataset.
    n_sims : int, default=1000
        Number of simulations to run.
    alpha : float, default=0.05
        Significance level.
        
    Returns
    -------
    dict
        Dictionary containing the simulation results, including power, mean estimated correlation,
        standard deviation of estimated correlation, and mean standard error.
    """
    results = {
        'estimated_corrs': [],
        'standard_errors': [],
        'p_values': [],
        'rejected_null': 0
    }
    
    # Run first simulation with verbose output
    est_corr, se, p_value = run_single_simulation(direct_corr, n_samples, seed=0, verbose=True)
    results['estimated_corrs'].append(est_corr)
    results['standard_errors'].append(se)
    results['p_values'].append(p_value)
    results['rejected_null'] += int(p_value < alpha)
    
    # Run remaining simulations
    for i in range(1, n_sims):
        est_corr, se, p_value = run_single_simulation(direct_corr, n_samples, seed=i)
        results['estimated_corrs'].append(est_corr)
        results['standard_errors'].append(se)
        results['p_values'].append(p_value)
        results['rejected_null'] += int(p_value < alpha)
    
    results['power'] = results['rejected_null'] / n_sims
    results['mean_est_corr'] = np.mean(results['estimated_corrs'])
    results['sd_est_corr'] = np.std(results['estimated_corrs'])
    results['mean_se'] = np.mean(results['standard_errors'])
    
    return results