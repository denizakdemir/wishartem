import numpy as np
from typing import List, Optional, Tuple
from joblib import Parallel, delayed
import multiprocessing


class OptimizedEMCovarianceCombiner:
    """
    Optimized implementation of EM algorithm for combining partial covariance matrices.
    
    This class implements an optimized version of the EM algorithm for combining multiple
    partially observed covariance matrices into a single combined covariance matrix,
    with specific optimizations for handling large matrices:
    
    1. Replace matrix inversions with solve operations for better numerical stability
    2. Parallelize the E-step for faster computation with large matrices
    3. Optimize sampling covariance computation
    4. Memory optimization with reduced precision option
    5. Regularization for numerical stability
    
    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations for the EM algorithm.
    tol : float, default=1e-6
        Convergence tolerance for the EM algorithm.
    track_loglik : bool, default=True
        Whether to track the log-likelihood values during iterations.
    use_float32 : bool, default=False
        Whether to use single precision (float32) instead of double precision.
        This can significantly reduce memory usage for very large matrices.
    regularization : float, default=1e-8
        Small constant added to diagonal for regularization to improve stability.
    n_jobs : int, default=None
        Number of parallel jobs for E-step. If None, uses all available CPU cores.
        
    Attributes
    ----------
    max_iter : int
        Maximum number of iterations for the EM algorithm.
    tol : float
        Convergence tolerance for the EM algorithm.
    track_loglik : bool
        Whether to track the log-likelihood values during iterations.
    dtype : np.dtype
        Data type used for computations (float32 or float64).
    regularization : float
        Small constant added to diagonal for regularization.
    n_jobs : int
        Number of parallel jobs for E-step.
    loglik_path : list
        Stores the log-likelihood values at each iteration.
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6, track_loglik: bool = True,
                use_float32: bool = False, regularization: float = 1e-8, n_jobs: Optional[int] = None):
        self.max_iter = max_iter
        self.tol = tol
        self.track_loglik = track_loglik
        self.dtype = np.float32 if use_float32 else np.float64
        self.regularization = regularization
        self.n_jobs = n_jobs if n_jobs is not None else min(8, multiprocessing.cpu_count())
        self.loglik_path = []
        
    def _initialize_psi(self, partial_covs: List[np.ndarray], 
                       var_indices: List[List[int]], n_vars: int) -> np.ndarray:
        """
        Initialize the combined covariance matrix using available variances.
        
        Parameters
        ----------
        partial_covs : List[np.ndarray]
            List of partial covariance matrices.
        var_indices : List[List[int]]
            List of variable indices for each partial covariance matrix.
        n_vars : int
            Total number of variables.
            
        Returns
        -------
        np.ndarray
            Initialized covariance matrix.
        """
        psi = np.eye(n_vars, dtype=self.dtype)
        
        for i in range(n_vars):
            variances = []
            for cov, idx in zip(partial_covs, var_indices):
                try:
                    idx_list = list(idx)
                    if i in idx_list:
                        i_local = idx_list.index(i)
                        variances.append(cov[i_local, i_local])
                except Exception as e:
                    print(f"Error processing index {i}: {str(e)}")
                    continue
                    
            if variances:
                psi[i, i] = np.mean(variances)
                
        # Add small regularization to ensure positive definiteness
        psi += self.regularization * np.eye(n_vars, dtype=self.dtype)
        return psi
    
    def _compute_conditional_expectation(self, ya: np.ndarray, psi: np.ndarray,
                                        obs_idx: List[int], missing_idx: List[int]) -> np.ndarray:
        """
        Compute conditional expectation for the E-step of the EM algorithm.
        Optimized version using solve instead of explicit matrix inversion.
        
        Parameters
        ----------
        ya : np.ndarray
            Observed partial covariance matrix.
        psi : np.ndarray
            Current estimate of the combined covariance matrix.
        obs_idx : List[int]
            Indices of observed variables.
        missing_idx : List[int]
            Indices of missing variables.
            
        Returns
        -------
        np.ndarray
            Conditional expectation of the complete covariance matrix.
        """
        # If there are no missing indices, just expand ya to full dimension
        if not missing_idx:
            n_total = psi.shape[0]
            result = np.zeros((n_total, n_total), dtype=self.dtype)
            result[np.ix_(obs_idx, obs_idx)] = ya
            return result

        obs_idx = np.array(obs_idx)
        missing_idx = np.array(missing_idx)

        n_total = psi.shape[0]
        result = np.zeros((n_total, n_total), dtype=self.dtype)

        try:
            # Extract submatrices from psi
            psi_aa = psi[np.ix_(obs_idx, obs_idx)]
            psi_ab = psi[np.ix_(obs_idx, missing_idx)]
            psi_bb = psi[np.ix_(missing_idx, missing_idx)]

            # Optimize: Use solve instead of computing explicit inverse
            # b_matrix_T represents psi_aa^-1 @ psi_ab
            b_matrix_T = np.linalg.solve(psi_aa, psi_ab)
            b_matrix = b_matrix_T.T  # Transpose to get the right dimensions

            # Fill observed part
            result[np.ix_(obs_idx, obs_idx)] = ya

            # Fill cross-terms
            exp_yab = b_matrix @ ya
            result[np.ix_(missing_idx, obs_idx)] = exp_yab
            result[np.ix_(obs_idx, missing_idx)] = exp_yab.T

            # Fill missing part (corrected formula)
            result[np.ix_(missing_idx, missing_idx)] = (
                psi_bb - b_matrix @ psi_ab + b_matrix @ ya @ b_matrix.T
            )

            return result

        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error in conditional expectation: {str(e)}")
            
            # Try with regularization if failure occurs
            psi_aa_reg = psi_aa + self.regularization * np.eye(len(obs_idx), dtype=self.dtype)
            b_matrix_T = np.linalg.solve(psi_aa_reg, psi_ab)
            b_matrix = b_matrix_T.T
            
            result[np.ix_(obs_idx, obs_idx)] = ya
            exp_yab = b_matrix @ ya
            result[np.ix_(missing_idx, obs_idx)] = exp_yab
            result[np.ix_(obs_idx, missing_idx)] = exp_yab.T
            result[np.ix_(missing_idx, missing_idx)] = (
                psi_bb - b_matrix @ psi_ab + b_matrix @ ya @ b_matrix.T
            )
            
            return result

    def _compute_log_likelihood(self, psi: np.ndarray, 
                                partial_covs: List[np.ndarray], 
                                var_indices: List[List[int]],
                                degrees_freedom: List[float]) -> float:
        """
        Compute the observed data log-likelihood.
        Optimized version using solve instead of explicit matrix inversion.
        
        Parameters
        ----------
        psi : np.ndarray
            Current estimate of the combined covariance matrix.
        partial_covs : List[np.ndarray]
            List of partial covariance matrices.
        var_indices : List[List[int]]
            List of variable indices for each partial covariance matrix.
        degrees_freedom : List[float]
            Degrees of freedom for each partial covariance matrix.
            
        Returns
        -------
        float
            The observed data log-likelihood.
        """
        log_lik = 0.0
        
        for ya, idx, nu in zip(partial_covs, var_indices, degrees_freedom):
            # Extract the submatrix of psi corresponding to the observed variables
            psi_subset = psi[np.ix_(idx, idx)]
            
            # Compute log-likelihood contribution from this partial covariance matrix
            sign, logdet_psi = np.linalg.slogdet(psi_subset)
            if sign <= 0:
                # Handle non-positive definite matrix
                return -np.inf
                
            # Optimize: Use solve instead of computing explicit inverse for trace term
            try:
                # Solve psi_subset @ X = ya for X and compute trace(X)
                X = np.linalg.solve(psi_subset, ya)
                trace_term = np.trace(X)
                
                # Add to log-likelihood (scaled by degrees of freedom)
                log_lik += -0.5 * nu * (logdet_psi + trace_term)
            except np.linalg.LinAlgError:
                try:
                    # Try with regularization
                    psi_subset_reg = psi_subset + self.regularization * np.eye(len(idx), dtype=self.dtype)
                    X = np.linalg.solve(psi_subset_reg, ya)
                    trace_term = np.trace(X)
                    
                    # Recompute log determinant with regularization
                    sign_reg, logdet_psi_reg = np.linalg.slogdet(psi_subset_reg)
                    log_lik += -0.5 * nu * (logdet_psi_reg + trace_term)
                except:
                    # If still fails, return -inf
                    return -np.inf
                
        return log_lik
        
    def _compute_sampling_covariance(self, psi: np.ndarray, 
                                    partial_covs: List[np.ndarray],
                                    var_indices: List[List[int]],
                                    degrees_freedom: List[float]) -> np.ndarray:
        """
        Compute the sampling covariance matrix for the combined covariance matrix.
        Optimized version with vectorization and numerical stability improvements.
        
        Parameters
        ----------
        psi : np.ndarray
            Combined covariance matrix.
        partial_covs : List[np.ndarray]
            List of partial covariance matrices.
        var_indices : List[List[int]]
            List of variable indices for each partial covariance matrix.
        degrees_freedom : List[float]
            Degrees of freedom for each partial covariance matrix.
            
        Returns
        -------
        np.ndarray
            Sampling covariance matrix for the combined covariance matrix.
        """
        n_vars = psi.shape[0]
        n_params = n_vars * (n_vars + 1) // 2  # Number of unique elements
        
        # Pre-compute the inverse once with regularization
        try:
            psi_reg = psi + self.regularization * np.eye(n_vars, dtype=self.dtype)
            psi_inv = np.linalg.inv(psi_reg)
        except np.linalg.LinAlgError:
            # Increase regularization if still singular
            psi_reg = psi + 10 * self.regularization * np.eye(n_vars, dtype=self.dtype)
            psi_inv = np.linalg.inv(psi_reg)
        
        # Create vectorized indices for efficiency using NumPy's triu_indices
        row_indices, col_indices = np.triu_indices(n_vars)
        vec_indices = {(row_indices[i], col_indices[i]): i for i in range(len(row_indices))}
        
        # Initialize matrices
        information = np.zeros((n_params, n_params), dtype=self.dtype)
        score_cov = np.zeros((n_params, n_params), dtype=self.dtype)
        
        # Process each partial covariance matrix
        for ya, idx, nu in zip(partial_covs, var_indices, degrees_freedom):
            missing_idx = sorted([i for i in range(n_vars) if i not in idx])
            
            # Get conditional expectation
            exp_y = self._compute_conditional_expectation(ya, psi, idx, missing_idx)
            
            # Compute score contribution
            score = exp_y - psi
            
            # Vectorize the score (more efficient than double loops)
            score_vec = np.zeros(n_params, dtype=self.dtype)
            for (i, j), vec_idx in vec_indices.items():
                score_vec[vec_idx] = score[i, j]
            
            # Update score covariance
            score_cov += nu * np.outer(score_vec, score_vec)
            
            # Compute information matrix more efficiently
            for (i, j), idx1 in vec_indices.items():
                for (k, l), idx2 in vec_indices.items():
                    info_val = nu * 0.5 * ( # Using the 0.5 factor to account for symmetry
                        psi_inv[i, k] * psi_inv[j, l] +
                        psi_inv[i, l] * psi_inv[j, k]
                    )
                    information[idx1, idx2] += info_val
        
        # Use a more stable approach for matrix inversion with regularization
        try:
            # Add small regularization to improve stability
            epsilon = self.regularization * np.mean(np.diag(information))
            reg_information = information + epsilon * np.eye(n_params, dtype=self.dtype)
            
            # Compute sampling covariance using solve rather than explicit inverse
            # For X = info_inv @ score_cov, solve information @ X = score_cov
            X = np.linalg.solve(reg_information, score_cov)
            
            # For sampling_cov = X @ info_inv.T, solve information.T @ Y = X.T for Y.T
            sampling_cov = np.linalg.solve(reg_information.T, X.T).T
            
            return sampling_cov
        except np.linalg.LinAlgError as e:
            print("Error computing sampling covariance:", str(e))
            # Return matrix of NaN with appropriate shape 
            return np.full((n_params, n_params), np.nan, dtype=self.dtype)

    def _compute_expectation_parallel(self, ya: np.ndarray, idx: List[int], 
                                     nu: float, psi: np.ndarray, n_vars: int) -> Tuple[np.ndarray, float]:
        """
        Helper function for parallel E-step computation.
        
        Parameters
        ----------
        ya : np.ndarray
            Observed partial covariance matrix.
        idx : List[int]
            Indices of observed variables.
        nu : float
            Degrees of freedom.
        psi : np.ndarray
            Current estimate of the combined covariance matrix.
        n_vars : int
            Total number of variables.
            
        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple of conditional expectation and corresponding degrees of freedom.
        """
        missing_idx = sorted([i for i in range(n_vars) if i not in idx])
        try:
            exp_y = self._compute_conditional_expectation(ya, psi, idx, missing_idx)
            return (exp_y, nu)
        except Exception as e:
            print(f"Error in expectation computation: {str(e)}")
            raise
        
    def fit(self, partial_covs: List[np.ndarray], 
            var_indices: List[List[int]], 
            degrees_freedom: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[List[float]]]:
        """
        Fit the EM algorithm to combine partial covariance matrices.
        Optimized version with parallel E-step and numerical stability improvements.
        
        Parameters
        ----------
        partial_covs : List[np.ndarray]
            List of partial covariance matrices.
        var_indices : List[List[int]]
            List of variable indices for each partial covariance matrix.
        degrees_freedom : Optional[List[float]], default=None
            Degrees of freedom for each partial covariance matrix.
            If None, defaults to 100 for each matrix.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[List[float]]]
            Tuple containing:
            - The combined covariance matrix
            - Its sampling covariance
            - The log-likelihood path if track_loglik=True, otherwise None
        
        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if not partial_covs or not var_indices:
            raise ValueError("Empty input provided")
        if len(partial_covs) != len(var_indices):
            raise ValueError("Number of covariance matrices must match number of index sets")
            
        # Determine the total number of variables
        n_vars = max(max(idx) for idx in var_indices) + 1
        
        if degrees_freedom is None:
            degrees_freedom = [100] * len(partial_covs)
        
        if len(degrees_freedom) != len(partial_covs):
            raise ValueError("Length of degrees_freedom must match number of matrices")
            
        # Reset log-likelihood path
        if self.track_loglik:
            self.loglik_path = []
            
        # Convert to selected precision
        partial_covs = [cov.astype(self.dtype) for cov in partial_covs]
            
        # Initialize Psi
        psi = self._initialize_psi(partial_covs, var_indices, n_vars)
        
        # Compute initial log-likelihood if tracking
        if self.track_loglik:
            loglik = self._compute_log_likelihood(psi, partial_covs, var_indices, degrees_freedom)
            self.loglik_path.append(loglik)
        
        # EM iterations
        for iter in range(self.max_iter):
            psi_old = psi.copy()
            
            # E-step (parallel version)
            n_jobs = min(self.n_jobs, len(partial_covs))
            try:
                expectations = Parallel(n_jobs=n_jobs)(
                    delayed(self._compute_expectation_parallel)(ya, idx, nu, psi, n_vars)
                    for ya, idx, nu in zip(partial_covs, var_indices, degrees_freedom)
                )
            except Exception as e:
                print(f"Error in parallel E-step (iteration {iter}): {str(e)}")
                # Fall back to sequential processing if parallel fails
                expectations = []
                for ya, idx, nu in zip(partial_covs, var_indices, degrees_freedom):
                    missing_idx = sorted([i for i in range(n_vars) if i not in idx])
                    try:
                        exp_y = self._compute_conditional_expectation(ya, psi, idx, missing_idx)
                        expectations.append((exp_y, nu))
                    except Exception as e2:
                        print(f"Error in sequential E-step: {str(e2)}")
                        raise
            
            # M-step with regularization
            psi_sum = sum(nu * exp_y for exp_y, nu in expectations)
            total_nu = sum(degrees_freedom)
            psi = psi_sum / total_nu
            
            # Add small regularization if needed
            min_eig = np.min(np.linalg.eigvalsh(psi))
            if min_eig < self.regularization:
                epsilon = max(self.regularization - min_eig, 0) + self.regularization
                psi += epsilon * np.eye(n_vars, dtype=self.dtype)
            
            # Compute log-likelihood if tracking
            if self.track_loglik:
                loglik = self._compute_log_likelihood(psi, partial_covs, var_indices, degrees_freedom)
                self.loglik_path.append(loglik)
            
            # Check convergence
            if np.max(np.abs(psi - psi_old)) < self.tol:
                break
        
        # Compute sampling covariance
        try:
            sampling_cov = self._compute_sampling_covariance(psi, partial_covs, var_indices, degrees_freedom)
        except Exception as e:
            print(f"Error computing sampling covariance: {str(e)}")
            sampling_cov = np.full((n_vars * (n_vars + 1) // 2, n_vars * (n_vars + 1) // 2), np.nan, dtype=self.dtype)
                
        return psi, sampling_cov, self.loglik_path if self.track_loglik else None