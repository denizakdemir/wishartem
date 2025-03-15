import numpy as np
from typing import List, Optional, Tuple

class EMCovarianceCombiner:
    """
    Implementation of EM algorithm for combining partial covariance matrices.
    
    This class implements the EM algorithm for combining multiple partially observed
    covariance matrices into a single combined covariance matrix, as described in
    the paper "NewWishartEM.pdf".
    
    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations for the EM algorithm.
    tol : float, default=1e-6
        Convergence tolerance for the EM algorithm.
    track_loglik : bool, default=True
        Whether to track the log-likelihood values during iterations.
        
    Attributes
    ----------
    max_iter : int
        Maximum number of iterations for the EM algorithm.
    tol : float
        Convergence tolerance for the EM algorithm.
    track_loglik : bool
        Whether to track the log-likelihood values during iterations.
    loglik_path : list
        Stores the log-likelihood values at each iteration.
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6, track_loglik: bool = True):
        self.max_iter = max_iter
        self.tol = tol
        self.track_loglik = track_loglik
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
        psi = np.eye(n_vars)
        
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
                
        return psi
    
    def _compute_conditional_expectation(self, ya: np.ndarray, psi: np.ndarray,
                                        obs_idx: List[int], missing_idx: List[int]) -> np.ndarray:
        """
        Compute conditional expectation for the E-step of the EM algorithm.
        
        This method computes the conditional expectation of the complete covariance matrix
        given the observed partial covariance matrix and current estimate of the combined 
        covariance matrix.
        
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
            result = np.zeros((n_total, n_total))
            result[np.ix_(obs_idx, obs_idx)] = ya
            return result

        obs_idx = np.array(obs_idx)
        missing_idx = np.array(missing_idx)

        n_total = psi.shape[0]
        result = np.zeros((n_total, n_total))

        try:
            # Extract submatrices from psi
            psi_aa = psi[np.ix_(obs_idx, obs_idx)]
            psi_ab = psi[np.ix_(obs_idx, missing_idx)]
            psi_bb = psi[np.ix_(missing_idx, missing_idx)]

            # Compute conditional expectations
            psi_aa_inv = np.linalg.inv(psi_aa)
            b_matrix = psi_ab.T @ psi_aa_inv

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
            print(f"Shapes - psi_aa: {psi_aa.shape}, psi_ab: {psi_ab.shape}, psi_bb: {psi_bb.shape}")
            print(f"ya shape: {ya.shape}, b_matrix shape: {b_matrix.shape if 'b_matrix' in locals() else 'not computed'}")
            raise

    def _compute_log_likelihood(self, psi: np.ndarray, 
                                partial_covs: List[np.ndarray], 
                                var_indices: List[List[int]],
                                degrees_freedom: List[float]) -> float:
        """
        Compute the observed data log-likelihood.
        
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
            # Using Wishart log-likelihood (ignoring constant terms)
            sign, logdet_psi = np.linalg.slogdet(psi_subset)
            if sign <= 0:
                # Handle non-positive definite matrix
                return -np.inf
                
            # Trace term (matrix product of inverse psi with observed covariance)
            try:
                psi_inv = np.linalg.inv(psi_subset)
                trace_term = np.trace(psi_inv @ ya)
                
                # Add to log-likelihood (scaled by degrees of freedom)
                log_lik += -0.5 * nu * (logdet_psi + trace_term)
            except np.linalg.LinAlgError:
                # Handle singular matrix
                return -np.inf
                
        return log_lik
        
    def _compute_sampling_covariance(self, psi: np.ndarray, 
                                    partial_covs: List[np.ndarray],
                                    var_indices: List[List[int]],
                                    degrees_freedom: List[float]) -> np.ndarray:
        """
        Compute the sampling covariance matrix for the combined covariance matrix.
        
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

        # Initialize matrices for sandwich formula
        information = np.zeros((n_params, n_params))
        score_cov = np.zeros((n_params, n_params))

        # Function to map matrix indices to vector index
        def matrix_to_vector_idx(i: int, j: int) -> int:
            if i < j:
                i, j = j, i  # Swap to ensure i >= j
            return i * (i + 1) // 2 + j

        # Compute observed information matrix and score covariance
        for ya, idx, nu in zip(partial_covs, var_indices, degrees_freedom):
            missing_idx = sorted([i for i in range(n_vars) if i not in idx])

            # Get conditional expectation
            exp_y = self._compute_conditional_expectation(ya, psi, idx, missing_idx)

            # Compute score contribution (no scaling)
            score = exp_y - psi  # No scaling here

            # Flatten the matrices
            score_vec = np.zeros(n_params)
            psi_inv = np.linalg.inv(psi)
            info_mat = np.zeros((n_params, n_params))

            # Map the score matrix to vector form
            for i in range(n_vars):
                for j in range(i + 1):
                    idx_vec = matrix_to_vector_idx(i, j)
                    score_vec[idx_vec] = score[i, j]

            # Update score covariance scaled by nu
            score_cov += nu * np.outer(score_vec, score_vec)

            # Compute the observed information matrix contribution
            for i in range(n_vars):
                for j in range(i + 1):
                    idx1 = matrix_to_vector_idx(i, j)
                    for k in range(n_vars):
                        for l in range(k + 1):
                            idx2 = matrix_to_vector_idx(k, l)
                            info_val = nu * (
                                psi_inv[i, k] * psi_inv[j, l] +
                                psi_inv[i, l] * psi_inv[j, k]
                            )
                            info_mat[idx1, idx2] += info_val
            information += info_mat

        # Compute sandwich covariance matrix
        try:
            info_inv = np.linalg.inv(information)
            sampling_cov = info_inv @ score_cov @ info_inv.T
            return sampling_cov
        except np.linalg.LinAlgError as e:
            print("Error computing sampling covariance:", str(e))
            raise

        
    def fit(self, partial_covs: List[np.ndarray], 
            var_indices: List[List[int]], 
            degrees_freedom: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[List[float]]]:
        """
        Fit the EM algorithm to combine partial covariance matrices.
        
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
            
        # Initialize Psi
        psi = self._initialize_psi(partial_covs, var_indices, n_vars)
        
        # Compute initial log-likelihood if tracking
        if self.track_loglik:
            loglik = self._compute_log_likelihood(psi, partial_covs, var_indices, degrees_freedom)
            self.loglik_path.append(loglik)
        
        # EM iterations
        for iter in range(self.max_iter):
            psi_old = psi.copy()
            
            # E-step
            expectations = []
            for ya, idx, nu in zip(partial_covs, var_indices, degrees_freedom):
                missing_idx = sorted([i for i in range(n_vars) if i not in idx])
                try:
                    exp_y = self._compute_conditional_expectation(ya, psi, idx, missing_idx)
                    expectations.append((exp_y, nu))
                except Exception as e:
                    print(f"Error in iteration {iter}: {str(e)}")
                    print(f"Matrix shapes - ya: {ya.shape}, psi: {psi.shape}")
                    print(f"Indices - observed: {idx}, missing: {missing_idx}")
                    raise
            
            # M-step
            psi_sum = sum(nu * exp_y for exp_y, nu in expectations)
            total_nu = sum(degrees_freedom)
            psi = psi_sum / total_nu
            
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
            sampling_cov = None
                
        return psi, sampling_cov, self.loglik_path if self.track_loglik else None