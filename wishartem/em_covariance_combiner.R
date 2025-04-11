#' @title EM Covariance Combiner
#' @description Implementation of EM algorithm for combining partial covariance matrices.
#' This function implements the EM algorithm for combining multiple partially observed
#' covariance matrices into a single combined covariance matrix.
#' 
#' @param partial_covs List of partial covariance matrices
#' @param var_indices List of variable indices for each partial covariance matrix
#' @param degrees_freedom Optional numeric vector of degrees of freedom for each matrix (default: 100 for each)
#' @param max_iter Maximum number of iterations for the EM algorithm (default: 100)
#' @param tol Convergence tolerance for the EM algorithm (default: 1e-6)
#' @param track_loglik Whether to track log-likelihood values during iterations (default: TRUE)
#' 
#' @return List containing:
#'   - psi: The combined covariance matrix
#'   - sampling_cov: Sampling covariance matrix for the combined covariance matrix
#'   - loglik_path: Vector of log-likelihood values at each iteration (if track_loglik=TRUE)
#' 
#' @importFrom Matrix solve
#' @importFrom stats cov
#' @export
EMCovarianceCombiner <- function(partial_covs, var_indices, degrees_freedom = NULL,
                                max_iter = 100, tol = 1e-6, track_loglik = TRUE) {
  
  # Validate inputs
  if (length(partial_covs) == 0 || length(var_indices) == 0) {
    stop("Empty input provided")
  }
  if (length(partial_covs) != length(var_indices)) {
    stop("Number of covariance matrices must match number of index sets")
  }
  
  # Determine total number of variables
  n_vars <- max(sapply(var_indices, function(idx) max(idx))) + 1
  
  # Set default degrees of freedom if not provided
  if (is.null(degrees_freedom)) {
    degrees_freedom <- rep(100, length(partial_covs))
  }
  if (length(degrees_freedom) != length(partial_covs)) {
    stop("Length of degrees_freedom must match number of matrices")
  }
  
  # Initialize log-likelihood path
  loglik_path <- if (track_loglik) numeric(0) else NULL
  
  # Initialize Psi (combined covariance matrix)
  psi <- initialize_psi(partial_covs, var_indices, n_vars)
  
  # Compute initial log-likelihood if tracking
  if (track_loglik) {
    loglik <- compute_log_likelihood(psi, partial_covs, var_indices, degrees_freedom)
    loglik_path <- c(loglik_path, loglik)
  }
  
  # EM iterations
  for (iter in 1:max_iter) {
    psi_old <- psi
    
    # E-step: Compute conditional expectations
    expectations <- vector("list", length(partial_covs))
    for (i in seq_along(partial_covs)) {
      ya <- partial_covs[[i]]
      idx <- var_indices[[i]]
      nu <- degrees_freedom[i]
      missing_idx <- setdiff(0:(n_vars-1), idx)
      
      # Calculate conditional expectation
      exp_y <- compute_conditional_expectation(ya, psi, idx, missing_idx)
      expectations[[i]] <- list(exp_y = exp_y, nu = nu)
    }
    
    # M-step: Update Psi
    psi_sum <- matrix(0, nrow = n_vars, ncol = n_vars)
    total_nu <- sum(degrees_freedom)
    
    for (exp in expectations) {
      psi_sum <- psi_sum + exp$nu * exp$exp_y
    }
    psi <- psi_sum / total_nu
    
    # Compute log-likelihood if tracking
    if (track_loglik) {
      loglik <- compute_log_likelihood(psi, partial_covs, var_indices, degrees_freedom)
      loglik_path <- c(loglik_path, loglik)
    }
    
    # Check convergence
    if (max(abs(psi - psi_old)) < tol) {
      break
    }
  }
  
  # Compute sampling covariance
  sampling_cov <- tryCatch({
    compute_sampling_covariance(psi, partial_covs, var_indices, degrees_freedom)
  }, error = function(e) {
    warning("Error computing sampling covariance: ", e$message)
    return(NULL)
  })
  
  # Return results
  return(list(
    psi = psi,
    sampling_cov = sampling_cov,
    loglik_path = loglik_path
  ))
}

#' Initialize the combined covariance matrix using available variances
#' 
#' @param partial_covs List of partial covariance matrices
#' @param var_indices List of variable indices for each partial covariance matrix
#' @param n_vars Total number of variables
#' @return Initialized covariance matrix
initialize_psi <- function(partial_covs, var_indices, n_vars) {
  psi <- diag(n_vars)
  
  for (i in 0:(n_vars-1)) {
    variances <- numeric(0)
    
    for (j in seq_along(partial_covs)) {
      cov <- partial_covs[[j]]
      idx <- var_indices[[j]]
      
      if (i %in% idx) {
        i_local <- match(i, idx) - 1  # Convert to 0-based index
        variances <- c(variances, cov[i_local + 1, i_local + 1])  # Convert back to 1-based for R
      }
    }
    
    if (length(variances) > 0) {
      psi[i + 1, i + 1] <- mean(variances)  # 1-based indexing in R
    }
  }
  
  return(psi)
}

#' Compute conditional expectation for the E-step of the EM algorithm
#' 
#' @param ya Observed partial covariance matrix
#' @param psi Current estimate of the combined covariance matrix
#' @param obs_idx Indices of observed variables (0-based)
#' @param missing_idx Indices of missing variables (0-based)
#' @return Conditional expectation of the complete covariance matrix
compute_conditional_expectation <- function(ya, psi, obs_idx, missing_idx) {
  # Convert to 1-based indices for R
  obs_idx_r <- obs_idx + 1
  missing_idx_r <- missing_idx + 1
  
  n_total <- nrow(psi)
  result <- matrix(0, nrow = n_total, ncol = n_total)
  
  # If there are no missing indices, just expand ya to full dimension
  if (length(missing_idx) == 0) {
    result[obs_idx_r, obs_idx_r] <- ya
    return(result)
  }
  
  # Extract submatrices from psi
  psi_aa <- psi[obs_idx_r, obs_idx_r, drop = FALSE]
  psi_ab <- psi[obs_idx_r, missing_idx_r, drop = FALSE]
  psi_bb <- psi[missing_idx_r, missing_idx_r, drop = FALSE]
  
  # Compute conditional expectations efficiently
  # Use Matrix::solve for potential speedup with sparse matrices
  psi_aa_inv <- solve(psi_aa)
  b_matrix <- t(psi_ab) %*% psi_aa_inv
  
  # Fill observed part
  result[obs_idx_r, obs_idx_r] <- ya
  
  # Fill cross-terms
  exp_yab <- b_matrix %*% ya
  result[missing_idx_r, obs_idx_r] <- exp_yab
  result[obs_idx_r, missing_idx_r] <- t(exp_yab)
  
  # Fill missing part
  result[missing_idx_r, missing_idx_r] <- psi_bb - b_matrix %*% psi_ab + b_matrix %*% ya %*% t(b_matrix)
  
  return(result)
}

#' Compute the observed data log-likelihood
#' 
#' @param psi Current estimate of the combined covariance matrix
#' @param partial_covs List of partial covariance matrices
#' @param var_indices List of variable indices for each partial covariance matrix
#' @param degrees_freedom Degrees of freedom for each partial covariance matrix
#' @return The observed data log-likelihood
compute_log_likelihood <- function(psi, partial_covs, var_indices, degrees_freedom) {
  log_lik <- 0.0
  
  for (i in seq_along(partial_covs)) {
    ya <- partial_covs[[i]]
    idx <- var_indices[[i]]
    nu <- degrees_freedom[i]
    
    # Convert to 1-based indices for R
    idx_r <- idx + 1
    
    # Extract the submatrix of psi corresponding to the observed variables
    psi_subset <- psi[idx_r, idx_r, drop = FALSE]
    
    # Check if matrix is positive definite
    eig <- try(eigen(psi_subset, symmetric = TRUE, only.values = TRUE)$values, silent = TRUE)
    if (inherits(eig, "try-error") || any(eig <= 0)) {
      return(-Inf)  # Non-positive definite matrix
    }
    
    # Compute log-likelihood contribution (Wishart log-likelihood, ignoring constant terms)
    logdet_psi <- sum(log(eig))  # More efficient than determinant()
    
    # Trace term using matrix inverse
    psi_inv <- solve(psi_subset)
    trace_term <- sum(diag(psi_inv %*% ya))
    
    # Add to log-likelihood (scaled by degrees of freedom)
    log_lik <- log_lik - 0.5 * nu * (logdet_psi + trace_term)
  }
  
  return(log_lik)
}

#' Map matrix indices to vector index for covariance parameters
#' 
#' @param i Row index (0-based)
#' @param j Column index (0-based)
#' @return Vector index
matrix_to_vector_idx <- function(i, j) {
  if (i < j) {
    # Swap to ensure i >= j
    tmp <- i
    i <- j
    j <- tmp
  }
  return(i * (i + 1) / 2 + j)
}

#' Compute the sampling covariance matrix for the combined covariance matrix
#' 
#' @param psi Combined covariance matrix
#' @param partial_covs List of partial covariance matrices
#' @param var_indices List of variable indices for each partial covariance matrix
#' @param degrees_freedom Degrees of freedom for each partial covariance matrix
#' @return Sampling covariance matrix for the combined covariance matrix
compute_sampling_covariance <- function(psi, partial_covs, var_indices, degrees_freedom) {
  n_vars <- nrow(psi)
  n_params <- n_vars * (n_vars + 1) / 2  # Number of unique elements
  
  # Initialize matrices for sandwich formula
  information <- matrix(0, nrow = n_params, ncol = n_params)
  score_cov <- matrix(0, nrow = n_params, ncol = n_params)
  
  # Compute the inverse of psi once
  psi_inv <- solve(psi)
  
  # Convert indices to 0-based for consistency with the matrix_to_vector_idx function
  for (i in seq_along(partial_covs)) {
    ya <- partial_covs[[i]]
    idx <- var_indices[[i]]  # 0-based
    nu <- degrees_freedom[i]
    
    missing_idx <- setdiff(0:(n_vars-1), idx)
    
    # Convert to 1-based for R computation
    idx_r <- idx + 1
    missing_idx_r <- missing_idx + 1
    
    # Get conditional expectation
    exp_y <- compute_conditional_expectation(ya, psi, idx, missing_idx)
    
    # Compute score contribution
    score <- exp_y - psi
    
    # Flatten the matrices efficiently
    score_vec <- numeric(n_params)
    
    # Map the score matrix to vector form
    for (i_idx in 0:(n_vars-1)) {
      for (j_idx in 0:i_idx) {
        idx_vec <- matrix_to_vector_idx(i_idx, j_idx) + 1  # +1 for R indexing
        score_vec[idx_vec] <- score[i_idx + 1, j_idx + 1]  # +1 for R indexing
      }
    }
    
    # Update score covariance scaled by nu
    score_cov <- score_cov + nu * tcrossprod(score_vec)
    
    # Compute the observed information matrix contribution - use efficient vectorization
    for (i_idx in 0:(n_vars-1)) {
      for (j_idx in 0:i_idx) {
        idx1 <- matrix_to_vector_idx(i_idx, j_idx) + 1  # +1 for R indexing
        
        for (k_idx in 0:(n_vars-1)) {
          for (l_idx in 0:k_idx) {
            idx2 <- matrix_to_vector_idx(k_idx, l_idx) + 1  # +1 for R indexing
            
            # +1 for R indexing
            info_val <- nu * (
              psi_inv[i_idx + 1, k_idx + 1] * psi_inv[j_idx + 1, l_idx + 1] +
              psi_inv[i_idx + 1, l_idx + 1] * psi_inv[j_idx + 1, k_idx + 1]
            )
            
            information[idx1, idx2] <- information[idx1, idx2] + info_val
          }
        }
      }
    }
  }
  
  # Compute sandwich covariance matrix
  info_inv <- solve(information)
  sampling_cov <- info_inv %*% score_cov %*% t(info_inv)
  
  return(sampling_cov)
}

#' Example usage of EMCovarianceCombiner
#' 
#' @examples
#' \dontrun{
#' # Create some example data
#' cov1 <- matrix(c(1.0, 0.5, 0.5, 2.0), nrow = 2)
#' cov2 <- matrix(c(1.5, 0.3, 0.3, 1.8), nrow = 2)
#' cov3 <- matrix(c(2.0, 1.0, 1.0, 3.0), nrow = 2)
#' 
#' # Define which variables are observed in each matrix (0-based indices)
#' idx1 <- c(0, 1)  # Variables 0 and 1 in first matrix
#' idx2 <- c(1, 2)  # Variables 1 and 2 in second matrix
#' idx3 <- c(0, 2)  # Variables 0 and 2 in third matrix
#' 
#' # Run the EM algorithm
#' result <- EMCovarianceCombiner(
#'   partial_covs = list(cov1, cov2, cov3),
#'   var_indices = list(idx1, idx2, idx3),
#'   degrees_freedom = c(100, 100, 100)
#' )
#' 
#' # Extract the combined covariance matrix
#' combined_cov <- result$psi
#' print(combined_cov)
#' 
#' # Extract the sampling covariance matrix
#' sampling_cov <- result$sampling_cov
#' print(sampling_cov)
#' 
#' # Plot the log-likelihood path
#' if (!is.null(result$loglik_path)) {
#'   plot(result$loglik_path, type = "l", 
#'        xlab = "Iteration", ylab = "Log-likelihood")
#' }
#' }