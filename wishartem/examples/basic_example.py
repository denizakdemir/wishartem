"""
Basic example demonstrating the usage of WishartEM.
"""
import numpy as np
from wishartem import EMCovarianceCombiner

def main():
    """Run a basic example with WishartEM."""
    # Test Case 1: Complete observations
    print("\nTest Case 1: Complete observations")
    cov1 = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 2.0, 0.4],
        [0.3, 0.4, 1.0]
    ])

    cov2 = np.array([
        [1.5, 0.6, 0.2],
        [0.6, 1.3, 0.5],
        [0.2, 0.5, 1.1]
    ])

    cov3 = np.array([
        [1.2, 0.4, 0.1],
        [0.4, 1.1, 0.3],
        [0.1, 0.3, 0.9]
    ])

    partial_covs = [cov1, cov2, cov3]
    var_indices = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]  # All variables match
    degrees_freedom = [100, 120, 90]

    # Initialize and fit
    combiner = EMCovarianceCombiner(max_iter=100, tol=1e-6)
    try:
        combined_cov, sampling_cov = combiner.fit(partial_covs, var_indices, degrees_freedom)
        print("\nCombined Covariance Matrix:")
        print(combined_cov)
        print("\nMatrix dimensions:", combined_cov.shape)
        
        if sampling_cov is not None:
            print("\nSampling Covariance Matrix (first 6x6 part):")
            print(sampling_cov[:6, :6])
            print("\nSampling Covariance dimensions:", sampling_cov.shape)
            
            # Print standard errors for variances
            n_vars = combined_cov.shape[0]
            print("\nStandard Errors for Variances:")
            for i in range(n_vars):
                idx = i * (i + 1) // 2 + i  # Correct formula for diagonal elements
                if idx < sampling_cov.shape[0]:
                    se = np.sqrt(sampling_cov[idx, idx])
                    print(f"Var[{i}]: {combined_cov[i,i]:.4f} ± {se:.4f}")
                    
    except Exception as e:
        print(f"Error during fitting: {str(e)}")
        
    # Test Case 2: Overlapping observations
    print("\nTest Case 2: Overlapping observations")
    var_indices_overlap = [[0, 3], [1, 2], [0, 2]]
    partial_covs_overlap = [
        np.array([[1.0, 0.5],
                 [0.5, 2.0]]),  # Variables 0,3
        np.array([[1.5, 0.6],
                 [0.6, 1.2]]),  # Variables 1,2
        np.array([[1.2, 0.3],
                 [0.3, 1.1]])   # Variables 0,2
    ]
    
    try:
        combined_cov_overlap, sampling_cov_overlap = combiner.fit(
            partial_covs_overlap, var_indices_overlap, degrees_freedom
        )
        print("\nCombined Covariance Matrix (overlapping case):")
        print(combined_cov_overlap)
        print("\nMatrix dimensions:", combined_cov_overlap.shape)
        
        if sampling_cov_overlap is not None:
            print("\nSampling Covariance Matrix (overlapping case, first 6x6 part):")
            print(sampling_cov_overlap[:6, :6])
            print("\nSampling Covariance dimensions:", sampling_cov_overlap.shape)
            
            # Extract and print standard errors for variances
            n_vars = combined_cov_overlap.shape[0]
            print("\nStandard Errors for Variances:")
            for i in range(n_vars):
                idx = i * (i + 1) // 2 + i  # Correct formula for diagonal elements
                if idx < sampling_cov_overlap.shape[0]:
                    se = np.sqrt(sampling_cov_overlap[idx, idx])
                    print(f"Var[{i}]: {combined_cov_overlap[i,i]:.4f} ± {se:.4f}")
                            
    except Exception as e:
        print(f"Error during fitting (overlapping case): {str(e)}")

if __name__ == "__main__":
    main()