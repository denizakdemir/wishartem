"""
Unit tests for the EMCovarianceCombiner class.
"""
import numpy as np
import unittest
from wishartem import EMCovarianceCombiner

class TestEMCovarianceCombiner(unittest.TestCase):
    """Test cases for the EMCovarianceCombiner class."""

    def setUp(self):
        """Set up test cases."""
        self.combiner = EMCovarianceCombiner(max_iter=100, tol=1e-6)
        
        # Complete observations test case
        self.cov1 = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        self.cov2 = np.array([
            [1.5, 0.6, 0.2],
            [0.6, 1.3, 0.5],
            [0.2, 0.5, 1.1]
        ])

        self.cov3 = np.array([
            [1.2, 0.4, 0.1],
            [0.4, 1.1, 0.3],
            [0.1, 0.3, 0.9]
        ])

        self.partial_covs = [self.cov1, self.cov2, self.cov3]
        self.var_indices = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        self.degrees_freedom = [100, 120, 90]
        
        # Overlapping observations test case
        self.var_indices_overlap = [[0, 3], [1, 2], [0, 2]]
        self.partial_covs_overlap = [
            np.array([[1.0, 0.5],
                     [0.5, 2.0]]),  # Variables 0,3
            np.array([[1.5, 0.6],
                     [0.6, 1.2]]),  # Variables 1,2
            np.array([[1.2, 0.3],
                     [0.3, 1.1]])   # Variables 0,2
        ]

    def test_initialize_psi(self):
        """Test initialization of Psi matrix."""
        n_vars = 3
        psi = self.combiner._initialize_psi(self.partial_covs, self.var_indices, n_vars)
        
        self.assertEqual(psi.shape, (n_vars, n_vars))
        self.assertTrue(np.allclose(np.diag(psi), [1.2333, 1.4667, 1.0000], rtol=1e-4))
        
    def test_compute_conditional_expectation_no_missing(self):
        """Test conditional expectation computation with no missing variables."""
        ya = self.cov1
        psi = np.eye(3)
        obs_idx = [0, 1, 2]
        missing_idx = []
        
        result = self.combiner._compute_conditional_expectation(ya, psi, obs_idx, missing_idx)
        
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.allclose(result, ya))
        
    def test_compute_conditional_expectation_with_missing(self):
        """Test conditional expectation computation with missing variables."""
        # Create a test case with missing variables
        ya = np.array([[1.0, 0.5], [0.5, 2.0]])  # 2x2 matrix for variables 0,1
        psi = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        obs_idx = [0, 1]
        missing_idx = [2]
        
        result = self.combiner._compute_conditional_expectation(ya, psi, obs_idx, missing_idx)
        
        self.assertEqual(result.shape, (3, 3))
        # Check that observed part matches the input
        self.assertTrue(np.allclose(result[0:2, 0:2], ya))
        # Check that the missing parts are filled in correctly
        self.assertFalse(np.allclose(result[2, 0:2], np.zeros(2)))
        self.assertFalse(np.allclose(result[0:2, 2], np.zeros(2)))
        self.assertFalse(np.isclose(result[2, 2], 0))
        
    def test_fit_complete_observations(self):
        """Test fitting with complete observations."""
        combined_cov, sampling_cov = self.combiner.fit(
            self.partial_covs, self.var_indices, self.degrees_freedom
        )
        
        self.assertEqual(combined_cov.shape, (3, 3))
        self.assertEqual(sampling_cov.shape, (6, 6))  # 3*(3+1)/2 = 6 unique elements
        
        # Check that the combined covariance is symmetric
        self.assertTrue(np.allclose(combined_cov, combined_cov.T))
        
        # Check that the combined covariance is positive definite
        eigenvals = np.linalg.eigvals(combined_cov)
        self.assertTrue(np.all(eigenvals > 0))
        
    def test_fit_overlapping_observations(self):
        """Test fitting with overlapping observations."""
        combined_cov, sampling_cov = self.combiner.fit(
            self.partial_covs_overlap, self.var_indices_overlap, self.degrees_freedom
        )
        
        self.assertEqual(combined_cov.shape, (4, 4))
        self.assertEqual(sampling_cov.shape, (10, 10))  # 4*(4+1)/2 = 10 unique elements
        
        # Check that the combined covariance is symmetric
        self.assertTrue(np.allclose(combined_cov, combined_cov.T))
        
        # Check that the combined covariance is positive definite
        eigenvals = np.linalg.eigvals(combined_cov)
        self.assertTrue(np.all(eigenvals > 0))
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test empty inputs
        with self.assertRaises(ValueError):
            self.combiner.fit([], [])
            
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            self.combiner.fit(self.partial_covs, self.var_indices[:-1])
            
        # Test mismatched degrees of freedom
        with self.assertRaises(ValueError):
            self.combiner.fit(self.partial_covs, self.var_indices, [100, 120])

if __name__ == '__main__':
    unittest.main()