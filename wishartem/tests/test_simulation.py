"""
Unit tests for the simulation module.
"""
import numpy as np
import unittest
from wishartem.simulation import (
    generate_triangle_data,
    create_partial_datasets,
    matrix_to_vector_idx,
    run_single_simulation
)

class TestSimulation(unittest.TestCase):
    """Test cases for the simulation module."""
    
    def test_generate_triangle_data(self):
        """Test generation of triangular correlation data."""
        # Test with positive correlation
        n_samples = 1000
        direct_corr = 0.5
        x1, x2, x3 = generate_triangle_data(n_samples, direct_corr, seed=42)
        
        self.assertEqual(x1.shape, (n_samples,))
        self.assertEqual(x2.shape, (n_samples,))
        self.assertEqual(x3.shape, (n_samples,))
        
        # Check correlations approximately match the expected values
        corr12 = np.corrcoef(x1, x2)[0, 1]
        corr23 = np.corrcoef(x2, x3)[0, 1]
        corr13 = np.corrcoef(x1, x3)[0, 1]
        
        self.assertAlmostEqual(corr12, direct_corr, delta=0.1)
        self.assertAlmostEqual(corr23, direct_corr, delta=0.1)
        self.assertAlmostEqual(corr13, direct_corr**2, delta=0.1)
        
        # Test with very high correlation (close to the boundary of positive definiteness)
        direct_corr = 0.9
        x1, x2, x3 = generate_triangle_data(n_samples, direct_corr, seed=42)
        self.assertAlmostEqual(np.corrcoef(x1, x3)[0, 1], direct_corr**2, delta=0.2)
            
    def test_create_partial_datasets(self):
        """Test creation of partial datasets."""
        n_samples = [500, 500]
        x1 = np.random.normal(0, 1, 1000)
        x2 = np.random.normal(0, 1, 1000)
        x3 = np.random.normal(0, 1, 1000)
        
        partial_covs, var_indices = create_partial_datasets(x1, x2, x3, n_samples)
        
        self.assertEqual(len(partial_covs), 2)
        self.assertEqual(len(var_indices), 2)
        self.assertEqual(var_indices, [[0, 1], [1, 2]])
        
        # Check dimensions of covariance matrices
        self.assertEqual(partial_covs[0].shape, (2, 2))
        self.assertEqual(partial_covs[1].shape, (2, 2))
        
    def test_matrix_to_vector_idx(self):
        """Test matrix to vector index conversion."""
        # Test a few cases
        self.assertEqual(matrix_to_vector_idx(0, 0, 3), 0)
        self.assertEqual(matrix_to_vector_idx(1, 1, 3), 3)  # Using formula from simulation.py
        self.assertEqual(matrix_to_vector_idx(2, 2, 3), 5)
        self.assertEqual(matrix_to_vector_idx(0, 1, 3), 1)
        self.assertEqual(matrix_to_vector_idx(1, 0, 3), 1)  # Should swap to ensure i < j
        self.assertEqual(matrix_to_vector_idx(0, 2, 3), 2)
        self.assertEqual(matrix_to_vector_idx(1, 2, 3), 4)
        
    def test_run_single_simulation(self):
        """Test running a single simulation."""
        direct_corr = 0.3
        n_samples = [500, 500]
        
        # Run simulation with fixed seed for reproducibility
        est_corr, se, p_value = run_single_simulation(direct_corr, n_samples, seed=42)
        
        # Check return types and ranges
        self.assertIsInstance(est_corr, float)
        self.assertIsInstance(se, float)
        self.assertIsInstance(p_value, float)
        
        self.assertTrue(-1 <= est_corr <= 1)
        self.assertTrue(se >= 0)
        self.assertTrue(0 <= p_value <= 1)
        
        # Check that estimated correlation is roughly close to expected value
        # (direct_corr^2 since it's the indirect correlation)
        self.assertAlmostEqual(est_corr, direct_corr**2, delta=0.2)

if __name__ == '__main__':
    unittest.main()