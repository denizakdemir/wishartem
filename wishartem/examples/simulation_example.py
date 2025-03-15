"""
Example script demonstrating the usage of WishartEM for simulation studies.
"""
import numpy as np
from wishartem import EMCovarianceCombiner
from wishartem.simulation import run_power_analysis

def main():
    """Run a simulation example."""
    # Example usage
    n_samples = [1000, 1000]  # Sample sizes for each partial dataset
    
    # Test Type I error rate (direct_corr = 0)
    print("\nRunning Type I Error Analysis...")
    type1_results = run_power_analysis(direct_corr=0.0, n_samples=n_samples, n_sims=100)
    print("\nType I Error Rate Analysis (true correlation = 0):")
    print(f"Rejection rate: {type1_results['power']:.3f}")
    print(f"Mean estimated correlation: {type1_results['mean_est_corr']:.3f}")
    print(f"SD of estimated correlation: {type1_results['sd_est_corr']:.3f}")
    print(f"Mean standard error: {type1_results['mean_se']:.3f}")
    
    # Test power (direct_corr = 0.3)
    print("\nRunning Power Analysis...")
    power_results = run_power_analysis(direct_corr=0.3, n_samples=n_samples, n_sims=100)
    print("\nPower Analysis (direct correlation = 0.3, indirect correlation = 0.09):")
    print(f"Power: {power_results['power']:.3f}")
    print(f"Mean estimated correlation: {power_results['mean_est_corr']:.3f}")
    print(f"SD of estimated correlation: {power_results['sd_est_corr']:.3f}")
    print(f"Mean standard error: {power_results['mean_se']:.3f}")
    print(f"\nTheoretical indirect correlation: {0.3**2:.3f}")

if __name__ == "__main__":
    main()