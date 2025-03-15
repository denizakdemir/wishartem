# WishartEM

WishartEM is a Python package implementing an EM algorithm for combining partial covariance matrices, as described in the paper "NewWishartEM.pdf".

## Features

- Combine multiple partially observed covariance matrices into a single combined covariance matrix
- Compute sampling covariance matrix for the combined covariance matrix
- Handle overlapping and non-overlapping observations
- Support for simulation studies to evaluate the performance of the algorithm

## Installation

```bash
pip install wishartem
```

Or install from source:

```bash
git clone https://github.com/denizakdemir/WishartEM.git
cd WishartEM
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from wishartem import EMCovarianceCombiner

# Example covariance matrices
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

partial_covs = [cov1, cov2]
var_indices = [[0, 1, 2], [0, 1, 2]]  # All variables match
degrees_freedom = [100, 120]

# Initialize and fit
combiner = EMCovarianceCombiner(max_iter=100, tol=1e-6)
combined_cov, sampling_cov = combiner.fit(partial_covs, var_indices, degrees_freedom)

print("Combined Covariance Matrix:")
print(combined_cov)
```

### Overlapping Observations Example

```python
# Example with overlapping observations
var_indices_overlap = [[0, 3], [1, 2], [0, 2]]
partial_covs_overlap = [
    np.array([[1.0, 0.5],
             [0.5, 2.0]]),  # Variables 0,3
    np.array([[1.5, 0.6],
             [0.6, 1.2]]),  # Variables 1,2
    np.array([[1.2, 0.3],
             [0.3, 1.1]])   # Variables 0,2
]

combined_cov_overlap, sampling_cov_overlap = combiner.fit(
    partial_covs_overlap, var_indices_overlap, degrees_freedom
)

print("Combined Covariance Matrix (overlapping case):")
print(combined_cov_overlap)
```

### Simulation Example

```python
from wishartem.simulation import run_power_analysis

# Run power analysis
n_samples = [1000, 1000]
power_results = run_power_analysis(direct_corr=0.3, n_samples=n_samples, n_sims=100)

print(f"Power: {power_results['power']:.3f}")
print(f"Mean estimated correlation: {power_results['mean_est_corr']:.3f}")
```

## API Documentation

### EMCovarianceCombiner

```python
class EMCovarianceCombiner(max_iter=100, tol=1e-6)
```

Implementation of EM algorithm for combining partial covariance matrices.

**Parameters:**
- `max_iter` : int, default=100
  Maximum number of iterations for the EM algorithm.
- `tol` : float, default=1e-6
  Convergence tolerance for the EM algorithm.

**Methods:**
- `fit(partial_covs, var_indices, degrees_freedom=None)` : Fit the EM algorithm to combine partial covariance matrices.
  - `partial_covs` : List of partial covariance matrices.
  - `var_indices` : List of variable indices for each partial covariance matrix.
  - `degrees_freedom` : Optional list of degrees of freedom for each partial covariance matrix.
  - Returns: Tuple of combined covariance matrix and its sampling covariance.

## Development

### Running Tests

```bash
python -m unittest discover -s wishartem/tests
```

## License

MIT

## Citation

If you use this package in your research, please cite the paper:

[Paper citation information]