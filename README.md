# NearestCorrelationMatrix

This is a light-weight pure-julia library for computing the nearest correlation matrix using a quadratically convergent algorithm.

## Usage

```julia
using NearestCorrelationMatrix

# compute the nearest correlation matrix
x = rand(10, 10)
R = cor_nearest_posdef(x)

# compute the nearest positive semidefinite correlation matrix
y = rand(10, 10)
S = cor_nearest_posdef(x, 0)

# compute a close (not nearest) correlation matrix
z = rand(10, 10)
P = cor_fast_posdef(y)

# compute a close (not nearest) correlation matrix in place
Q = rand(10, 10)
cor_fast_posdef!(Q)
```

## Details

- `cor_nearest_posdef` uses the quadratically convergent Newton method as described by Qi and Sun
- `cor_fast_posdef` computes the eigen decomposition of the input matrix, and replaces any negative eigenvalues with a small positive value, then re-creates the input matrix with the adjusted eigenvalues

## References

* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.