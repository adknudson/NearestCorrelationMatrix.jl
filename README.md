# NearestCorrelationMatrix

This is a light-weight pure-julia library for computing the nearest correlation matrix using a quadratically convergent algorithm.

## Usage

```julia
using NearestCorrelationMatrix

# compute the nearest correlation matrix with default settings
X = rand(10, 10)
P = nearest_cor(X)

# compute the nearest positive semidefinite correlation matrix
X = rand(10, 10)
P = nearest_cor(X, Newton(τ=0.0))

# compute the nearest correlation matrix in place
P = rand(10, 10)
nearest_cor!(P)

# compute the nearest correlation matrix using the Alternating Projection method
X = rand(10, 10)
P = nearest_cor(X, AlternatingProjection(maxiter=10))
```

## Details

The method `nearest_cor` accepts a matrix and a nearest correlation algorithm. The following algorithms are implemented:

- `Newton`: A quadratically convergent algorithm
- `AlternatingProjection`: Slower, but more accurate algorithm

## References

* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.
* https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/