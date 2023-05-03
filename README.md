# NearestCorrelationMatrix

This is a light-weight pure-julia library for computing the nearest correlation matrix using a quadratically convergent algorithm.

## Usage

```julia
using NearestCorrelationMatrix

# compute the nearest correlation matrix with default settings
x = rand(10, 10)
Q = nearest_cor(x)

# compute the nearest positive semidefinite correlation matrix
y = rand(10, 10)
R = nearest_cor(x, Newton(τ=0.0))

# compute the nearest correlation matrix in place
S = rand(10, 10)
nearest_cor!(S)
```

## Details

The method `nearest_cor` accepts a matrix and a nearest correlation algorithm. The following algorithms are implemented:

- `Newton`: A quadratically convergent algorithm

## References

* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.