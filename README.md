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

# compute a "close" correlation matrix
X = rand(10, 10)
p = nearest_cor(X, DirectProjection())
```

## Benchmarks and Accuracy

The speed of each algorithm will vary from machine to machine, but the relative speeds should be consistent. These benchmarks are done using a Ryzen 9-3900X.

```julia
# evaluating accuracy in the Frobenius norm
using Random
Random.seed!(0xc0ffee)
X = rand(100, 100)

@time p1 = nearest_cor(X, Newton());
# 0.039854 seconds (737 allocations: 4.646 MiB)

@time p2 = nearest_cor(X, AlternatingProjection());
# 1.250281 seconds (4.30 k allocations: 88.480 MiB, 0.81% gc time)

@time p3 = nearest_cor(X, DirectProjection());
# 0.013710 seconds (416 allocations: 573.047 KiB)

norm(X - p1), norm(X - p2), norm(X - p3)
# (27.47958819327149, 27.460326444472884, 36.4857064980167)

X = rand(3000, 3000)
@time nearest_cor!(X, Newton());
# 33.347586 seconds (1.17 k allocations: 6.203 GiB, 0.67% gc time, 0.01% compilation time)
```

## Details

The method `nearest_cor` accepts a matrix and a nearest correlation algorithm. The following algorithms are implemented:

- `Newton`: An accurate and quadratically convergent algorithm
- `AlternatingProjection`: A more accurate but linearly convergent algorithm
- `DirectProjection`: A fast, one-step projection onto the set of correlation matrices

The default algorithm is the Newton method, which offers a great balance between accuracy and speed. 

## NCM Interface

Those wanting to implement their own solver must define the following:

```julia
```

## References

* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.
* https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/