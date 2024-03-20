# NearestCorrelationMatrix

This is a light-weight pure-julia library for computing the nearest correlation matrix. A few algorithms are provided, including a quadratically convergent Newton algorithm.


## Usage


### CommonSolve.jl Interface

This package follows the [CommonSolve.jl](https://docs.sciml.ai/CommonSolve/stable/) interface for the nearest correlation matrix (NCM) problem.

```julia
using NearestCorrelationMatrix, LinearAlgebra

A = Symmetric(rand(10, 10))
prob = NCMProblem(A)
solver = init(prob)
sol = solve!(solver)
X = sol.X # get the nearest correlation matrix

# alternatively you can solve the problem directly

sol = solve(prob) # equivalent to solve!(init(prob))
```

You can initialize the solver with a given algorithm and options

```julia
A = rand(10, 10)
prob = NCMProblem(A)

# initialization keywords
solver = init(prob;
    fix_sym=true, # tell the solver to fix non-symmetric inputs
    ensure_pd=true, # ensure that the resulting matrix is positive definite
    verbose=true, # show more output during solving
)

# initialize with an algorithm with default parameters
solver = init(prob, Newton())

# initalize with an algorithm that specializes its parameters to the input
solver = init(prob, Newton)

# the above is equivalent to
alg = autotune(Newton, prob)
solver = init(prob, alg)
```


### Simplified Interface

The `nearest_cor` and `nearest_cor!` methods are the "batteries included" methods for finding the nearest correlation matrix, and are designed to just work. They also have a final check to ensure that the solution is strictly positive-definite, as often that is required for other algorithms (e.g. Cholesky decomposition).

```julia
r0 = rand(10, 10)
nearest_cor(r0) # uses the default algorithm
nearest_cor(r0, AlternatingProjections()) # uses AP with default values

r0 = rand(Float32, 10, 10)
nearest_cor(r0, Newton()) # uses Newton method with default parameters
nearest_cor(r0, Newton) # uses Newton method with parameters tuned to the input matrix

r = rand(10, 10)
nearest_cor!(r) # computes NCM and overwrites r
```


### Solver Keyword Argumenets

#### General Controls

- `alias_A`: Whether to alias the matrix ``A`` or use a copy by default. When `true`,
  algorithms that operate in place can save memory by reusing ``A``. Defaults to `true` if
  the algorithm is known not to modify ``A``, and `false` otherwise.
- `fix_sym`: If `true`, then makes the input matrix symmetric if it is not already. Defaults
   to `false`, and init will fail if the input matrix is not symmetric.
- `uplo`: If `fix_sym` is `true`, then the upper (`:U`) or lower (`:L`) triangle of the
  input is used to make a symmetric matrix. Defaults to `:U`.
- `convert_f16`: If the algorithm does not support `Float16` values, then the input matrix
  will be converted to an `AbstractMatrix{Float32}`. Defaults to `false`, and init will
  fail if the algorithm does not support `Float16`.
- `force_f16`: If `true`, then the algorithm will be forced to use the input matrix, even
  if the algorithm doesn't fully support `Float16` values in a stable way.
- `ensure_pd`: Checks (and corrects) that the resulting matrix is positive definite.
  Defaults to `false`.
- `verbose`: Whether to print extra information. Defaults to `false`.

#### Solver Controls

- `abstol`: The absolute tolerance. Defaults to `√(eps(eltype(A)))`.
- `reltol`: The relative tolerance. Defaults to `√(eps(eltype(A)))`.
- `maxiters`: The number of iterations allowed. Defaults to `size(A,1)`


## Benchmarks

The speed of each algorithm will vary from machine to machine, but the relative speeds should be consistent. These benchmarks are done using a Ryzen 9-3900X.

```julia
using NearestCorrelationMatrix, LinearAlgebra
using Random

Random.seed!(0xc0ffee)
X = Matrix(Symmetric(rand(100, 100)))
prob = NCMProblem(X)

@time p1 = solve(prob, Newton());
# 0.023863 seconds (988 allocations: 11.740 MiB)

@time p2 = solve(prob, AlternatingProjections());
# 0.219939 seconds (3.33 k allocations: 81.715 MiB, 2.99% gc time)

@time p3 = solve(prob, DirectProjection());
# 0.003280 seconds (31 allocations: 510.984 KiB)

# evaluating accuracy in the Frobenius norm
norm(X - p1.X), norm(X - p2.X), norm(X - p3.X)
# (23.039971040930606, 23.039971034688392, 30.218305070132693)

X = rand(3000, 3000)
@time nearest_cor!(X, Newton());
# 24.934312 seconds (19.92 k allocations: 9.201 GiB, 1.92% gc time, 0.48% compilation time)
```

Using the MKL linear algebra backend can improve performance:

```julia
using NearestCorrelationMatrix, LinearAlgebra
using Random

Random.seed!(0xc0ffee)
X = Matrix(Symmetric(rand(3000, 3000)))
prob = NCMProblem(X)

BLAS.get_config()
# LinearAlgebra.BLAS.LBTConfig
# Libraries: 
# └ [ILP64] libopenblas64_.dll

@time solve(prob, Newton());
# 23.246179 seconds (1.22 M allocations: 8.876 GiB, 2.29% gc time, 2.43% compilation time)

using MKL

BLAS.get_config()
# LinearAlgebra.BLAS.LBTConfig
# Libraries:
# ├ [ILP64] mkl_rt.2.dll
# └ [ LP64] mkl_rt.2.dll

@time solve(prob, Newton());
# 18.310447 seconds (2.08 k allocations: 8.796 GiB, 3.13% gc time)
```

## Available Algorithms

The following algorithms are implemented:

- `Newton`: An accurate and quadratically convergent algorithm
- `AlternatingProjections`: A simple linearly convergent algorithm
- `DirectProjection`: A fast, one-step projection onto the set of correlation matrices

The default algorithm is the Newton method, which offers a great balance between accuracy and speed.

If `JuMP.jl` is installed and loaded, then the `JuMPAlgorithm` also becomes available:

```julia
using NearestCorrelationMatrix
using JuMP, COSMO

A = rand(10, 10)
prob = NCMProblem(A)

# COSMO is our recommended optimizer for NCM problems
optimizer = optimizer_with_attributes(
    COSMO.Optimizer,
    MOI.Silent() => true,
    "rho" => 1.0, # our testing shows that `rho=1.0` speeds up convergence
)

sol = solve(prob, JuMPAlgorithm(optimizer); fix_sym=true, uplo=:L)
```


## NCM Interface

Those wanting to implement their own solver must define the following:

```julia
using NearestCorrelationMatrix
using NearestCorrelationMatrix: build_ncm_solution

struct MyAlgorithm <: NCMAlgorithm end

function NearestCorrelationMatrix.solve!(solver::NCMSolver, alg::MyAlgorithm)
    A = solver.A
    
    # your implementation here
    X = my_method_that_finds_ncm(A)

    return build_ncm_solution(alg, X, resid, solver; iters=k)
end
```

If your algorithm can make use of cache, then you can define:

```julia
mutable struct MyAlgorithmCache{T1, T2}
    data1::T1
    data2::T2
end

function NearestCorrelationMatrix.init_cacheval(alg::MyAlgorithm, A, maxiters, abstol, reltol, verbose)
    # allocate whatever you need

    return MyAlgorithmCache(data1, data2)
end
```

The cache object can be whatever you want, and the `init_cacheval` method is called during the `init` step. The cache object is accesible via `solver.cacheval` in the `solve!` method.

There are also a few traits that can be defined:

```julia
default_iters(::MyAlgorithm, A) = size(A, 1)
modifies_in_place(::MyAlgorithm) = true
supports_float16(::MyAlgorithm) = false
supports_symmetric(::MyAlgorithm) = false
supports_parameterless_construction(::MyAlgorithm) = false
```


## Common Gotchas

Because a correlation matrix is defined as only positive semi-definite, it is possible for an algorithm to converge successfully but still not be useable for other methods such as a cholesky decomposition (common in probability models). Furthermore, the smallest eigenvalues may be negative on the order of machine precision (e.g. -2.2e-16) due to the inherent nature of floating point numbers. If a positive definite matrix is absolutely required, then pass the argument `ensure_pd=true` to the solver. This will replace the smallest eigenvalues with a small positive value and reconstruct the NCM:

```julia
λ, P = eigen(X)
replace(x -> max(x, ϵ), λ)
X .= P * Diagonal(λ) * P'
cov2cor!(X) # ensure that the transformed matrix is still a correlation matrix
```


## References

* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.
* https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/