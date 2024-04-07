# Quick Start

This package follows the [CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) interface for finding the nearest correlation matrix. In brief, this means you define the problem, initialize the solver, then call `solve!` on the solver.

By default, the input matrix is assumed to be a square symmetric matrix of real values. The following defines a symmetric matrix and an `NCMProblem` which is subsequently solved by the default algorithm.

```@example ncmprob1
using NearestCorrelationMatrix, LinearAlgebra

G = Symmetric(rand(4, 4))
prob = NCMProblem(G)
solver = init(prob)
sol = solve!(solver)
sol.X
```

You can also bypass the init step and call `solve` directly on the problem. Extra arguments are passed on to `init`. In this example, we use the alternating projections method.

```@example ncmprob1
sol = solve(prob, AlternatingProjections())
sol.X
```

Note: `solve(prob, args...; kwargs...)` is equivalent to `solve!(init(prob, args...; kwargs...))`

## Ensuring Positive Definiteness

A valid correlation matrix is only guaranteed to be positive semi-definite. If your application requires the solution to be positive definite (e.g. for Cholesky decomposition), then you can pass in the argument `ensure_pd=true`.

```@example ncmprob1
G = Symmetric(rand(100, 100))
prob = NCMProblem(G)
sol = solve(prob; ensure_pd=true)
isposdef(sol.X)
```

## Ensuring Correct Input

By default, the initialization step will complain about any malformed input. This forces the user to be explicit about how the input is treated and processed. As an example, the init step will complain if the input matrix is not symmetric.

```@example ncmprob1
G = rand(4, 4)
prob = NCMProblem(G)
try #hide
init(prob)
catch e; showerror(stderr, e); end #hide
```

To fix the error, we can follow the suggestion and pass in `fix_sym=true` and `uplo=:U` to use the upper part of the matrix.

```@example ncmprob1
init(prob; fix_sym=true, uplo=:U)
nothing #hide
```

