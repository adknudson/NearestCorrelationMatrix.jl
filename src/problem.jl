"""
    NCMProblem(A, p=NullParameters(); kwargs...)

Defines the semi-definite programming problem of finding the nearest correlation matrix to
a given input matrix.

## Mathematical Specification of a Correlation Matrix

To define a `NCMProblem`, you only need to provide a square `AbstractMatrix` ``A``.

## Problem Type

### Constructors

There is only one constructor for a `NCMProblem`

```julia
NCMProblem(A, p=NullParameters(); kwargs...)
```

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers.

### Fields

- `A`: The input matrix. Must be square. Should be symmetric.
- `p`: The parameters for the problem. Defaults to `NullParameters`. Currently unused.
- `kwargs`: The keyword arguments passed on to the solvers.
"""
struct NCMProblem{T,P,K}
    A::T
    p::P
    kwargs::K
    function NCMProblem(A, p=NullParameters(); kwargs...)
        require_matrix(A)
        require_square(A)
        require_real(A)

        return new{typeof(A),typeof(p),typeof(kwargs)}(A, p, kwargs)
    end
end
