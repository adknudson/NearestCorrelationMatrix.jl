"""
    NCMProblem(A, p=NullParameters(); mask=nothing, kwargs...)

Defines the semi-definite programming problem of finding the nearest correlation matrix to a
given input matrix.

To define a `NCMProblem`, you only need to provide a square `AbstractMatrix` ``A``.
Optionally, a mask of fixed element-pairs can be supplied. Only certain algorithms can make
use of the mask.

## Problem Type

### Constructors

```julia
NCMProblem(A, mask=nothing, p=NullParameters(); kwargs...)
```

Parameters are optional, and if not given, then a `NullParameters()` singleton will be used,
which will throw nice errors if you try to index non-existent parameters. Any extra keyword
arguments are stored in the `kwargs` field and forwarded on to the solvers.

### Fields

- `A`: The input matrix. Must be square. Should be symmetric.
- `p`: The parameters for the problem. Defaults to `NullParameters`. Currently unused.
- `mask`: A pattern
- `kwargs`: The keyword arguments passed on to the solvers.
"""
struct NCMProblem{T, M, P, K}
    A::T
    mask::M
    p::P
    kwargs::K
    function NCMProblem(A, p = NullParameters(); mask = nothing, kwargs...)
        require_matrix(A)
        require_square(A)
        require_real(A)
        return new{typeof(A), typeof(mask), typeof(p), typeof(kwargs)}(A, mask, p, kwargs)
    end
end
