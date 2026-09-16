"""
    NCMProblem(A, p=NullParameters(); mask=nothing, kwargs...)

Defines the semi-definite programming problem of finding the nearest correlation matrix to a
given input matrix.

To define a `NCMProblem`, you only need to provide a square matrix ``A``.
Optionally, a mask of fixed element-pairs can be supplied. Only certain algorithms can make
use of the mask.

## Arguments

- `A`: The input matrix. Must be square. Should be symmetric.
- `p`: The parameters for the problem. Defaults to `NullParameters`. Currently unused.

## Keyword Arguments

- `mask`: A BitMatrix or a matrix of 1s/0s indicating which elements must remain fixed.
- `kwargs`: Additional keyword arguments passed on to the `init` function.
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
