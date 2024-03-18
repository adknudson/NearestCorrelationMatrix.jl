"""
    NCMProblem(A, p=NullParameters(); kwargs...)

Defines the semi-definite programming problem of finding the nearest correlation matrix to
a given input matrix.
"""
struct NCMProblem{T, P, K}
    A::T
    p::P
    kwargs::K
    function NCMProblem(A, p=NullParameters(); kwargs...)
        require_matrix(A)
        require_square(A)

        new{typeof(A), typeof(p), typeof(kwargs)}(A, p, kwargs)
    end
end
