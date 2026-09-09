using LinearAlgebra

export project_s, project_u, project_f, project_f!

"""
    project_s(X, WHalf, WHalfInv)

Project ``X`` onto the set of symmetric positive semi-definite matrices with a W-norm.
"""
function project_s(X, Whalf, Whalfinv)
    Y = Whalfinv * project_psd(Whalf * X * Whalf) * Whalfinv
    return Symmetric(Y)
end

"""
    project_s(X)

Project ``X`` onto the set of symmetric positive semi-definite matrices.
"""
project_s(X) = Symmetric(project_psd(X))

"""
    project_u(X)

Project ``X`` onto the set of symmetric matrices with unit diagonal.
"""
function project_u(X)
    Y = copy(X)
    setdiag!(Y, one(eltype(Y)))
    return Symmetric(Y)
end

"""
    project_f!(X, A, M)

Project ``X`` onto the fixed-element subspace defined by the mask ``M`` and the values ``A``.

For every upper-triangular position ``(i, j)`` (with ``i < j``) for which ``M[i, j]`` is ``true``,
set ``X[i, j] = A[i, j]`` and its symmetric mirror ``X[j, i] = A[j, i]``. The diagonal is left
untouched, since it is always forced to 1 by ``project_u``.

# Examples

```julia
julia> A = [1.0 0.1 0.5; 0.1 1.0 0.4; 0.5 0.4 1.0];   # original input (value source)

julia> X = [1.0 0.2 0.3; 0.2 1.0 0.6; 0.3 0.6 1.0];   # current iterate

julia> M = [false false false; false false true; false true false];

julia> project_f!(X, A, M)
3×3 Matrix{Float64}:
 1.0  0.2  0.3
 0.2  1.0  0.5
 0.3  0.5  1.0
```

The masked position `(2, 3)` (and its mirror) is set to the value from `A`, leaving the other
entries of `X` untouched.
"""
function project_f!(X::AbstractMatrix, A::AbstractMatrix, M::AbstractMatrix)
    n, m = size(X)
    T = eltype(X)
    require_square(A)
    require_square(M)
    size(A) == size(M) ||
        throw(DimensionMismatch("A and M must have the same size: $(size(A)) vs $(size(M))"))
    size(X) == size(A) ||
        throw(DimensionMismatch("X and A must have the same size: $(size(X)) vs $(size(A))"))
    # Symmetric/Hermitian wrappers only permit diagonal writes, so set both mirrored entries in
    # the parent storage. For a plain AbstractMatrix, parent is the matrix itself.
    Xs = parent(X)
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):m
            if M[i, j]
                @inbounds Xs[i, j] = T(A[i, j])
                @inbounds Xs[j, i] = T(A[j, i])
            end
        end
    end
    return X
end

"""
    project_f(X, A, M)

Return a copy of ``X`` projected onto the fixed-element subspace (see [`project_f!`](@ref)).
"""
project_f(X, A, M) = project_f!(copy(X), A, M)
