export
    clamp_cor,
    cov2cor!,
    eigen_sym,
    setdiag!,
    symmetrize!


"""
    clamp_cor(x::Real)
    clamp_cor(x::Real)

Constrain a value between -1 and 1.
"""
clamp_cor(x::Real) = clamp(x, -1, 1)
clamp_cor(x) = x

"""
    cov2cor!(X)

Compute the correlation matrix from the covariance matrix ``X`` and overwrite its values.
More generally this transforms the input matrix ``X`` into a correlation matrix without
changing its positive-definiteness.
"""
function cov2cor!(X::AbstractMatrix)
    Base.require_one_based_indexing(X)
    s = map(sqrt, view(X, diagind(X)))
    n = length(s)
    size(X) == (n, n) || throw(DimensionMismatch("inconsistent dimensions"))
    for j in 1:n
        sj = s[j]
        for i in 1:(j - 1)
            X[i, j] = adjoint(X[j, i])
        end
        X[j, j] = oneunit(X[j, j])
        for i in (j + 1):n
            X[i, j] = clamp_cor(X[i, j] / (s[i] * sj))
        end
    end
    return X
end

# Preserve structure of Symmetric covariance matrices
function cov2cor!(X::Symmetric{<:Real})
    s = map(sqrt, view(X, diagind(X)))
    n = length(s)
    size(X) == (n, n) || throw(DimensionMismatch("inconsistent dimensions"))
    A = parent(X)
    if X.uplo === 'U'
        for j in 1:n
            sj = s[j]
            for i in 1:(j - 1)
                A[i, j] = clamp_cor(A[i, j] / (s[i] * sj))
            end
            A[j, j] = oneunit(A[j, j])
        end
    else
        for j in 1:n
            sj = s[j]
            A[j, j] = oneunit(A[j, j])
            for i in (j + 1):n
                A[i, j] = clamp_cor(A[i, j] / (s[i] * sj))
            end
        end
    end
# Preserve structure of Symmetric covariance matrices
function cov2cor!(X::Symmetric{<:Real})
    s = map(sqrt, view(X, diagind(X)))
    n = length(s)
    size(X) == (n, n) || throw(DimensionMismatch("inconsistent dimensions"))
    A = parent(X)
    if X.uplo === 'U'
        for j in 1:n
            sj = s[j]
            for i in 1:(j - 1)
                A[i, j] = clamp_cor(A[i, j] / (s[i] * sj))
            end
            A[j, j] = oneunit(A[j, j])
        end
    else
        for j in 1:n
            sj = s[j]
            A[j, j] = oneunit(A[j, j])
            for i in (j + 1):n
                A[i, j] = clamp_cor(A[i, j] / (s[i] * sj))
            end
        end
    end
    return X
end

"""
    eigen_sym(X)

Compute the eigen decomposition of the symmetric matrix ``X``. Eigenvalues are sorted in
descending order.

If ``X`` is not symmetric, then a symmetric view of its upper/lower triangle will be created
and used instead.
"""
eigen_sym(X::Symmetric) = eigen(X; sortby = x -> -x)

function eigen_sym(X::Symmetric{Float16})
    E = eigen(X; sortby = x -> -x)
    values = convert(AbstractVector{Float16}, E.values)
    vectors = convert(AbstractMatrix{Float16}, E.vectors)
    return Eigen(values, vectors)
end

eigen_sym(X, uplo = :U) = eigen_sym(Symmetric(X, uplo))

"""
    setdiag!(X, v)

Set the diagonal elements of ``X`` to ``v``.
"""
function setdiag!(X::AbstractMatrix{T}, v::S) where {T, S}
    require_square(X)
    vt = convert(T, v)
    @inbounds for i in diagind(X)
        X[i] = vt
    end
    return X
end

"""
    symmetrize!(X::AbstractMatrix, uplo::Symbol=:U)

Symmetrize a square matrix `X` in-place by mirroring the upper (`:U`)
or lower (`:L`) triangle to the opposite side.
"""
function symmetrize!(X::AbstractMatrix, uplo::Symbol = :U)
    n = require_square(X)
    if uplo === :U
        @inbounds for j in 1:n
            for i in (j + 1):n
                X[i, j] = X[j, i]
            end
        end
    elseif uplo === :L
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                X[i, j] = X[j, i]
            end
        end
    else
        throw(ArgumentError(lazy"uplo must be either :U or :L, got $uplo"))
    end
    return X
end

symmetrize!(X::Symmetric, ::Symbol) = symmetrize!(parent(X), X.uplo === 'U' ? :U : :L)
symmetrize!(X::Symmetric) = symmetrize!(parent(X), X.uplo === 'U' ? :U : :L)
symmetrize!(X::Diagonal, ::Symbol) = X
symmetrize!(X::Diagonal) = X
