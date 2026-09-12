export
    clamp_cor,
    setdiag!,
    symmetrize!,
    cov2cor!,
    eigen_sym


"""
    clamp_cor(x::Real)

Constrain a value between -1 and 1.
"""
clamp_cor(x::Real) = clamp(x, -1, 1)
clamp_cor(x) = x

"""
    setdiag!(X, v)

Set the diagonal elements of ``X`` to ``v``.
"""
function setdiag!(X::AbstractMatrix{T}, v::T) where {T}
    for i in diagind(X)
        @inbounds X[i] = v
    end

    return X
end

"""
    symmetrize!(X, uplo=:U)

Make ``X`` symmetric in place by copying either the upper (`uplo=:U`) or lower (`uplo=:L`)
triangle of ``X``.
"""
function symmetrize!(X::AbstractMatrix, uplo::Symbol = :U)
    nr, nc = size(X)
    nr == nc || error("X must be a square matrix.")
    uplo ∈ (:U, :L)  || error("uplo must be in (:U, :L)")
    if uplo === :U # copy upper to lower
        for j in 1:(nc - 1)
            for i in (j + 1):nr
                X[i, j] = X[j, i]
            end
        end
    else # copy lower to upper
        for j in 1:(nc - 1)
            for i in (j + 1):nr
                X[j, i] = X[i, j]
            end
        end
    end
    return X
end

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
        C[j, j] = oneunit(C[j, j])
        for i in (j + 1):n
            C[i, j] = clamp_cor(C[i, j] / (s[i] * sj))
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
