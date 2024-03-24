using LinearAlgebra

export get_negdef_matrix,
    rand_negdef,
    clamp_pm1,
    clamp_pm1!,
    setdiag!,
    sym_uplo,
    symmetric!,
    corconstrain!,
    cov2cor,
    cov2cor!,
    cor2cov,
    cor2cov!,
    eigen_sym,
    project_psd!,
    project_psd

"""
    get_negdef_matrix(Type)

Get a negative definite matrix for testing.
"""
function get_negdef_matrix(::Type{T}) where {T}
    r = [
        1.0 -0.2188 -0.79 0.7773
        -0.2188 1.0 0.2559 -0.5977
        -0.79 0.2559 1.0 0.2266
        0.7773 -0.5977 0.2266 1.0
    ]

    return convert(AbstractMatrix{T}, r)
end

"""
    rand_negdef(T, n)

Generate a random negative definite matrix of size `n × n` with eltype ``T``.
"""
function rand_negdef(::Type{T}, n) where {T}
    while true
        r = 2 * rand(T, n, n) .- one(T)
        symmetric!(r)
        r[diagind(r)] .= one(T)

        !isposdef(r) && return r
    end
end

"""
    clamp_pm1(x::Real)

Constrain a value between -1 and 1.
"""
clamp_pm1(x::Real) = clamp(x, -one(x), one(x))

"""
    clamp_pm1!(A::AbstractArray)

Contstrain all values of an array to be between -1 and 1.
"""
function clamp_pm1!(A::AbstractArray{T,N}) where {T,N}
    @inbounds for i in eachindex(A)
        A[i] = clamp_pm1(A[i])
    end
    return A
end

"""
    setdiag!(X, v)

Set the diagonal elements of ``X`` to ``v``.
"""
function setdiag!(X::AbstractMatrix{T}, v::T) where {T}
    require_square(X)

    for i in diagind(X)
        @inbounds X[i] = v
    end

    return X
end

"""
    sym_uplo(uplo::Char)

Convert the `'U'` and `'L'` characters to their corresponding symbols (`:U` and `:L`).
"""
function sym_uplo(uplo::Char)
    if uplo == 'U'
        return :U
    elseif uplo == 'L'
        return :L
    else
        throw_uplo()
    end
end

@noinline throw_uplo() =
    throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))

"""
    symmetric!(X, uplo=:U)

Make ``X`` symmetric in place by copying either the upper (`uplo=:U`) or lower (`uplo=:L`)
triangle of ``X``.
"""
function symmetric!(X::AbstractMatrix, uplo::Symbol=:U)
    if uplo === :U
        _copytolower!(X)
    elseif uplo === :L
        _copytoupper!(X)
    else
        throw_uplo()
    end

    return X
end

symmetric!(X::Symmetric, ::Symbol=:U) = X
symmetric!(X::Diagonal, ::Symbol=:U) = X

function _copytolower!(X::AbstractMatrix)
    require_square(X)

    nr, nc = size(X)
    for j in 1:nc-1
        for i in j+1:nr
            @inbounds X[i, j] = X[j, i]
        end
    end
    return X
end

function _copytoupper!(X::AbstractMatrix)
    require_square(X)

    nr, nc = size(X)
    for j in 1:nc-1
        for i in j+1:nr
            @inbounds X[j, i] = X[i, j]
        end
    end
    return X
end

"""
    corconstrain!(X, uplo=:U)

Constrain ``X`` in place to be a pre-correlation.

A pre-correlation matrix must:

- be square
- be symmetric
- be constrained to ±1
- have diagonals equal to 1
"""
function corconstrain!(X::AbstractMatrix{T}, uplo::Symbol=:U) where {T}
    clamp_pm1!(X)
    setdiag!(X, one(T))
    symmetric!(X, uplo)
    return X
end

function corconstrain!(X::Symmetric{T}, ::Symbol=:U) where {T}
    clamp_pm1!(X.data)
    setdiag!(X, one(T))
    symmetric!(X.data, sym_uplo(X.uplo))
    return X
end

function corconstrain!(X::Diagonal{T}, ::Symbol=:U) where {T}
    fill!(X.diag, one(T))
    return X
end

"""
    cov2cor(X)

Compute the correlation matrix from the covariance matrix. More generally this transforms
the input matrix ``X`` into a correlation matrix without changing its positive-definiteness.
"""
function cov2cor(X::AbstractMatrix{T}) where {T}
    D = sqrt(inv(Diagonal(X)))
    return corconstrain!(D * X * D)
end

"""
    cov2cor!(X)

Compute the correlation matrix from the covariance matrix ``X`` and overwrite its values.
More generally this transforms the input matrix ``X`` into a correlation matrix without
changing its positive-definiteness.
"""
function cov2cor!(X::AbstractMatrix{T}) where {T}
    D = sqrt(inv(Diagonal(X)))
    lmul!(D, X)
    rmul!(X, D)
    setdiag!(X, one(T))
    symmetric!(X)
    return X
end

function cov2cor!(X::Symmetric{T}) where {T}
    symmetric!(X.data, sym_uplo(X.uplo))
    cov2cor!(X.data)
    return X
end

"""
    cor2cov(C, s)
"""
cor2cov(C::AbstractMatrix{T}, s::AbstractVector{T}) where {T} = cor2cov!(copy(C), s)

"""
    cor2cov!(C, s)

Compute the covariance matrix from the correlation matrix ``C`` and a vector of variances ``s``.
"""
function cor2cov!(C::AbstractMatrix{T}, s::AbstractVector{T}) where {T}
    for i in CartesianIndices(size(C))
        @inbounds C[i] *= s[i[1]] * s[i[2]]
    end
    return C
end

function cor2cov!(C::Symmetric{T}, s::AbstractVector{T}) where {T}
    symmetric!(C.data)
    cor2cov!(C.data, s)
    return C
end

"""
    eigen_sym(X)

Compute the eigen decomposition of the symmetric matrix ``X``. Eigenvalues are sorted in
descending order.

If ``X`` is not symmetric, then a symmetric view of its upper/lower triangle will be created
and used instead.
"""
eigen_sym(X::Symmetric) = eigen(X; sortby=x -> -x)

function eigen_sym(X::Symmetric{Float16})
    E = eigen(X; sortby=x -> -x)
    values = convert(AbstractVector{Float16}, E.values)
    vectors = convert(AbstractMatrix{Float16}, E.vectors)
    return Eigen(values, vectors)
end

eigen_sym(X, uplo=:U) = eigen_sym(Symmetric(X, uplo))

"""
    project_psd!(X, ϵ)

Project ``X`` onto the cone of positive semi-definite matrices. This method works by
computing the eigen decomposition of ``X`` and replacing eigenvalues below a threshold with
the threshold value, and then reconstructing the matrix.
"""
function project_psd!(X::AbstractMatrix{T}, ϵ::T=zero(T)) where {T}
    ϵ = max(ϵ, zero(T))
    λ, P = eigen_sym(X)
    replace!(x -> max(x, ϵ), λ)
    X .= P * Diagonal(λ) * P'
    return X
end

function project_psd!(X::Symmetric{T}, ϵ::T=zero(T)) where {T}
    ϵ = max(ϵ, zero(T))
    λ, P = eigen_sym(X)
    replace!(x -> max(x, ϵ), λ)
    X.data .= P * Diagonal(λ) * P'
    return X
end

"""
    project_psd(X, ϵ)

Project ``X`` onto the cone of positive semi-definite matrices. This method works by
computing the eigen decomposition of ``X`` and replacing eigenvalues below a threshold with
the threshold value, and then reconstructing the matrix.
"""
function project_psd(X::AbstractMatrix{T}, ϵ::T=zero(T)) where {T}
    return project_psd!(copy(X), ϵ)
end
