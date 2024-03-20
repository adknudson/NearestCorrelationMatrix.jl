using LinearAlgebra: diag, eigmin


export
    issquare,
    require_square,
    require_matrix,
    require_real,
    diagonals_are_one,
    constrained_to_pm_one,
    ispossemidef,
    isprecorrelation,
    iscorrelation


function issquare(X::AbstractMatrix)
    m, n = size(X)
    return m == n
end


function require_square(X::AbstractMatrix)
    issquare(X) || throw_square()
end

@noinline throw_square() = throw(DimensionMismatch("Matrix required to be square"))


function require_matrix(::T) where T
    throw(ArgumentError("Input required to be an AbstractMatrix. Got $T instead"))
end

require_matrix(::AbstractMatrix) = nothing


function require_real(::AbstractMatrix{T}) where T
    throw(DomainError(T, "Input matrix is required to have real values"))
end

require_real(::AbstractMatrix{<:Real}) = nothing


function diagonals_are_one(X::AbstractMatrix{T}) where {T<:Real}
    return all(==(one(T)), diag(X))
end


function constrained_to_pm_one(X::AbstractMatrix{T}) where {T<:Real}
    return all(x -> -one(T) ≤ x ≤ one(T), X)
end


ispossemidef(X, ϵ=-sqrt(eps(eltype(X)))) = eigmin(X) ≥ ϵ


"""
    isprecorrelation(X)

Test that a matrix passes all the pre-qualifications to be a correlation matrix.

A pre-correlation matrix must:

- be square
- be symmetric
- be constrained to ±1
- have diagonals equal to 1
"""
function isprecorrelation(X::AbstractMatrix{T}) where {T}
    issquare(X)              || return false
    issymmetric(X)           || return false
    diagonals_are_one(X)     || return false
    constrained_to_pm_one(X) || return false
    return true
end


"""
    iscorrelation(X)

Test that a matrix passes all the qualifications to be a correlation matrix including being
positive (semi) definite.

A correlation matrix must:

- be square
- be symmetric
- be constrained to ±1
- have diagonals equal to 1
- be positive definite
"""
function iscorrelation(X::AbstractMatrix{T}) where {T<:Real}
    return isprecorrelation(X) && ispossemidef(X)
end
