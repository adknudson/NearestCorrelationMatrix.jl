using LinearAlgebra: diag, eigmin


export
    issquare,
    require_square,
    require_matrix,
    require_real,
    has_unit_diagonal,
    constrained_to_pm1,
    ispossemidef,
    isprecorrelation,
    iscorrelation


"""
    issquare(X)

Test whether a value is a square matrix.
"""
issquare(::Any) = false
issquare(X::AbstractMatrix) = ==(size(X)...)


"""
    require_square(X)

Require that a matrix is square. Throw an error if it is not.
"""
require_square(X) = issquare(X) || throw_square()
@noinline throw_square() = throw(DimensionMismatch("Matrix required to be square"))


"""
    require_matrix(X)

Require that an input be an `AbstractMatrix`. Throw an error if it is not.
"""
require_matrix(::Any) = throw_matrix()
require_matrix(::AbstractMatrix) = nothing
@noinline throw_matrix() = throw(ArgumentError("Input required to be an `AbstractMatrix`"))


"""
    require_real(X)

Require that a matrix has real-valued elements. Throw an error if it does not.
"""
require_real(::AbstractMatrix{T}) where T = throw_real(T)
require_real(::AbstractMatrix{<:Real}) = nothing
@noinline throw_real(T) = throw(DomainError(T, "Matrix required to have real values"))


"""
    has_unit_diagonal(X)

Test whether all the diagonal elements of a matrix are equal to 1.
"""
has_unit_diagonal(X::AbstractMatrix{T}) where T = all(==(one(T)), diag(X))


"""
    constrained_to_pm1(X)

Testh whether all elements of ``X`` are constrained between -1 and 1.
"""
function constrained_to_pm1(X)
    T = eltype(X)
    return all(x -> -one(T) ≤ x ≤ one(T), X)
end


"""
    ispossemidef(X, ϵ)

Test whether a matrix is positive semi-definite within machine precision.
"""
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
function isprecorrelation(X)
    X isa AbstractMatrix  || return false
    issquare(X)           || return false
    issymmetric(X)        || return false
    has_unit_diagonal(X)  || return false
    constrained_to_pm1(X) || return false
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
function iscorrelation(X)
    return isprecorrelation(X) && ispossemidef(X)
end
