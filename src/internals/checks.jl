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
function issquare(X::AbstractMatrix)
    m, n = size(X)
    return m == n
end

"""
    require_square(X)

Require that a matrix is square, then return its common dimension.
"""
function require_square(X)
    m, n = size(X)
    m == n || throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(X))"))
    return m
end

"""
    require_matrix(X)

Require that an input be an `AbstractMatrix`. Throw an error if it is not.
"""
require_matrix(@nospecialize X) = X isa AbstractMatrix ||
    throw(ArgumentError(lazy"Input required to be an AbstractMatrix, got $(typeof(X))"))

"""
    require_real(X)

Require that a matrix has real-valued elements. Throw an error if it does not.
"""
function require_real(::AbstractMatrix{T}) where {T}
    T <: Real || throw(ArgumentError(lazy"Matrix element type must be Real, got $T"))
    return nothing
end

"""
    has_unit_diagonal(X)

Test whether all the diagonal elements of a matrix are equal to 1.
"""
has_unit_diagonal(X::AbstractMatrix{T}) where {T} = all(==(one(T)), diag(X))

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
ispossemidef(X, ϵ = -sqrt(eps(eltype(X)))) = eigmin(X) ≥ ϵ

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
    X isa AbstractMatrix || return false
    issquare(X) || return false
    issymmetric(X) || return false
    has_unit_diagonal(X) || return false
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
