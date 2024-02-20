_clampcor(x::Real) = clamp(x, -one(x), one(x))


function _diagonals_are_one(X::AbstractMatrix{T})  where {T<:Real}
    return all(==(one(T)), diag(X))
end


function _constrained_to_pm_one(X::AbstractMatrix{T}) where {T<:Real}
    return all(-one(T) .≤ X .≤ one(T))
end


function _is_correlation(X::AbstractMatrix{<:AbstractFloat})
    issymmetric(X)            || return false
    _diagonals_are_one(X)     || return false
    _constrained_to_pm_one(X) || return false
    isposdef(X)               || return false

    return true
end


function _is_square(X::AbstractMatrix)
    m, n = size(X)
    return m == n
end


function _set_diag!(X::AbstractMatrix{T}, v::T) where T
    _is_square(X) || throw(DimensionMismatch("Matrix must be square."))

    for i in diagind(X)
        X[i] = v
    end

    X
end


"""
Copy the upper triangle of a matrix to the lower triangle.
"""
function _copytolower!(X::AbstractMatrix{T}) where T
    nr, nc = size(X)
    nr == nc || throw(DimensionMismatch("Matrix must be square."))

    for j in 1:nc-1
        for i in j+1:nr
            @inbounds X[i,j] = X[j,i]
        end
    end

    X
end

_copytolower!(X::Symmetric) = X


function _cor_constrain!(X::AbstractMatrix{T}) where {T<:Real}
    X .= _clampcor.(X)
    _copytolower!(X)
    _set_diag!(X, one(eltype(X)))
    return X
end


function _cov2cor!(X::AbstractMatrix{T}) where {T<:Real}
    D = sqrt(inv(Diagonal(X)))
    X .= D * X * D
    _cor_constrain!(X)
    return X
end

function _cov2cor!(X::Symmetric)
    D = sqrt(inv(Diagonal(X)))
    X.data .= D * X * D
    _cor_constrain!(X)
    return X
end


function _prep_matrix!(R::AbstractMatrix{T}) where {T<:Real}
    _is_square(R) || throw(DimensionMismatch("The input matrix must be square."))

    if !issymmetric(R)
        _copytolower!(R)
    end

    if !_diagonals_are_one(R)
        _set_diag!(R, one(T))
    end

    return size(R, 1)
end


function _eigen_reversed(X)
    λ, P = eigen(X)
    reverse!(λ)
    reverse!(P; dims=2)
    return λ, P
end
