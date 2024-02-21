_clampcor(x::Real) = clamp(x, -one(x), one(x))


function _is_square(X::AbstractMatrix)
    m, n = size(X)
    return m == n
end


function _diagonals_are_one(X::AbstractMatrix{T})  where {T<:Real}
    return all(==(one(T)), diag(X))
end


function _constrained_to_pm_one(X::AbstractMatrix{T}) where {T<:Real}
    return all(-one(T) .≤ X .≤ one(T))
end


function _is_correlation(X::AbstractMatrix{T}) where {T<:Real}
    issymmetric(X)            || return false
    _diagonals_are_one(X)     || return false
    _constrained_to_pm_one(X) || return false
    isposdef(X)               || return false
    return true
end


function _set_diag!(X::AbstractMatrix{T}, v::T) where T
    _is_square(X) || throw(DimensionMismatch("Matrix must be square."))
    for i in diagind(X)
        X[i] = v
    end
    X
end


function _copytolower!(X::AbstractMatrix{T}) where T
    _is_square(X) || throw(DimensionMismatch("Matrix must be square."))
    nr, nc = size(X)
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
    _set_diag!(X, one(T))
    return X
end

function _cor_constrain!(X::Symmetric{T}) where {T<:Real}
    X.data .= _clampcor.(X)
    _set_diag!(X.data, one(T))
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


function _prep_matrix!(X::AbstractMatrix{T}) where {T<:Real}
    _is_square(X) || throw(DimensionMismatch("The input matrix must be square."))

    if !issymmetric(X)
        _copytolower!(X)
    end

    if !_diagonals_are_one(X)
        _set_diag!(X, one(T))
    end

    return X
end


function _eigen_reversed(X)
    λ, P = eigen(X)
    reverse!(λ)
    reverse!(P; dims=2)
    return λ, P
end
