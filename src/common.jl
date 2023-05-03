_diagonals_are_one(X::Matrix{<:AbstractFloat}) = all(==(one(eltype(X))), diag(X))

_constrained_to_pm_one(X::Matrix{<:AbstractFloat}) = all(-one(eltype(X)) .≤ X .≤ one(eltype(X)))

function _iscorrelation(X::Matrix{<:AbstractFloat})
    return issymmetric(X) && _diagonals_are_one(X) && _constrained_to_pm_one(X) && isposdef(X)
end

function _cor_constrain!(X::Matrix{<:AbstractFloat}, uplo=:U)
    X .= clampcor.(X)
    X .= Symmetric(X, uplo)
    X[diagind(X)] .= one(eltype(X))
    return X
end

function _cov2cor!(X::Matrix{<:AbstractFloat})
    D = sqrt(inv(Diagonal(X)))
    X .= D * X * D
    _cor_constrain!(X)
    return X
end