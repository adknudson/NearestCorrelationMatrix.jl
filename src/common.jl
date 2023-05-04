_diagonals_are_one(X::AbstractMatrix{<:AbstractFloat}) = all(==(one(eltype(X))), diag(X))


_constrained_to_pm_one(X::AbstractMatrix{<:AbstractFloat}) = all(-one(eltype(X)) .≤ X .≤ one(eltype(X)))


function _iscorrelation(X::AbstractMatrix{<:AbstractFloat})
    return issymmetric(X) && _diagonals_are_one(X) && _constrained_to_pm_one(X) && isposdef(X)
end


_issquare(X::AbstractMatrix) = begin m, n = size(X); return m == n end


function _set_diag!(X::AbstractMatrix{T}, v::T) where T
    _issquare(X) || throw(DimensionMismatch("Matrix must be square."))

    for i in diagind(X)
        X[i] = v
    end

    X
end


function _make_symmetric!(X::AbstractMatrix)
    nr, nc = size(X)
    nr == nc || throw(DimensionMismatch("Matrix must be square."))

    for j in 1:nc-1
        for i in j+1:nr
            X[i,j] = X[j,i]
        end
    end

    X
end


function _cor_constrain!(X::AbstractMatrix{<:AbstractFloat})
    X .= clampcor.(X)
    _make_symmetric!(X)
    _set_diag!(X, one(eltype(X)))
    return X
end


function _cov2cor!(X::AbstractMatrix{<:AbstractFloat})
    D = sqrt(inv(Diagonal(X)))
    X .= D * X * D
    _cor_constrain!(X)
    return X
end


function _prep_matrix!(R::AbstractMatrix{T}) where {T<:AbstractFloat}
    _issquare(R) || throw(DimensionMismatch("The input matrix must be square."))
        
    if !issymmetric(R)
        @warn "The input matrix is not symmetric. Using the upper triangle to create a symmetric view."
        _make_symmetric!(R)
    end

    if !_diagonals_are_one(R)
        @warn "The diagonal elements are not all equal to 1. Explicitly setting the values to 1."
        _set_diag!(R, one(T))
    end

    return size(R, 1)
end