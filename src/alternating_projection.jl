struct AlternatingProjection <: NearestCorrelationAlgorithm 
    maxiter::Int
    tol::Float64

    function AlternatingProjection(; maxiter=100, tol=1e-7)
        return new(maxiter, tol)
    end
end


function _getAplus(A::Matrix{T}) where {T<:AbstractFloat}
    λ, P = eigen(Symmetric(A))
    λ[λ .< zero(T)] .= zero(T)
    return P * Diagonal(λ) * transpose(P)
end


function _getPs!(X::Matrix{T}, A::Matrix{T}, W::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    W05 = sqrt(W)
    X .= inv(W05) * _getAplus(W05 * A * W05) * inv(W05)
    return X
end


function _getPu!(A::Matrix{T}, X::Matrix{T}, W::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    copyto!(A, X)
    A[diagind(A)] .= diag(W)
    return A
end


function _nearest_cor!(A::Matrix{T}, alg::AlternatingProjection) where {T<:AbstractFloat}
    n = _prep_matrix!(A)

    tol = T(alg.tol)

    W = Diagonal(ones(T, n))
    Δ = zeros(T, n, n)
    X = zeros(T, n, n)
    R = zeros(T, n, n)

    i = 0
    converged = false
    conv = typemax(T)

    while i < alg.maxiter && !converged
        R .= A - Δ
        _getPs!(X, R, W)
        Δ .= X - R
        _getPu!(A, X, W)

        conv = norm(A - X) / norm(A)
        converged = conv ≤ tol
        i += 1
    end

    return A
end