"""
    AlternatingProjection(; maxiter=100, tol=1e-6)

The alternating projections algorithm developed by Nick Higham.
"""
Base.@kwdef struct AlternatingProjection <: NearestCorrelationAlgorithm
    maxiter::Int = 100
    tol::Real = 1e-6
end


function _nearest_cor!(X::AbstractMatrix{T}, alg::AlternatingProjection) where {T<:AbstractFloat}
    checkmat!(X)
    n = size(X, 1)

    tol = T(alg.tol)

    W = Diagonal(ones(T, n))
    Sk = zeros(T, n, n)
    Rk = zeros(T, n, n)
    Xk = zeros(T, n, n)
    Yk = copy(X)

    i = 0
    converged = false
    conv = typemax(T)

    while i < alg.maxiter && !converged
        Rk .= Yk - Sk
        Xk .= _project_symmetric(Rk, W)
        Sk .= Xk - Rk
        Yk .= _project_unitdiag(Xk)

        conv = norm(Yk - Xk) / norm(Yk)
        converged = conv ≤ tol && isposdef(Yk)
        i += 1
    end

    X .= Yk
    cov2cor!(X)
    return X
end



function _project_symmetric(A::AbstractMatrix{T}, W::Diagonal{T}) where {T<:AbstractFloat}
    W05 = sqrt(W)
    return inv(W05) * project_psd(W05 * A * W05) * inv(W05)
end


function _project_unitdiag(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    Y = copy(X)
    setdiag!(Y, one(T))
    return Y
end
