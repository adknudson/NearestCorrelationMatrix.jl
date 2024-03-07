"""
    AlternatingProjection(; maxiter=100, tol=1e-6)

The alternating projections algorithm developed by Nick Higham.
"""
Base.@kwdef struct AlternatingProjection <: NearestCorrelationAlgorithm
    maxiter::Int = 100
    tol::Real = 1e-8
end


function ncm!(A::AbstractMatrix{T}, alg::AlternatingProjection) where {T<:AbstractFloat}
    checkmat!(A)
    n = size(A, 1)

    tol = max(T(alg.tol), eps(T))

    W = Diagonal(ones(T, n))
    Whalf = sqrt(W)
    Whalfinv = inv(Whalf)

    ΔS = zeros(T, n, n)
    Y = copy(A)
    X = similar(A)
    R = similar(A)

    Xold = similar(X)
    Yold = similar(Y)

    i = 0
    converged = false

    while i < alg.maxiter && !converged
        Xold .= X
        Yold .= Y

        R .= Y - ΔS
        X .= _project_s(R, Whalf, Whalfinv)
        ΔS .= X - R
        Y .= _project_u(X)

        rel_x = norm(X - Xold, Inf) / norm(X, Inf)
        rel_y = norm(Y - Yold, Inf) / norm(Y, Inf)
        rel_yx = norm(Y - X, Inf) / norm(Y, Inf)

        converged = max(rel_x, rel_y, rel_yx) ≤ tol
        i += 1
    end

    copyto!(A, Y)
    force_pd!(A)
    cov2cor!(A)
    return A
end



"""
Project `X` onto the set of symmetric positive semi-definite matrices with a W-norm.
"""
function _project_s(
    X::AbstractMatrix{T},
    Whalf::AbstractMatrix{T},
    Whalfinv::AbstractMatrix{T}
) where {T<:AbstractFloat}
    Y = Whalfinv * project_psd(Whalf * X * Whalf) * Whalfinv
    return Symmetric(Y)
end


"""
Project X onto the set of symmetric matrices with unit diagonal.
"""
function _project_u(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    Y = copy(X)
    setdiag!(Y, one(T))
    return Symmetric(Y)
end
