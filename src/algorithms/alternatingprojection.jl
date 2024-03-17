"""
    AlternatingProjections(args...; kwargs...)

The alternating projections algorithm developed by Nick Higham.
"""
struct AlternatingProjections{A, K} <: NCMAlgorithm
    args::A
    kwargs::K
end

AlternatingProjections(args...; kwargs...) = AlternatingProjections(args, kwargs)

default_iters(::AlternatingProjections, A) = max(size(A,1), 20)


function CommonSolve.solve!(solver::NCMSolver, alg::AlternatingProjections; kwargs...)
    Y = solver.A
    n = size(Y, 1)
    T = eltype(Y)

    W = Diagonal(ones(T, n))
    WHalf = sqrt(W)
    WHalfInv = inv(WHalf)

    R = similar(Y)
    X = similar(Y)
    S = zeros(T, n, n)

    Yold = similar(Y)
    Xold = similar(Y)

    i = 0
    resid = Inf

    while i < solver.maxiters && resid ≥ solver.reltol
        Yold .= Y
        Xold .= X

        R .= Y - S
        X .= project_s(R, WHalf, WHalfInv)
        S .= X - R
        Y .= project_u(X)

        rel_y = norm(Y - Yold, Inf) / norm(Y, Inf)
        rel_x = norm(X - Xold, Inf) / norm(X, Inf)
        rel_yx = norm(Y - X, Inf) / norm(Y, Inf)

        resid = max(rel_x, rel_y, rel_yx)

        i += 1
    end

    cov2cor!(solver.A)

    return build_ncm_solution(alg, solver.A, resid, solver; iters=i)
end


"""
Project `X` onto the set of symmetric positive semi-definite matrices with a W-norm.
"""
function project_s(
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
function project_u(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    Y = copy(X)
    setdiag!(Y, one(T))
    return Symmetric(Y)
end
