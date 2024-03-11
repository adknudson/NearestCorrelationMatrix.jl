"""
    AlternatingProjections(args...; kwargs...)

The alternating projections algorithm developed by Nick Higham.
"""
struct AlternatingProjections{A, K} <: NCMAlgorithm
    args::A
    kwargs::K
end

AlternatingProjections(args...; kwargs...) = AlternatingProjections(args, kwargs)

default_iters(::AlternatingProjections, A) = 20


mutable struct AlternatingProjectionsCache{W1, W2, W3, TA, TS}
    W::W1
    WHalf::W2
    WHalfInv::W3
    X::TA
    S::TS
    R::TA
    Yold::TA
    Xold::TA
end


function init_cacheval(::AlternatingProjections, A, maxiters::Int, abstol, reltol, verbose::Bool)
    T = eltype(A)
    n = size(A, 1)

    W = Diagonal(ones(T, n))
    WHalf = sqrt(W)
    WHalfInv = inv(WHalf)

    S = zeros(T, n, n)
    X = similar(A)
    R = similar(A)

    Yold = similar(A)
    Xold = similar(A)

    return AlternatingProjectionsCache(W, WHalf, WHalfInv, X, S, R, Yold, Xold)
end



function CommonSolve.solve!(solver::NCMSolver, alg::AlternatingProjections; kwargs...)
    solver.verbose && println("Solving using Alternating Projections")

    solver.isfresh = false

    i = 0
    converged = false
    resid = Inf

    while i < solver.maxiters && !converged
        step!(solver)
        resid = residual(solver, alg)
        converged = resid ≤ solver.reltol
        i += 1
    end

    force_pd!(solver.A)
    cov2cor!(solver.A)

    return build_ncm_solution(alg, solver.A, resid, solver; iters=i)
end



function CommonSolve.step!(solver::NCMSolver, ::AlternatingProjections, args...; kwargs...)
    @unpack A = solver
    @unpack X, S, R, Yold, Xold = solver.cacheval
    @unpack WHalf, WHalfInv = solver.cacheval

    Yold .= A
    Xold .= X

    R .= A - S
    X .= _project_s(R, WHalf, WHalfInv)
    S .= X - R
    A .= _project_u(X)
end



function residual(solver::NCMSolver, ::AlternatingProjections)
    @unpack A = solver
    @unpack X, Yold, Xold = solver.cacheval

    rel_y = norm(A - Yold, Inf) / norm(A, Inf)
    rel_x = norm(X - Xold, Inf) / norm(X, Inf)
    rel_yx = norm(A - X, Inf) / norm(A, Inf)

    return max(rel_x, rel_y, rel_yx)
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
