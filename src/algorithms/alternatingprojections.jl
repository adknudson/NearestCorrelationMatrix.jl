"""
    AlternatingProjections(; tau=0)

The alternating projections algorithm developed by Nick Higham.
"""
struct AlternatingProjections{A,K} <: NCMAlgorithm
    tau::Real
    args::A
    kwargs::K
end

function AlternatingProjections(args...; tau::Real=0, kwargs...)
    return AlternatingProjections(tau, args, kwargs)
end

default_iters(::AlternatingProjections, A) = clamp(size(A, 1), 20, 200)
modifies_in_place(::AlternatingProjections) = true
supports_float16(::AlternatingProjections) = true
supports_symmetric(::AlternatingProjections) = false
supports_parameterless_construction(::Type{AlternatingProjections}) = true

function autotune(::Type{AlternatingProjections}, prob::NCMProblem)
    return AlternatingProjections(; tau=eps(eltype(prob.A)))
end

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

    # do one more projection to ensure PSD
    tau = convert(T, alg.tau)
    project_psd!(Y, tau)
    cov2cor!(Y)
    i += 1

    return build_ncm_solution(alg, Y, resid, solver; iters=i)
end
