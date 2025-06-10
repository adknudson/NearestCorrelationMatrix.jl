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

    R = similar(Y)
    X = similar(Y)
    S = zeros(T, n, n)

    i = 0
    resid = Inf

    while i < solver.maxiters && resid ≥ solver.reltol
        @. R = Y - S
        X .= project_s(R)
        @. S = X - R
        Y .= project_u(X)

        # ref (4.1). Only need ‖Y-X‖/‖Y‖, and don't need ‖Xₖ-Xₖ₋₁‖\‖Xₖ‖ nor ‖Yₖ-Yₖ₋₁‖\‖Yₖ‖
        resid = norm(Y .- X, Inf) / norm(Y, Inf)

        i += 1
    end

    # do one more projection to ensure PSD
    tau = convert(T, alg.tau)
    project_psd!(Y, tau)
    cov2cor!(Y)
    i += 1

    return build_ncm_solution(alg, Y, resid, solver; iters=i)
end
