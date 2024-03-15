"""
    DirectProjection(; tau=sqrt(eps()))

Single step projection of the input matrix into the set of correlation matrices. Useful when
a "close" correlation matrix is needed without concern for it being the most optimal.

# Parameters
- `τ`: a tuning parameter controlling the smallest eigenvalue of the resulting matrix
"""
struct DirectProjection{A, K} <: NCMAlgorithm
    tau::Real
    args::A
    kwargs::K
end

function DirectProjection(args...; tau::Real=sqrt(eps()), kwargs...)
    return DirectProjection(tau, args, kwargs)
end

default_iters(::DirectProjection) = 0

function CommonSolve.solve!(solver::NCMSolver, alg::DirectProjection; kwargs...)
    X = solver.A
    T = eltype(X)
    tau = max(T(alg.tau), eps(T))

    X[diagind(X)] .-= tau
    project_psd!(X)
    X[diagind(X)] .+= tau
    cov2cor!(X)

    return build_ncm_solution(alg, X, nothing, solver; iters = 1)
end
