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

function DirectProjection(args...; tau::Real=eps(), kwargs...)
    return DirectProjection(tau, args, kwargs)
end

default_iters(::DirectProjection) = 0

autotune(dp::Type{DirectProjection}, prob::NCMProblem) = autotune(dp, prob.A)
autotune(::Type{DirectProjection}, A::AbstractMatrix{Float64}) = Newton(tau=1e-12)
autotune(::Type{DirectProjection}, A::AbstractMatrix{Float32}) = Newton(tau=1e-8)
autotune(::Type{DirectProjection}, A::AbstractMatrix{Float16}) = Newton(tau=1e-4)

function CommonSolve.solve!(solver::NCMSolver, alg::DirectProjection; kwargs...)
    X = solver.A
    T = eltype(X)
    tau = max(T(alg.tau), eps(T))

    X[diagind(X)] .-= tau
    project_psd!(X, eps(T))
    X[diagind(X)] .+= tau

    cov2cor!(X)

    return build_ncm_solution(alg, X, nothing, solver; iters = 1)
end
