"""
    DirectProjection(; tau=eps())

Single step projection of the input matrix into the set of correlation matrices. Useful when
a "close" correlation matrix is needed without concern for it being the most optimal.

# Parameters
- `tau`: a tuning parameter controlling the smallest eigenvalue of the resulting matrix
"""
struct DirectProjection{A, K} <: NCMAlgorithm
    tau::Real
    args::A
    kwargs::K
end

function DirectProjection(args...; tau::Real=1e-6, kwargs...)
    return DirectProjection(tau, args, kwargs)
end

autotune(::Type{DirectProjection}, prob::NCMProblem) = _autotune(DirectProjection, prob.A)

_autotune(::Type{DirectProjection}, A::AbstractMatrix{Float64}) = DirectProjection(tau=1e-12)

function _autotune(::Type{DirectProjection}, A::AbstractMatrix{Float32})
    n = size(A, 1)

    tau = if n ≤ 25
        1e-6
    elseif n ≤ 100
        5e-6
    elseif n ≤ 500
        1e-5
    else
        5e-5
    end

    return DirectProjection(tau=tau)
end

function _autotune(::Type{DirectProjection}, A::AbstractMatrix{Float16})
    n = size(A, 1)

    tau = if n ≤ 10
        5e-3
    elseif n ≤ 25
        1e-2
    elseif n ≤ 50
        2.5e-2
    else
        5e-2
    end

    return DirectProjection(tau=tau)
end

function CommonSolve.solve!(solver::NCMSolver, alg::DirectProjection; kwargs...)
    X = solver.A
    T = eltype(X)
    tau = max(T(alg.tau), eps(T))

    project_psd!(X, tau)
    cov2cor!(X)

    return build_ncm_solution(alg, X, nothing, solver; iters = 1)
end
