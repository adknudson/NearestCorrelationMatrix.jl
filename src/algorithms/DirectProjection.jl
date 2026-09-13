"""
    DirectProjection(args...; tau=0, kwargs...)

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

function DirectProjection(args...; tau::Real = 0, kwargs...)
    return DirectProjection(tau, args, kwargs)
end

modifies_in_place(::DirectProjection) = true
supports_symmetric(::DirectProjection) = true
supports_float16(::DirectProjection) = true
supports_parameterless_construction(::Type{DirectProjection}) = true

autotune(::Type{DirectProjection}, prob::NCMProblem) = _autotune(DirectProjection, prob.A)

function _autotune(::Type{DirectProjection}, ::AbstractMatrix{Float64})
    return DirectProjection(; tau = 1.0e-12)
end

function _autotune(::Type{DirectProjection}, A::AbstractMatrix{Float32})
    n = size(A, 1)

    tau = if n ≤ 25
        1.0e-6
    elseif n ≤ 100
        5.0e-6
    elseif n ≤ 500
        1.0e-5
    else
        5.0e-5
    end

    return DirectProjection(; tau = tau)
end

function _autotune(::Type{DirectProjection}, A::AbstractMatrix{Float16})
    n = size(A, 1)

    tau = if n ≤ 10
        5.0e-3
    elseif n ≤ 25
        1.0e-2
    elseif n ≤ 50
        2.5e-2
    else
        5.0e-2
    end

    return DirectProjection(; tau = tau)
end

function CommonSolve.solve!(solver::NCMSolver, alg::DirectProjection; kwargs...)
    A = solver.A
    X = copy(A)
    tau = convert(eltype(X), alg.tau)

    project_psd!(X, tau)
    cov2cor!(X)

    resid = norm(X .- A) / norm(X)

    return build_ncm_solution(alg, X, resid, solver; iters = 1)
end
