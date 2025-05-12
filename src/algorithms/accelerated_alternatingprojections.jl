"""
    AcceleratedAlternatingProjections(; tau=0)
"""
struct AcceleratedAlternatingProjections{A, K} <: NCMAlgorithm
    tau::Real
    args::A
    kwargs::K
end

function AcceleratedAlternatingProjections(args...; tau::Real=0, kwargs...)
    return AcceleratedAlternatingProjections(tau, args, kwargs)
end

default_iters(::AcceleratedAlternatingProjections, A) = clamp(size(A, 1), 10, 100)
modifies_in_place(::AcceleratedAlternatingProjections) = true
supports_float16(::AcceleratedAlternatingProjections) = true
supports_symmetric(::AcceleratedAlternatingProjections) = false
supports_parameterless_construction(::Type{AcceleratedAlternatingProjections}) = true

function autotune(::Type{AcceleratedAlternatingProjections}, prob::NCMProblem)
    return AlternatingProjections(; tau=eps(eltype(prob.A)))
end

function CommonSolve.solve!(solver::NCMSolver, alg::AcceleratedAlternatingProjections; kwargs...)
    Y = solver.A
    n = size(Y, 1)
    T = eltype(Y)
end
