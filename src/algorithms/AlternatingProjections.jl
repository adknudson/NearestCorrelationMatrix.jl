"""
    AlternatingProjections(; tau=0)

The alternating projections algorithm developed by Nick Higham.
"""
struct AlternatingProjections{A, K} <: NCMAlgorithm
    tau::Real
    args::A
    kwargs::K
end

function AlternatingProjections(args...; tau::Real = 0, kwargs...)
    return AlternatingProjections(tau, args, kwargs)
end

struct APCache{TA}
    scratch::TA
    X::TA
    R::TA
    S::TA
end

function init_cacheval(::AlternatingProjections, A; kwargs...)
    scratch = similar(A)
    X = similar(A)
    R = similar(A)
    S = similar(A)
    return APCache{typeof(A)}(scratch, X, R, S)
end

default_iters(::AlternatingProjections, A) = clamp(size(A, 1), 20, 200)
modifies_in_place(::AlternatingProjections) = true
supports_float16(::AlternatingProjections) = true
supports_symmetric(::AlternatingProjections) = false
supports_parameterless_construction(::Type{<:AlternatingProjections}) = true
supports_mask(::Type{<:AlternatingProjections}) = true

function autotune(::Type{<:AlternatingProjections}, prob::NCMProblem)
    T = eltype(prob.A)
    tau = sqrt(eps(T))

    # if the problem implements masking, then err on the safe side for tau
    if prob.mask !== nothing
        tau = 20 * tau
    end

    return AlternatingProjections(; tau = tau)
end

function CommonSolve.solve!(solver::NCMSolver, alg::AlternatingProjections; kwargs...)
    Y = solver.A
    tau = convert(eltype(Y), alg.tau)
    mask = solver.mask
    A_orig = solver.A_orig

    scratch = solver.cacheval.scratch
    X = solver.cacheval.X
    R = solver.cacheval.R
    ΔS = solver.cacheval.S
    fill!(ΔS, zero(eltype(ΔS)))

    iter = 0
    resid = Inf

    while iter < solver.maxiters
        iter += 1

        R .= Y .- ΔS
        project_psd!(X, R, tau, scratch)
        ΔS .= X .- R
        copyto!(Y, X)

        if mask !== nothing
            project_fixed!(Y, A_orig, mask)
        end

        # project unit after projecting fixed to ensure that the unit diagonal is preserved
        project_unit!(Y)

        resid = norm(Y .- X) / norm(Y)

        if resid ≤ solver.reltol
            break
        end
    end

    return build_ncm_solution(alg, Y, resid, solver; iters = iter)
end
