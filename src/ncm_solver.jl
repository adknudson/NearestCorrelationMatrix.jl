"""
    NCMSolver

Common interface for solving NCM problems. Algorithm-specific cache is stored in the
`cacheval` field.

- `A`: the input matrix
- `p`: the parameters (currently unused)
- `alg`: the NCM algorithm to use
- `cacheval`: algorithm-specific cache
- `isfresh`: `true` if the cacheval hasn't been set yet
- `abstol`:
- `reltol`:
- `maxiters`:
- `verbose`:
"""
mutable struct NCMSolver{TA, P, Talg, Tc, Ttol}
    A::TA         # the input matrix
    p::P          # parameters
    alg::Talg     # ncm algorithm
    cacheval::Tc  # store algorithm cache here
    isfresh::Bool # false => cacheval is set wrt A, true => update cacheval wrt A
    abstol::Ttol  # absolute tolerance for convergence
    reltol::Ttol  # relative tolerance for convergence
    maxiters::Int # maximum number of iterations
    verbose::Bool
end



function CommonSolve.init(prob::NCMProblem, alg::NCMAlgorithm, args...;
    alias_A = default_alias_A(alg, prob.A),
    abstol = default_tol(real(eltype(prob.A))),
    reltol = default_tol(real(eltype(prob.A))),
    maxiters::Int = default_iters(alg, prob.A),
    verbose::Bool = false,
    kwargs...
)
    @unpack A, p = prob

    A = if alias_A
        verbose && println("Aliasing A")
        A
    elseif A isa Matrix
        verbose && println("Creating a copy of A")
        copy(A)
    else
        verbose && println("Creating a deep copy of A")
        deepcopy(A)
    end

    # Guard against type mismatch for user-specified reltol/abstol
    reltol = real(eltype(A))(reltol)
    abstol = real(eltype(A))(abstol)

    cacheval = init_cacheval(alg, A, maxiters, abstol, reltol, verbose)
    isfresh = true
    Tc = typeof(cacheval)

    solver = NCMSolver{typeof(A), typeof(p), typeof(alg), Tc, typeof(reltol)}(
        A, p, alg, cacheval, isfresh, abstol, reltol, maxiters, verbose)

    return solver
end



function CommonSolve.init(prob::NCMProblem, args...; kwargs...)
    return init(prob, nothing, args...; kwargs...)
end



function CommonSolve.init(prob::NCMProblem, alg::Nothing, args...; kwargs...)
    return init(prob, default_alg(prob.A), args...; kwargs...)
end



function CommonSolve.step!(solver::NCMSolver, args...; kwargs...)
    return step!(solver, solver.alg, args...; kwargs...)
end



"""
    residual(solver, alg)

Calculates the residual of the solver.
"""
residual(::NCMSolver, ::NCMAlgorithm) = nothing
