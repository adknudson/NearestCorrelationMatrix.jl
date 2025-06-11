"""
    NCMSolution(x, resid, alg, iters, solver, stats)

Representation of the solution to an NCM problem defined by a `NCMProblem`

## Fields

- `X`: The solution to the NCM problem.
- `resid`: The residual of the solver.
- `alg`: The algorithm used by the solver.
- `iters`: The number of iterations used to solve the NCM problem.
- `solver`: The `NCMSolver` object containing the solver's internal cached variables.
- `stats`: Statistics of the solver.
"""
struct NCMSolution{T,R,A,C,S}
    X::T
    resid::R
    alg::A
    iters::Int
    solver::C
    stats::S
end

"""
    build_ncm_solution(alg, X, resid, solver; iters = 0, stats = nothing)

Build the NCMSolution object from the given arguments.
"""
function build_ncm_solution(alg, X, resid, solver; iters=0, stats=nothing)
    Y = Symmetric(X)
    return NCMSolution{typeof(Y),typeof(resid),typeof(alg),typeof(solver),typeof(stats)}(
        Y, resid, alg, iters, solver, stats
    )
end

"""
    solve(prob, alg, args...; kwargs...)

Solve the NCM problem with the given algorithm.
"""
function CommonSolve.solve(prob::NCMProblem, alg::NCMAlgorithm, args...; kwargs...)
    return solve!(init(prob, alg, args...; kwargs...))
end

"""
    solve(prob, algtype, args...; kwargs...)

Solve the NCM problem with the given algorithm type. 
The algorithm will be autotuned to the problem.
"""
function CommonSolve.solve(
    prob::NCMProblem, algtype::Type{<:NCMAlgorithm}, args...; kwargs...
)
    return solve!(init(prob, algtype, args...; kwargs...))
end

"""
    solve(prob, args...; kwargs...)

Solve the NCM problem with the default algorithm.
"""
function CommonSolve.solve(prob::NCMProblem, args...; kwargs...)
    return solve(prob, nothing, args...; kwargs...)
end

"""
    solve(prob, nothing, args...; kwargs...)

Solve the NCM problem with the default algorithm.
"""
function CommonSolve.solve(prob::NCMProblem, ::Nothing, args...; kwargs...)
    return solve!(init(prob, nothing, args...; kwargs...))
end

"""
    solve!(solver, args...; kwargs...)

Solve the initialized NCM problem.
"""
function CommonSolve.solve!(solver::NCMSolver, args...; kwargs...)
    sol = solve!(solver, solver.alg, args...; kwargs...)

    if sol.solver.ensure_pd && !isposdef(sol.X)
        project_psd!(sol.X, sqrt(eps(eltype(sol.X))))
        cov2cor!(sol.X)
    end

    return sol
end
