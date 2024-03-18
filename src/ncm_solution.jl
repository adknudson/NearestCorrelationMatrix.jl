"""
    NCMSolution
"""
struct NCMSolution{T, R, A, C, S}
    X::T
    resid::R
    alg::A
    iters::Int
    solver::C
    stats::S
end


"""
    build_ncm_solution(args...; kwargs...)

Build the NCMSolution.
"""
function build_ncm_solution(alg, X, resid, solver; iters = 0, stats = nothing)
    return NCMSolution{typeof(X), typeof(resid), typeof(alg), typeof(solver), typeof(stats)}(
        X,
        resid,
        alg,
        iters,
        solver,
        stats)
end


"""
    solve(prob, alg)

Solve the NCM problem with the given algorithm.
"""
function CommonSolve.solve(prob::NCMProblem, alg::NCMAlgorithm, args...; kwargs...)
    return solve!(init(prob, alg, args...; kwargs...))
end

"""
    solve(prob)

Solve the NCM problem with the default algorithm.
"""
function CommonSolve.solve(prob::NCMProblem, args...; kwargs...)
    return solve(prob, nothing, args...; kwargs...)
end

"""
    solve(prob, nothing)

Solve the NCM problem with the default algorithm.
"""
function CommonSolve.solve(prob::NCMProblem, ::Nothing, args...; kwargs...)
    return solve!(init(prob, nothing, args...; kwargs...))
end


"""
    solve!(solver)

Solve the initialized NCM problem.
"""
function CommonSolve.solve!(solver::NCMSolver, args...; kwargs...)
    # Algorithms will dispatch on this method signature
    return solve!(solver, solver.alg, args...; kwargs...)
end
