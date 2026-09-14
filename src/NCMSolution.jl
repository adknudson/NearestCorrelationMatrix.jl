"""
    NCMSolution(X, resid, alg, iters, solver, stats)

Representation of the solution to an NCM problem defined by a `NCMProblem`

# Fields

- `X`: The solution to the NCM problem.
- `resid`: The residual of the solver.
- `alg`: The algorithm used by the solver.
- `iters`: The number of iterations used to solve the NCM problem.
- `solver`: The `NCMSolver` object containing the solver's internal cached variables.
- `stats`: Statistics of the solver.
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
    build_ncm_solution(alg, X, resid, solver; iters = 0, stats = nothing)

Build the NCMSolution object from the given arguments.
"""
function build_ncm_solution(alg, X, resid, solver; iters = 0, stats = nothing)
    Y = Symmetric(X)
    return NCMSolution{typeof(Y), typeof(resid), typeof(alg), typeof(solver), typeof(stats)}(
        Y, resid, alg, iters, solver, stats
    )
end

"""
    solve!(solver, args...; kwargs...)

Solve the initialized NCM problem.
"""
function CommonSolve.solve!(solver::NCMSolver, args...; kwargs...)
    verbose = solver.verbose

    verbose && println("Beginning solve...")

    # solve! must dispatch on both the solver and the algorithm
    sol = solve!(solver, solver.alg; kwargs...)

    verbose && println("Finished solving...")

    if solver.ensure_pd
        verbose && println("Checking that the solution matrix is positive definite")
        δ = solver.min_eigenvalue
        δ = max(δ, eps(eltype(sol.X)))

        attempt = 0

        while attempt < solver.max_pd_attempts
            if isposdef(sol.X)
                verbose && println("Solution matrix is positive definite")
                break
            end

            λpre = eigmin(sol.X)

            if solver.mask === nothing
                project_psd!(sol.X, δ)
                cov2cor!(sol.X)
            else
                # Strict PD and exact fixed-element feasibility cannot both be guaranteed: repairing
                # definiteness perturbs every entry, so re-apply the mask afterwards. The fixed elements
                # (and unit diagonal) take precedence.
                project_psd!(sol.X, δ)
                project_fixed!(sol.X, solver.A_orig, solver.mask)
                project_unit!(sol.X)
            end

            λpost = eigmin(sol.X)
            attempt += 1

            verbose && println("Attempt=$attempt, δ=$δ, λ_min_pre=$λpre, λ_min_post=$λpost")

            # if δ∈(0, 1), then repeatedly applying `√` causes δᵢ to converge to 1.
            δ = sqrt(δ)
        end
    end

    return sol
end
