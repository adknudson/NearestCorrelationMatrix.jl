module NCMJuMPExt

using NearestCorrelationMatrix
using NearestCorrelationMatrix: build_ncm_solution
using NearestCorrelationMatrix.Internals: project_psd!, cov2cor!

using JuMP
using LinearAlgebra

NearestCorrelationMatrix.default_iters(::JuMPAlgorithm, ::Any) = 0
NearestCorrelationMatrix.supports_float16(::JuMPAlgorithm) = true
NearestCorrelationMatrix.modifies_in_place(::JuMPAlgorithm) = false
NearestCorrelationMatrix.supports_symmetric(::JuMPAlgorithm) = true
NearestCorrelationMatrix.supports_parameterless_construction(::Type{JuMPAlgorithm}) = false

function NearestCorrelationMatrix.solve!(solver::NCMSolver, alg::JuMPAlgorithm)
    model = solver.cacheval

    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model) && solver.verbose
        @warn "The model was not solved correctly"
    end

    X = JuMP.value(model[:X])
    project_psd!(X)
    cov2cor!(X)

    return build_ncm_solution(alg, X, nothing, solver)
end

function NearestCorrelationMatrix.init_cacheval(
    alg::JuMPAlgorithm, A, maxiters, abstol, reltol, verbose
)
    n = size(A, 1)

    model = JuMP.Model(alg.optimizer)

    @variable(model, X[1:n, 1:n], PSD)

    v = vec(A)
    q = -v
    r = 0.5 * dot(v, v)
    x = vec(X)
    @objective(model, Min, 0.5 * dot(x, x) + dot(q, x) + r)

    for i in 1:n
        @constraint(model, X[i, i] == 1.0)
    end

    return model
end

end
