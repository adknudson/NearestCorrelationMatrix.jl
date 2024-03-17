module NCMJuMPCOSMOExt


using NearestCorrelationMatrix

using JuMP
using COSMO
using LinearAlgebra


export JuMP_COSMO


struct JuMP_COSMO <: NCMAlgorithm end

NearestCorrelationMatrix.default_iters(::JuMP_COSMO, ::Any) = 0

function NearestCorrelationMatrix.solve!(solver::NCMSolver, alg::JuMP_COSMO)
    A = solver.A
    n = size(A, 1)
    q = -vec(A)
    r = vec(A)' * vec(A) / 2
    m = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => false))
    @variable(m, X[1:n, 1:n], PSD)
    x = vec(X)
    @objective(m, Min, x' * x / 2 + q' * x + r)

    for i = 1:n
        @constraint(m, X[i, i] == one(T))
    end

    JuMP.optimize!(m)

    return NearestCorrelationMatrix.build_ncm_solution(alg, JuMP.value(X), nothing, solver)
end


end
