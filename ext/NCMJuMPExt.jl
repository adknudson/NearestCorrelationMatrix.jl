module NCMJuMPExt

import NearestCorrelationMatrix as NCM
using NearestCorrelationMatrix: build_ncm_solution

using JuMP
using LinearAlgebra

export JuMPAlgorithm

"""
    JuMPAlgorithm(optimizer)

An algorithm that uses the JuMP interface to set up and solve the nearest correlation matrix
problem. The optimizer can be any that supports semi-definite programming (see the
[supported solvers](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers)).

## Supported Optimizers

This list is not exhaustive, as we've only tested the freely available optimizers listed in
the JuMP docs that specify support for semidefinite programming (SDP). Compatible optimizers
must also support the `PSD` constraint.

- Clarabel.jl
- COSMO.jl
- ProxSDP.jl
- SCS.jl
- Hypatia.jl
- Pajarito.jl

In our performance testing, COSMO is the fastest optimizer, with SCS being the next fastest.
We also recommend passing `"rho=>1.0"` to the COSMO optimizer to speed up convergence.

## Examples

```julia
using NearestCorrelationMatrix
using JuMP, Clarabel, COSMO, ProxSDP, SCS, Hypatia, Pajarito, HiGHS

# SCS.jl
alg = JuMPAlgorithm(SCS.Optimizer)

# COSMO.jl
alg = JuMPAlgorithm(COSMO.Optimizer)
opt = optimizer_with_attributes(COSMO.Optimizer, "rho" => 1.0)
alg = JuMPAlgorithm(opt)

# Pajarito.jl
opt = optimizer_with_attributes(
    Pajarito.Optimizer,
    "oa_solver" => optimizer_with_attributes(
        HiGHS.Optimizer,
        MOI.Silent() => true,
        "mip_feasibility_tolerance" => 1e-8,
        "mip_rel_gap" => 1e-6,
    ),
    "conic_solver" => optimizer_with_attributes(
        Hypatia.Optimizer,
        MOI.Silent() => true
    )
)
alg = JuMPAlgorithm(opt)
```
"""
struct JuMPAlgorithm{O, A, K} <: NCM.NCMAlgorithm
    optimizer::O
    args::A
    kwargs::K

    function JuMPAlgorithm(optimizer, args...; kwargs...)
        return new{typeof(optimizer), typeof(args), typeof(kwargs)}(
            optimizer, args, kwargs
        )
    end
end

NCM.default_iters(::JuMPAlgorithm, ::Any) = 0
NCM.supports_float16(::JuMPAlgorithm) = true
NCM.modifies_in_place(::JuMPAlgorithm) = false
NCM.supports_symmetric(::JuMPAlgorithm) = true
NCM.supports_parameterless_construction(::Type{JuMPAlgorithm}) = false

function NCM.solve!(solver::NCMSolver, alg::JuMPAlgorithm)
    model = solver.cacheval

    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model) && solver.verbose
        @warn "The model was not solved correctly"
    end

    X = JuMP.value(model[:X])

    return build_ncm_solution(alg, X, nothing, solver)
end

function NCM.init_cacheval(
        alg::JuMPAlgorithm, A; kwargs...
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
