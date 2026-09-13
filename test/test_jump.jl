using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: default_negdef
using JuMP, COSMO

include("CustomTestMacros.jl")
using .CustomTestMacros

A = default_negdef()
prob = NCMProblem(A)

@test_isdefined JuMPAlgorithm
@test_isimplemented JuMPAlgorithm(COSMO.Optimizer)

@test_throws Exception autotune(JuMPAlgorithm, prob)

optimizer = optimizer_with_attributes(
    COSMO.Optimizer, MOI.Silent() => true, "rho" => 1.0
)
alg = JuMPAlgorithm(optimizer)

@test_nothrow solve(prob, alg)
