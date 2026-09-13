using Test
using NearestCorrelationMatrix
using JuMP, COSMO

include("Datasets.jl")
using .Datasets

include("CustomTestMacros.jl")
using .CustomTestMacros

r0 = default_negdef(Float64)
prob = NCMProblem(r0)

@test_isdefined JuMPAlgorithm
@test_isimplemented JuMPAlgorithm(COSMO.Optimizer)

@test_throws Exception autotune(JuMPAlgorithm, prob)

optimizer = optimizer_with_attributes(
    COSMO.Optimizer, MOI.Silent() => true, "rho" => 1.0
)
alg = JuMPAlgorithm(optimizer)

@test_nothrow solve(prob, alg)
