using Test
using LinearAlgebra
using InteractiveUtils
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: default_negdef

include("CustomTestMacros.jl")
using .CustomTestMacros

internal_algtypes = setdiff(subtypes(NCMAlgorithm), (JuMPAlgorithm,))

A = default_negdef()

# variants of NCMProblem
@test_isdefined NCMProblem
@test_isimplemented NCMProblem(A)
prob = NCMProblem(A)

# variations of init
@test_isdefined init
@test_isimplemented init(prob)
@test_isimplemented init(prob, Newton())
@test_isimplemented init(prob, Newton)
@test init(prob) isa NCMSolver

# init with an algtype must return the correct algtype
for algtype in internal_algtypes
    solver = init(prob, algtype)
    @test solver.alg isa algtype
end

# solve with an algtype must return the correct algtype (#31)
for algtype in internal_algtypes
    sol = solve(prob, algtype)
    @test sol.alg isa algtype
end

# variations of solve
@test_isdefined solve
@test_isimplemented solve(prob)
@test_isimplemented solve(prob, Newton())
@test_isimplemented solve(prob, Newton)
@test solve(prob) isa NCMSolution

# variations of solve!
cache = init(prob)
@test_isdefined solve!
@test_isimplemented solve!(cache)
@test solve!(cache) isa NCMSolution

# ==> alias_A ==>
alg = AlternatingProjections # AP is known to modify in place
A = default_negdef()
prob = NCMProblem(A)

# when not aliased, A and sol.X must be different
sol = solve(prob, alg; alias_A = false)
@test !isapprox(sol.X, A)

# when aliased, A and sol.X might be same (true for AP)
sol = solve(prob, alg; alias_A = true)
@test isapprox(sol.X, A)

# test with Symmetric input
A = default_negdef()
S = Symmetric(A)
prob = NCMProblem(S)

# when not aliased, A and sol.X must be different
solver = init(prob; alias_A = false)
@test solver.A !== S
sol = solve!(solver)
@test !isapprox(sol.X, A)
@test !isapprox(sol.X, S)

# when aliased, A and sol.X might be same (true for AP)
solver = init(prob; alias_A = true)
@test solver.A === S
sol = solve!(solver)
@test_broken isapprox(sol.X, A)
@test_broken isapprox(sol.X, S)
# <== end alias_A <==
