using Test
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

# alias_A must be respected
A = default_negdef()
prob = NCMProblem(A)
@test Base.mightalias(A, prob.A)
solver = init(prob; alias_A = true)
@test Base.mightalias(prob.A, solver.A)
solver = init(prob; alias_A = false)
@test !Base.mightalias(prob.A, solver.A)
S = Symmetric(A)
prob = NCMProblem(S)
@test Base.mightalias(S, prob.A)
solver = init(prob; alias_A = true)
@test Base.mightalias(prob.A, solver.A)
solver = init(prob; alias_A = false)
@test !Base.mightalias(prob.A, solver.A)
