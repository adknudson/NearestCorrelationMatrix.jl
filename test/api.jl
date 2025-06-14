using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: get_negdef_matrix

include("test_macros.jl")

@testset "Common Solve API" begin
    r0 = get_negdef_matrix(Float64)

    # variants of NCMProblem
    @test_isdefined NCMProblem
    @test_isimplemented NCMProblem(r0)
    prob = NCMProblem(r0)

    # variations of init
    @test_isdefined init
    @test_isimplemented init(prob)
    @test_isimplemented init(prob, Newton())
    @test_isimplemented init(prob, Newton)
    @test init(prob) isa NCMSolver

    # init with an algtype must return the correct algtype
    for algtype in (AlternatingProjections, Newton, DirectProjection)
        solver = init(prob, algtype)
        @test solver.alg isa algtype
    end

    # solve with an algtype must return the correct algtype (#31)
    for algtype in (AlternatingProjections, Newton, DirectProjection)
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
end
