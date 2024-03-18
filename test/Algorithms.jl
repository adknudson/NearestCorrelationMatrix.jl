using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix: supports_float16
using NearestCorrelationMatrix.Internals
using LinearAlgebra: isposdef, Symmetric


macro test_iscorrelation(ex)
    return quote
        @test issquare($(esc(ex)))
        @test issymmetric($(esc(ex)))
        @test diagonals_are_one($(esc(ex)))
        @test constrained_to_pm_one($(esc(ex)))
        @test ispossemidef($(esc(ex)))
    end
end


function test_simple(algtype)
    @testset "$algtype" begin
        r0 = get_negdef_matrix(Float64)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        cache = init(prob, alg)
        sol = solve!(cache)

        @test_iscorrelation sol.X
        @test isposdef(sol.X) == true


        # Handle Symmetric type matrices
        r0 = get_negdef_matrix(Float64)
        prob = NCMProblem(Symmetric(r0))
        alg = autotune(algtype, prob)
        @test_nothrow solve(prob, alg)


        # Handle Float16 input matrices
        r0 = get_negdef_matrix(Float16)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        if supports_float16(alg)
            @test_nothrow solve(prob, alg)
        else
            @test_throws Exception solve(prob, alg)
            @test_nothrow solve(prob, alg; convert_f16=true)
        end
    end
end


@testset verbose=true "Simple Tests" begin
    test_simple(Newton)
    test_simple(DirectProjection)
    test_simple(AlternatingProjections)
end