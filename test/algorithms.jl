using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix: supports_float16
using NearestCorrelationMatrix: construct_algorithm, supports_parameterless_construction
using NearestCorrelationMatrix.Internals
using LinearAlgebra: issymmetric, isposdef, Symmetric

include("test_macros.jl")

function test_simple(algtype)
    @testset "$algtype" begin
        r0 = get_negdef_matrix(Float64)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        cache = init(prob, alg)
        sol = solve!(cache)

        @test_iscorrelation sol.X

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

@testset "Constructors" begin
    prob = NCMProblem(rand(4, 4))

    for algtype in (Newton, AlternatingProjections, DirectProjection)
        @test supports_parameterless_construction(algtype) == true

        alg = construct_algorithm(algtype)
        @test alg isa algtype

        alg = autotune(algtype, prob)
        @test alg isa algtype

        # supports_parameterless_construction works on the type, not the instance
        @test_throws MethodError supports_parameterless_construction(alg)
    end
end

@testset verbose = true "Simple Tests" begin
    test_simple(Newton)
    test_simple(DirectProjection)
    test_simple(AlternatingProjections)
end
