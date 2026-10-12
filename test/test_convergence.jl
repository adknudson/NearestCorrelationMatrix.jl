using Test
using InteractiveUtils
using LinearAlgebra
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: default_negdef
import NearestCorrelationMatrix as NCM

include("CustomTestMacros.jl")
using .CustomTestMacros

internal_algtypes = setdiff(subtypes(NCMAlgorithm), (JuMPAlgorithm,))

function test_simple(algtype)
    return @testset "$(NCM.alg_name(algtype))" begin
        A = default_negdef()
        prob = NCMProblem(A)
        alg = autotune(algtype, prob)
        cache = init(prob, alg)
        sol = solve!(cache)

        @test_iscorrelation sol.X

        # Issue #19: Solution must always be Symmetric
        @test sol.X isa Symmetric

        # Handle Symmetric type matrices
        A = default_negdef()
        prob = NCMProblem(Symmetric(A))
        alg = autotune(algtype, prob)
        @test_nothrow solve(prob, alg)

        # Handle Float16 input matrices
        A = default_negdef(Float16)
        prob = NCMProblem(A)
        alg = autotune(algtype, prob)
        if NCM.supports_float16(alg)
            @test_nothrow solve(prob, alg)
        else
            @test_throws Exception solve(prob, alg)
            @test_nothrow solve(prob, alg; convert_f16 = true)
        end
    end
end

for algtype in internal_algtypes
    test_simple(algtype)
end
