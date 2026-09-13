using Test
using InteractiveUtils
using LinearAlgebra
using NearestCorrelationMatrix
import NearestCorrelationMatrix as NCM

include("Datasets.jl")
using .Datasets

include("CustomTestMacros.jl")
using .CustomTestMacros

internal_algtypes = setdiff(subtypes(NCMAlgorithm), (JuMPAlgorithm,))

function test_simple(algtype)
    return @testset "$(NCM.alg_name(algtype))" begin
        r0 = default_negdef(Float64)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        cache = init(prob, alg)
        sol = solve!(cache)

        @test_iscorrelation sol.X

        # Issue #19: Solution must always be Symmetric
        @test sol.X isa Symmetric

        # Handle Symmetric type matrices
        r0 = default_negdef(Float64)
        prob = NCMProblem(Symmetric(r0))
        alg = autotune(algtype, prob)
        @test_nothrow solve(prob, alg)

        # Handle Float16 input matrices
        r0 = default_negdef(Float16)
        prob = NCMProblem(r0)
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
