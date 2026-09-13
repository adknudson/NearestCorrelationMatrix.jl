using Test
using InteractiveUtils
using NearestCorrelationMatrix
import NearestCorrelationMatrix as NCM

internal_algtypes = setdiff(subtypes(NCMAlgorithm), (JuMPAlgorithm,))

prob = NCMProblem(rand(4, 4))

for algtype in internal_algtypes
    @testset "$(NCM.alg_name(algtype))" begin
        @test NCM.supports_parameterless_construction(algtype)

        alg = NCM.construct_algorithm(algtype)
        @test alg isa algtype

        alg = autotune(algtype, prob)
        @test alg isa algtype

        # supports_parameterless_construction works on the type, not the instance
        @test_throws MethodError NCM.supports_parameterless_construction(alg)
    end
end
