using Test
using LinearAlgebra: issymmetric
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals;


function test_iscorrelation(X)
    @test issquare(X)
    @test issymmetric(X)
    @test diagonals_are_one(X)
    @test constrained_to_pm_one(X)
    @test ispossemidef(X)
end


function test_alg(algType, msg)
    @testset "$msg" begin
        @testset "In-place" begin
            for T in (Float64, Float32, Float16)
                @testset "$T" begin
                    r0 = get_negdef_matrix(T)
                    alg = autotune(algType, NCMProblem(r0))

                    r = copy(r0)
                    @test_nowarn nearest_cor!(r, alg)

                    @test eltype(r) === T
                    @test r != r0
                    @test iscorrelation(r0) == false
                    test_iscorrelation(r)
                end
            end
        end

        @testset "Out-of-place" begin
            for T in (Float64, Float32, Float16)
                @testset "$T" begin
                    r0 = get_negdef_matrix(T)
                    alg = autotune(algType, NCMProblem(r0))

                    @test_nowarn nearest_cor(r0, alg)

                    r = nearest_cor(r0, alg)

                    @test eltype(r) === T
                    @test r != r0
                    @test iscorrelation(r0) == false
                    test_iscorrelation(r)
                end
            end
        end
    end
end


@testset "Public API" begin
    test_alg(Newton, "Newton Method")
    test_alg(DirectProjection, "Direct Projection")
    test_alg(AlternatingProjections, "Alternating Projections")
end
