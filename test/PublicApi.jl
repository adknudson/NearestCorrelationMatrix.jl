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


function test_alg(alg, msg)
    r_negdef = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]

    @testset "$msg" begin
        @testset "In-place" begin
            for T in (Float64, Float32)
                rcopy = convert(AbstractMatrix{T}, copy(r_negdef))
                p = copy(rcopy)

                @test_nowarn nearest_cor!(p, alg)

                @test eltype(p) === T
                @test p != rcopy
                @test iscorrelation(rcopy) == false
                test_iscorrelation(p)
            end
        end

        @testset "Out-of-place" begin
            for T in (Float64, Float32)
                rcopy = convert(AbstractMatrix{T}, copy(r_negdef))

                @test_nowarn nearest_cor(rcopy, alg)

                p = nearest_cor(rcopy, alg)

                @test eltype(p) === T
                @test p != rcopy
                @test iscorrelation(rcopy) == false
                test_iscorrelation(p)
            end
        end
    end
end


@testset "Public API" begin
    test_alg(default_alg(), "Default Algorithm")
    test_alg(Newton(), "Newton Method")
    test_alg(DirectProjection(), "Direct Projection")
    test_alg(AlternatingProjection(), "Alternating Projections")
end
