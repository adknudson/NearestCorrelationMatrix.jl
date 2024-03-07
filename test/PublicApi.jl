using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Utils;


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
                r = convert(AbstractMatrix{T}, r_negdef)
                rcopy = copy(r)

                @test_nowarn nearest_cor!(r, alg)
                @test eltype(r) === T
                @test r != rcopy
                @test iscorrelation(r) == true
                @test iscorrelation(rcopy) == false
            end
        end

        @testset "Out-of-place" begin
            for T in (Float64, Float32)
                r = convert(AbstractMatrix{T}, r_negdef)

                @test_nowarn nearest_cor(r, alg)

                p = nearest_cor(r, alg)

                @test eltype(p) === T
                @test p != r
                @test iscorrelation(p) == true
                @test iscorrelation(r) == false
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
