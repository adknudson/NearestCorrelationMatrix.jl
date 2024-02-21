using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix: _is_correlation


function test_alg(alg, msg)
    r_negdef = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]

    @testset "$msg" begin
        for T in (Float64, Float32)
            r = T.(r_negdef)
            @test_nowarn nearest_cor!(r, alg)
            @test eltype(r) === T
            @test _is_correlation(r) == true

            r = T.(r_negdef)
            @test_nowarn nearest_cor(r, alg)
            p = nearest_cor(r, alg)
            @test eltype(p) === T
            @test p != r
            @test _is_correlation(p) == true
            @test _is_correlation(r) == false
        end
    end
end


@testset "Public API" begin
    test_alg(default_alg(), "Default Algorithm")
    test_alg(Newton(), "Newton Method")
    test_alg(DirectProjection(), "Direct Projection")
    test_alg(AlternatingProjection(), "Alternating Projections")
end
