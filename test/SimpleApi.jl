using Test
using NearestCorrelationMatrix


@testset "Simple API" begin
    @test_isdefined nearest_cor
    @test_isdefined nearest_cor!

    r = rand(4, 4)

    @test_isimplemented nearest_cor(r)
    @test_isimplemented nearest_cor(r, Newton())
    @test_isimplemented nearest_cor(r, Newton)

    @test nearest_cor(r) isa AbstractMatrix

    @test_isimplemented nearest_cor!(r)
    @test_isimplemented nearest_cor!(r, Newton())
    @test_isimplemented nearest_cor!(r, Newton)

    @test nearest_cor!(r) isa AbstractMatrix
end
