using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: get_negdef_matrix


@testset "Simple API" begin
    @test_isdefined nearest_cor
    @test_isdefined nearest_cor!

    r = get_negdef_matrix(Float64)

    @test_isimplemented nearest_cor(r)
    @test_isimplemented nearest_cor(r, Newton())
    @test_isimplemented nearest_cor(r, Newton)

    @test nearest_cor(r) isa AbstractMatrix

    @test_isimplemented nearest_cor!(r)
    @test_isimplemented nearest_cor!(r, Newton())
    @test_isimplemented nearest_cor!(r, Newton)

    @test nearest_cor!(r) isa AbstractMatrix
end
