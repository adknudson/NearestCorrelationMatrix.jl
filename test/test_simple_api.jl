@testset "Simple API" begin
    @test_isdefined nearest_cor
    @test_isdefined nearest_cor!

    r = default_negdef(Float64)

    @test_isimplemented nearest_cor(r)
    @test_isimplemented nearest_cor(r, Newton())
    @test_isimplemented nearest_cor(r, Newton)

    @test nearest_cor(r) isa AbstractMatrix

    @test_isimplemented nearest_cor!(r)
    @test_isimplemented nearest_cor!(r, Newton())
    @test_isimplemented nearest_cor!(r, Newton)

    @test nearest_cor!(r) isa AbstractMatrix

    # not symmetric input
    r = rand(4, 4)
    @test_nothrow nearest_cor(r)
    @test_nothrow nearest_cor!(r)

    # Symmetric type input
    r = Symmetric(rand(4, 4))
    @test_nothrow nearest_cor(r)
    @test_nothrow nearest_cor!(r)

    # Float16 input
    r = rand(Float16, 4, 4)
    @test_nothrow nearest_cor(r)
    @test_nothrow nearest_cor!(r)
end
