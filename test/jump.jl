using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals
using JuMP, COSMO


include("test_macros.jl")


@testset "JuMP Extension" begin
    r0 = get_negdef_matrix(Float64)
    prob = NCMProblem(r0)

    @test_isdefined JuMPAlgorithm
    @test_isimplemented JuMPAlgorithm(COSMO.Optimizer)

    optimizer = optimizer_with_attributes(
		COSMO.Optimizer,
		MOI.Silent() => true,
        "rho" => 1.0
    )
    alg = JuMPAlgorithm(optimizer)

    @test_nothrow solve(prob, alg)
end
