using NearestCorrelationMatrix
using LinearAlgebra
using Test

r_negdef = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]

supported_types = [Float16, Float32, Float64]

@testset "NearestCorrelationMatrix.jl" begin

    @testset "Nearest positive definite" begin
        r = nearest_cor(r_negdef)
        @test NearestCorrelationMatrix._iscorrelation(r)

        # Must respect input eltype
        for T in supported_types
            r = T.(r_negdef)
            @test eltype(nearest_cor(r)) === T
        end
    end

    @testset "Nearest positive semidefinite" begin
        r = nearest_cor(r_negdef, Newton(τ=0.0))
        λ = eigvals(r)
        @test issymmetric(r)
        @test all(λ .≥ 0)
        @test NearestCorrelationMatrix._diagonals_are_one(r)
        @test NearestCorrelationMatrix._constrained_to_pm_one(r)
    end

    @testset "Nearest positive definite in place" begin
        r = copy(r_negdef)
        nearest_cor!(r)
        @test NearestCorrelationMatrix._iscorrelation(r)

        # Must respect input eltype
        for T in supported_types
            r = T.(r_negdef)
            nearest_cor!(r)
            @test eltype(r) === T
        end
    end
end
