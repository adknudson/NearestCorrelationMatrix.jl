using Test
using Aqua
using NearestCorrelationMatrix
using LinearAlgebra


Aqua.test_all(NearestCorrelationMatrix)


const r_negdef = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]

const supported_types = [Float64, Float32]


@testset "Setup" begin
    @test NearestCorrelationMatrix._constrained_to_pm_one(r_negdef)
    @test NearestCorrelationMatrix._diagonals_are_one(r_negdef)
    @test issymmetric(r_negdef)
    @test !isposdef(r_negdef)
end


@testset "Utilities" begin
    x = rand(10, 10)
    NearestCorrelationMatrix._copytolower!(x)
    @test issymmetric(x)
end


@testset "Newton Algorithm" begin

    @testset "Positive Definite" begin
        @testset "Copy" begin
            alg = Newton()
            r = nearest_cor(r_negdef, alg)

            @test NearestCorrelationMatrix._diagonals_are_one(r)
            @test NearestCorrelationMatrix._constrained_to_pm_one(r)
            @test issymmetric(r)
            @test isposdef(r)
        end

        @testset "In Place" begin
            for T in supported_types
                alg = Newton(τ=sqrt(eps(T)))
                r = T.(r_negdef)
                nearest_cor!(r, alg)

                @test NearestCorrelationMatrix._diagonals_are_one(r)
                @test NearestCorrelationMatrix._constrained_to_pm_one(r)
                @test issymmetric(r)
                @test isposdef(r)

                @test eltype(r) === T
            end
        end
    end

    @testset "Positive Semidefinite" begin
        alg = Newton(τ=0.0)
        r = nearest_cor(r_negdef, alg)

        λ = eigvals(r)
        T = eltype(r)

        approx_zero(x) = isapprox(0, x, atol=eps(typeof(x)), rtol=0)
        greater_zero(x) = x > 0

        # all eigenvalues are ≈0 or greater
        @test all(l -> greater_zero(l) || approx_zero(l), λ)

        @test NearestCorrelationMatrix._diagonals_are_one(r)
        @test NearestCorrelationMatrix._constrained_to_pm_one(r)
        @test issymmetric(r)
    end

end


@testset "Alternating Projection Algorithm" begin
    @testset "Copy" begin
        alg = AlternatingProjection()
        r = nearest_cor(r_negdef, alg)

        @test NearestCorrelationMatrix._diagonals_are_one(r)
        @test NearestCorrelationMatrix._constrained_to_pm_one(r)
        @test issymmetric(r)
        @test isposdef(r)
    end

    @testset "In Place" begin
        for T in supported_types
            alg = AlternatingProjection(tol=sqrt(eps(T)))
            r = T.(r_negdef)
            nearest_cor!(r, alg)

            @test NearestCorrelationMatrix._diagonals_are_one(r)
            @test NearestCorrelationMatrix._constrained_to_pm_one(r)
            @test issymmetric(r)
            @test isposdef(r)

            @test eltype(r) === T
        end
    end
end


@testset "Projection Algorithm" begin
    @testset "Copy" begin
        alg = DirectProjection()
        r = nearest_cor(r_negdef, alg)

        @test NearestCorrelationMatrix._diagonals_are_one(r)
        @test NearestCorrelationMatrix._constrained_to_pm_one(r)
        @test issymmetric(r)
        @test isposdef(r)
    end

    @testset "In Place" begin
        for T in supported_types
            alg = DirectProjection(sqrt(eps(T)))
            r = T.(r_negdef)
            nearest_cor!(r, alg)

            @test NearestCorrelationMatrix._diagonals_are_one(r)
            @test NearestCorrelationMatrix._constrained_to_pm_one(r)
            @test issymmetric(r)
            @test isposdef(r)

            @test eltype(r) === T
        end
    end
end
