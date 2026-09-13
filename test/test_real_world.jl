using Test
using InteractiveUtils
using NearestCorrelationMatrix
import NearestCorrelationMatrix as NCM
using NearestCorrelationMatrix.Internals: iscorrelation

include("Datasets.jl")
using .Datasets

include("CustomTestMacros.jl")
using .CustomTestMacros

internal_algtypes = setdiff(subtypes(NCMAlgorithm), (JuMPAlgorithm,))
fast_algtypes = (Newton, DirectProjection)

for alg in internal_algtypes
    @info "Working on $alg"

    @testset "$(NCM.alg_name(alg))" begin
        @testset "BEYU11" begin
            A = beyu11()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "BHWI01" begin
            A = bhwi01()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "COR1399" begin
            A = cor1399()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        if alg ∈ fast_algtypes
            @testset "BCCD16" begin
                A = bccd16()
                @test !iscorrelation(A)
                nearest_cor!(A, alg)
                @test_iscorrelation A
            end

            @testset "COR3120" begin
                A = cor3120()
                @test !iscorrelation(A)
                nearest_cor!(A, alg)
                @test_iscorrelation A
            end
        end

        @testset "FING97" begin
            # unmasked
            A, _ = fing97()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A

            if NCM.supports_mask(alg)
                A, m = fing97()
                @test !iscorrelation(A)
                B = nearest_cor(A, alg; mask = m)
                @test_iscorrelation B
                @test B[m] == A[m]
            end
        end

        @testset "HIGH02" begin
            A = high02()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "MMB13" begin
            A = mmb13()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "TEC03" begin
            A = tec03()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "TYDA99R1" begin
            A = tyda99r1()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "TYDA99R2" begin
            A = tyda99r2()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "TYDA99R3" begin
            A = tyda99r3()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A
        end

        @testset "USGS13" begin
            A, _ = usgs13()
            @test !iscorrelation(A)
            nearest_cor!(A, alg)
            @test_iscorrelation A

            if NCM.supports_mask(alg)
                A, m = usgs13()
                @test !iscorrelation(A)
                B = nearest_cor(A, alg; mask = m)
                @test_iscorrelation B
                @test B[m] == A[m]
            end
        end
    end
end
