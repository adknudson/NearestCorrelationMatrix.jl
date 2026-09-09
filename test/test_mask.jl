@testset "Masking (fixed elements)" begin
    using NearestCorrelationMatrix
    using LinearAlgebra

    # A simple 3x3 problem with a fixed (2,3) element
    r0 = Float64[
        1.0 0.2 0.9
        0.2 1.0 0.3
        0.9 0.3 1.0
    ]
    mask = [
        false false false
        false false true
        false true false
    ]

    @testset "mask=nothing is a no-op" begin
        @test mask(NCMProblem(r0)) === nothing
    end

    @testset "normalize_mask" begin
        prob = NCMProblem(r0, mask = mask)
        @test prob.mask == mask

        # asymmetic input is symmetrized
        asym = [
            false false false
            false false false
            false true false
        ]
        pm = normalize_mask(r0, asym)
        @test pm[2, 3] == true
        @test pm[3, 2] == true

        # diagonal is always cleared
        diag = trues(3, 3)
        pm = normalize_mask(r0, diag)
        @test all(diag(pm)) == false

        # wrong size throws
        @test_throws DimensionMismatch normalize_mask(r0, falses(2, 2))
    end

    @testset "AlternatingProjections enforces mask" begin
        X = nearest_cor(copy(r0), AlternatingProjections(); mask = mask)
        @test_iscorrelation X
        @test X[2, 3] ≈ r0[2, 3]
        @test X[3, 2] ≈ r0[3, 2]
    end

    @testset "AlternatingProjectionsAA enforces mask" begin
        X = nearest_cor(copy(r0), AlternatingProjectionsAA(); mask = mask)
        @test_iscorrelation X
        @test X[2, 3] ≈ r0[2, 3]
        @test X[3, 2] ≈ r0[3, 2]
    end

    @testset "mask kwarg overrides problem mask" begin
        prob = NCMProblem(r0, mask = mask)
        override = [
            false false false
            false false false
            false false true
        ]
        X = nearest_cor(prob, AlternatingProjections(); mask = override)
        @test X[3, 3] ≈ r0[3, 3]
    end

    @testset "informative error for unsupported algorithm" begin
        @test_throws ErrorException nearest_cor(
            copy(r0), Newton(); mask = mask
        )
        @test_throws ErrorException nearest_cor(
            copy(r0), DirectProjection(); mask = mask
        )
    end
end
