using Test
using LinearAlgebra: diag, issymmetric, Symmetric, Diagonal
using NearestCorrelationMatrix.Utils


r_negdef = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]


supported_types = (Float64, Float32, Float16)


function test_isprecorrelation(X)
    @test issquare(X)
    @test issymmetric(X)
    @test diagonals_are_one(X)
    @test constrained_to_pm_one(X)
end


@testset "Internal Utilities" begin
    @testset "Matrix Properties" begin
        for T in supported_types
            sqr_mat = rand(T, 3, 3)
            rect_mat = rand(T, 3, 5)

            @test_nowarn issquare(sqr_mat)
            @test issquare(sqr_mat) == true
            @test issquare(rect_mat) == false

            r = convert(AbstractMatrix{T}, r_negdef)
            @test_nowarn iscorrelation(sqr_mat)
            @test iscorrelation(r) == false
        end
    end


    @testset "Out-of-place Methods" begin
        for T in supported_types
            # clampcor
            x = rand(T)
            @test_nowarn clampcor(x)
            @test typeof(clampcor(x)) === T


            # cov2cor
            x = symmetric!(rand(T, 10, 10))
            sym_mat = Symmetric(copy(x))

            @test isprecorrelation(x) == false
            @test_nowarn cov2cor(x)
            x = cov2cor(x)
            test_isprecorrelation(x)

            @test isprecorrelation(sym_mat) == false
            @test_nowarn cov2cor(sym_mat)
            sym_mat = cov2cor(sym_mat)
            test_isprecorrelation(sym_mat)


            # eigen_safe
            x = symmetric!(rand(T, 10, 10))
            sym_mat = Symmetric(x)

            @test_nowarn eigen_safe(x)
            λ, P = eigen_safe(x)
            @test eltype(λ) === T
            @test eltype(P) === T

            @test_nowarn eigen_safe(sym_mat)
            λ, P = eigen_safe(sym_mat)
            @test eltype(λ) === T
            @test eltype(P) === T

            x = 2 * rand(T, 10, 10) .- 1
            @test issymmetric(x) == false
            @test_throws Exception eigen_safe(x)
        end
    end


    @testset "In-place Methods" begin
        for T in supported_types
            # clampcor!
            x = T[-2 1; -1 3]
            @test_nowarn clampcor!(x)
            @test all(-one(T) .≤ x .≤ one(T))


            # setdiag!
            x = rand(T, 4, 4)
            @test_nowarn setdiag!(x, one(T))
            @test all(==(one(T)), diag(x))
            @test_throws Exception setdiag!(x, 3//4)
            rect_mat = rand(T, 3, 4)
            @test_throws Exception setdiag!(rect_mat, one(T))


            # symmetric!
            for uplo in (:U, :L)
                x = rand(T, 4, 4)
                @test issymmetric(x) == false
                @test_nowarn symmetric!(x, uplo)
                @test issymmetric(x) == true

                rect_mat = rand(T, 3, 4)
                @test_throws Exception symmetric!(rect_mat, uplo)
            end

            x = rand(T, 4, 4)
            @test_throws Exception symmetric!(x, :u)

            sym_mat = Symmetric(rand(T, 4, 4))
            @test_nowarn symmetric!(sym_mat)
            @test sym_mat isa Symmetric

            diag_mat = Diagonal(rand(T, 4))
            @test_nowarn symmetric!(sym_mat)
            @test diag_mat isa Diagonal


            # corconstrain!
            x = T[
                2.0 0.8 0.1
                0.8 1.0 0.6
                0.1 0.6 0.2
            ]
            sym_mat = Symmetric(copy(x))
            diag_mat = Diagonal(diag(x))

            @test isprecorrelation(x) == false
            @test_nowarn corconstrain!(x)
            test_isprecorrelation(x)

            @test isprecorrelation(sym_mat) == false
            @test_nowarn corconstrain!(sym_mat)
            test_isprecorrelation(sym_mat)

            @test isprecorrelation(diag_mat) == false
            @test_nowarn corconstrain!(diag_mat)
            test_isprecorrelation(diag_mat)


            # cov2cor!
            x = convert(AbstractMatrix{T}, r_negdef)
            cor2cov!(x, T[5, 4, 3, 2])
            sym_mat = Symmetric(copy(x))

            @test isprecorrelation(x) == false
            @test_nowarn cov2cor!(x)
            test_isprecorrelation(x)

            @test isprecorrelation(sym_mat) == false
            @test_nowarn cov2cor!(sym_mat)
            test_isprecorrelation(sym_mat)


            # checkmat!
            x = rand(T, 4, 4)
            @test issymmetric(x) == false
            @test_nowarn checkmat!(x; warn=false)
            @test issymmetric(x) == true

            x = rand(T, 4, 4)
            @test issymmetric(x) == false
            @test_warn "The input matrix is not symmetric. Replacing with (X + X') / 2" checkmat!(x; warn=true)

            x = rand(T, 4, 5)
            @test_throws Exception checkmat!(x)


            # project_psd!
            x = symmetric!(rand(T, 4, 4))
            @test issymmetric(x) == true
            @test_nowarn project_psd!(x)

            x = rand(T, 4, 4)
            @test issymmetric(x) == false
            @test_throws Exception project_psd!(x)

            x = Symmetric(rand(T, 4, 4))
            @test_nowarn project_psd!(x)
        end
    end
end
