using Test
using LinearAlgebra
using NearestCorrelationMatrix.Internals


supported_types = (Float64, Float32, Float16)


@testset "Internal Utilities" begin
    @testset "Matrix Properties" begin
        for T in supported_types
            sqr_mat = 2 * rand(T, 10, 10) .- one(T)
            rect_mat = 2 * rand(T, 10, 7) .- one(T)
            @test issquare(sqr_mat) == true
            @test issquare(rect_mat) == false


            x = T[one(T) T(0.3); T(0.3) one(T)]
            y = T[nextfloat(one(T)) T(0.3); T(0.3) one(T)]
            @test diagonals_are_one(x) == true
            @test diagonals_are_one(y) == false


            r = get_negdef_matrix(T)
            @test iscorrelation(r) == false
            @test iscorrelation(sqr_mat) == false
            @test iscorrelation(rect_mat) == false
        end
    end


    @testset "Out-of-place Methods" begin
        for T in supported_types
            # clampcor
            x = rand(T) + one(T)
            @test typeof(clampcor(x)) === T


            # cov2cor
            x = get_negdef_matrix(T)
            cor2cov!(x, T[5, 4, 3, 2])
            sym_mat = Symmetric(copy(x))

            x = cov2cor(x)
            @test issymmetric(x)
            @test diagonals_are_one(x)
            @test constrained_to_pm_one(x)

            sym_mat = cov2cor(sym_mat)
            @test issymmetric(sym_mat)
            @test diagonals_are_one(sym_mat)
            @test constrained_to_pm_one(sym_mat)


            # eigen_sym
            x = symmetric!(2 * rand(T, 10, 10) .- one(T))
            sym_mat = Symmetric(x)

            λ, P = eigen_sym(x)
            @test eltype(λ) === T
            @test eltype(P) === T

            λ, P = eigen_sym(sym_mat)
            @test eltype(λ) === T
            @test eltype(P) === T
        end
    end


    @testset "In-place Methods" begin
        for T in supported_types
            # clampcor!
            x = T[-2 1; -1 3]
            clampcor!(x)
            @test all(-one(T) .≤ x .≤ one(T))


            # setdiag!
            x = 2 * rand(T, 10, 10) .- one(T)
            sym_mat = Symmetric(2 * rand(T, 10, 10) .- one(T))
            rect_mat = 2 * rand(T, 10, 7) .- one(T)

            setdiag!(x, one(T))
            @test all(==(one(T)), diag(x))

            setdiag!(sym_mat, one(T))
            @test all(==(one(T)), diag(sym_mat))

            @test_throws Exception setdiag!(x, 3//4)
            @test_throws Exception setdiag!(rect_mat, one(T))


            # symmetric!
            for uplo in (:U, :L)
                x = 2 * rand(T, 10, 10) .- one(T)
                if !issymmetric(x)
                    symmetric!(x, uplo)
                    @test issymmetric(x) == true
                else
                    error("Test matrix expected to be non-symmetric. Try re-running the tests.")
                end

                rect_mat = 2 * rand(T, 10, 7) .- one(T)
                @test_throws Exception symmetric!(rect_mat, uplo)
            end

            x = 2 * rand(T, 10, 10) .- one(T)
            @test_throws ArgumentError symmetric!(x, :u)

            sym_mat = Symmetric(2 * rand(T, 10, 10) .- one(T))
            symmetric!(sym_mat)
            @test sym_mat isa Symmetric

            diag_mat = Diagonal(rand(T, 4))
            symmetric!(sym_mat)
            @test diag_mat isa Diagonal


            # corconstrain!
            x = T[
                2.0 0.8 0.1
                0.8 1.0 0.6
                0.1 0.6 0.2
            ]
            sym_mat = Symmetric(copy(x))
            diag_mat = Diagonal(diag(x))

            corconstrain!(x)
            @test diagonals_are_one(x)
            @test constrained_to_pm_one(x)

            corconstrain!(sym_mat)
            @test diagonals_are_one(sym_mat)
            @test constrained_to_pm_one(sym_mat)

            corconstrain!(diag_mat)
            @test diagonals_are_one(diag_mat)
            @test constrained_to_pm_one(diag_mat)


            # cov2cor!
            x = get_negdef_matrix(T)
            cor2cov!(x, T[5, 4, 3, 2])
            sym_mat = Symmetric(copy(x))

            cov2cor!(x)
            @test issymmetric(x)
            @test diagonals_are_one(x)
            @test constrained_to_pm_one(x)

            cov2cor!(sym_mat)
            @test issymmetric(sym_mat)
            @test diagonals_are_one(sym_mat)
            @test constrained_to_pm_one(sym_mat)
        end
    end
end
