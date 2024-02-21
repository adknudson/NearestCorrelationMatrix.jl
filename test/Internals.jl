using Test
using NearestCorrelationMatrix
using LinearAlgebra: issymmetric, Symmetric
using NearestCorrelationMatrix: _clampcor, _diagonals_are_one, _constrained_to_pm_one,
    _is_correlation, _is_square, _set_diag!, _copytolower!, _cor_constrain!, _cov2cor!,
    _prep_matrix!, _eigen_reversed


r_negdef = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]


supported_types = (Float64, Float32)


@testset "Internal Utilities" begin
    for T in supported_types
        x = rand(T)
        @test_nowarn _clampcor(x)
        @test typeof(_clampcor(x)) === T

        x = rand(T, 3, 3)
        @test_nowarn _is_square(x)
        @test _is_square(x) == true
        x = rand(T, 3, 5)
        @test _is_square(x) == false

        x = T[1.0 0.5; 0.5 1.0]
        @test_nowarn _diagonals_are_one(x)
        @test _diagonals_are_one(x) == true
        x[1,1] = T(2)
        @test _diagonals_are_one(x) == false

        x = rand(T, 4, 4)
        @test_nowarn _constrained_to_pm_one(x)
        @test _constrained_to_pm_one(x) == true
        x = T[2.0 0.5; 0.5 1.0]
        @test _constrained_to_pm_one(x) == false

        x = T.(r_negdef)
        @test_nowarn _is_correlation(x)
        @test _is_correlation(x) == false

        x = rand(T, 5, 5)
        @test_nowarn _set_diag!(x, one(T))
        @test_throws Exception _set_diag!(x, 3//4)
        x = rand(T, 4, 5)
        @test_throws Exception _set_diag!(x, one(T))

        x = rand(T, 5, 5)
        @test_nowarn _copytolower!(x)
        @test eltype(x) === T
        @test issymmetric(x) == true
        x = rand(T, 5, 4)
        @test_throws Exception _copytolower!(x)
        @test eltype(x) === T
        x = Symmetric(rand(T, 4, 4))
        @test_nowarn _copytolower!(x)
        @test eltype(x) === T
        @test issymmetric(x) == true

        x = rand(T, 4, 4)
        @test_nowarn _cor_constrain!(x)
        @test eltype(x) === T
        x = Symmetric(rand(T, 4, 4))
        @test_nowarn _cor_constrain!(x)
        @test eltype(x) === T

        x = T.(r_negdef)
        @test_nowarn _cov2cor!(x)
        @test eltype(x) === T
        x = Symmetric(T.(r_negdef))
        @test_nowarn _cov2cor!(x)
        @test eltype(x) === T
        @test typeof(x) <: Symmetric

        x = rand(T, 5, 5)
        @test_nowarn _prep_matrix!(x)
        @test eltype(x) === T
        @test issymmetric(x) == true
        @test _diagonals_are_one(x) == true

        x = T.(r_negdef)
        @test_nowarn _eigen_reversed(x)
        λ, P = _eigen_reversed(x)
        @test eltype(λ) === T
        @test eltype(P) === T
        x = Symmetric(x)
        @test_nowarn _eigen_reversed(x)
        λ, P = _eigen_reversed(x)
        @test eltype(λ) === T
        @test eltype(P) === T
    end
end
