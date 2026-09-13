using Test
using LinearAlgebra
using NearestCorrelationMatrix.Internals

supported_types = (Float64, Float32, Float16)

@testset "clamp_cor" begin
    # Assumptions:
    # works for a wide range of numeric types
    # values are constrained to ±1
    # type is preserved after clamping
    for T in (Int64, BigFloat, Float64, Float32, Float16, Rational{Int64})
        x_gt_cor = T(2) * one(T)
        x_lt_cor = T(-2) * one(T)
        x_eq_ub = one(T)
        x_eq_lb = -one(T)
        x_valid_cor = zero(T)

        @test clamp_cor(x_gt_cor) == one(T)
        @test clamp_cor(x_lt_cor) == -one(T)
        @test clamp_cor(x_eq_ub) == x_eq_ub
        @test clamp_cor(x_eq_lb) == x_eq_lb
        @test clamp_cor(x_valid_cor) == x_valid_cor

        @test typeof(clamp_cor(x_gt_cor)) === T
        @test typeof(clamp_cor(x_lt_cor)) === T
        @test typeof(clamp_cor(x_eq_ub)) === T
        @test typeof(clamp_cor(x_eq_lb)) === T
        @test typeof(clamp_cor(x_valid_cor)) === T
    end
end

@testset "cov2cor!" begin
    # Assumptions:
    # works on dense matrices and symmetric matrices
    # works on eltypes of (Float16, Float32, Float64, BigFloat)
    # result is symmetric
    # non-diagonal elements are constrained to ±1
    # diagonal elements are equal to 1 after the transformation
    # for symmetric matrices, writes to the parent matrix
    X = [
        2.5 0.75 0.175
        0.75 0.7 0.135
        0.175 0.135 0.043
    ]

    for T in (Float16, Float32, Float64, BigFloat)
        Y = convert(Matrix{T}, X)
        cov2cor!(Y)
        @test has_unit_diagonal(Y)
        @test constrained_to_pm1(Y)
        @test issymmetric(Y)

        S = Symmetric(convert(Matrix{T}, X))
        cov2cor!(S)
        @test has_unit_diagonal(S)
        @test constrained_to_pm1(S)
        @test issymmetric(S)
    end
end

@testset "eigen_sym" begin
    # Assumptions:
    # works on dense and symmetric matrices
    # eltype of eigenvalues and eigenvectors matches the eltype of the input matrix
    # works on eltypes of (Float16, Float32, Float64)
    # eigenvalues are sorted in descending order

    # If X is symmetric, then the spectral decomposition is guaranteed to return real values

    # For Julia 1.10, eigen(Symmetric(X)) where eltype(X) == Float16 would return a
    # decomposition with Float32 values. We define our own `eigen_sym` that respects
    # the eltype of the input matrix, even though this is now fixed in Julia 1.12.

    X = [
        2.5 0.75 0.175
        0.75 0.7 0.135
        0.175 0.135 0.043
    ]

    for T in (Float16, Float32, Float64)
        Y = convert(Matrix{T}, X)
        λ, P = eigen_sym(Y)
        @test eltype(λ) === T
        @test eltype(P) === T
        @test issorted(λ; rev = true)

        S = Symmetric(Y)
        λ, P = eigen_sym(Y)
        @test eltype(λ) === T
        @test eltype(P) === T
        @test issorted(λ; rev = true)
    end
end

@testset "setdiag!" begin
    # Assumptions:
    # works on square matrices
    # works on dense, symmetric, and diagonal matrices
    # rejects non-square matrices

    v = 3.14
    X = rand(Float64, 8, 8)
    S = Symmetric(copy(X))
    D = Diagonal(copy(X))
    R = rand(Float64, 8, 5)

    setdiag!(X, v)
    @test all(==(v), diag(X))

    setdiag!(S, v)
    @test all(==(v), diag(S))

    setdiag!(D, v)
    @test all(==(v), diag(D))

    @test_throws DimensionMismatch setdiag!(R, v)
end

@testset "symmetrize!" begin
    # Assumptions:
    # works on square matrices
    # works on dense, symmetric, and diagonal matrices
    # rejects non-square matrices
    # rejects `uplo` not in (:U, :L)
    # defaults to copying the upper view if `uplo` cannot be inferred

    n = 8
    X = rand(Float64, n, n)
    S = Symmetric(copy(X))
    D = Diagonal(copy(X))
    R = rand(Float64, n, n - 1)

    symmetrize!(X)
    @test issymmetric(X)

    symmetrize!(S)
    @test issymmetric(S)
    @test issymmetric(parent(S))

    symmetrize!(D)
    @test issymmetric(D)

    @test_throws DimensionMismatch symmetrize!(R)

    for uplo in (:u, :l, :upper, :lower, :Upper, :Lower, :a, :b)
        @test_throws ArgumentError symmetrize!(X, uplo)
    end

    # Upper view
    A = rand(Float64, n, n)
    X = copy(A)
    symmetrize!(X, :U)
    @test triu(X) == triu(A)
    @test tril(X)' == triu(A)
    @test triu(X)' != tril(A)
    @test tril(X) != tril(A)

    S = Symmetric(copy(A), :U)
    symmetrize!(S)
    P = parent(S)
    @test triu(P) == triu(A)
    @test tril(P)' == triu(A)
    @test triu(P)' != tril(A)
    @test tril(P) != tril(A)

    # Lower view
    A = rand(Float64, n, n)
    X = copy(A)
    symmetrize!(X, :L)
    @test triu(X)' == tril(A)
    @test tril(X) == tril(A)
    @test triu(X) != triu(A)
    @test tril(X)' != triu(A)

    S = Symmetric(copy(A), :L)
    symmetrize!(S)
    P = parent(S)
    @test triu(P)' == tril(A)
    @test tril(P) == tril(A)
    @test triu(P) != triu(A)
    @test tril(P)' != triu(A)
end
