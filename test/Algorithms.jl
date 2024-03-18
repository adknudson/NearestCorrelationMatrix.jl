using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix: supports_symmetric, supports_float16
using NearestCorrelationMatrix.Internals
using LinearAlgebra: issymmetric, isposdef, Symmetric


macro test_iscorrelation(ex)
    return quote
        @test issquare($(esc(ex)))
        @test issymmetric($(esc(ex)))
        @test diagonals_are_one($(esc(ex)))
        @test constrained_to_pm_one($(esc(ex)))
        @test ispossemidef($(esc(ex)))
    end
end


function test_simple(algtype)
    @testset "Simple Test" begin
        r0 = get_negdef_matrix(Float64)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        cache = init(prob, alg)
        sol = solve!(cache)

        @test iscorrelation(sol.X) == true
        @test isposdef(sol.X) == true

        # Handle Symmetric type matrices
        r0 = get_negdef_matrix(Float64)
        prob = NCMProblem(Symmetric(r0))
        alg = autotune(algtype, prob)
        @test_nothrow solve(prob, alg)


        # Handle Float16 input matrices
        r0 = get_negdef_matrix(Float16)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        if supports_float16(alg)
            @test_nothrow solve(prob, alg)
        else
            @test_throws Exception solve(prob, alg)
            @test_nothrow solve(prob, alg; fix_f16=true)
        end
    end
end


function build_and_solve(algtype, n, T; kwargs...)
    r0 = rand_negdef(T, n)
    prob = NCMProblem(r0)
    alg = autotune(algtype, prob)
    cache = init(prob, alg; kwargs...)
    sol = solve!(cache)
    return sol
end


function test_robust_reps(nreps, algtype, size, T; kwargs...)
    @testset "$(size)×$(size)" begin
        for _ in 1:nreps
            sol = build_and_solve(algtype, size, T; kwargs...)
            @test iscorrelation(sol.X) == true
            @test isposdef(sol.X) == true
        end
    end
end


function test_robust(algtype, T; kwargs...)
    @testset "Robust Test $(nameof(T))" begin
        test_robust_reps(100, algtype, 10, T; kwargs...)
        test_robust_reps(100, algtype, 25, T; kwargs...)
        test_robust_reps(100, algtype, 50, T; kwargs...)
        test_robust_reps(100, algtype, 100, T; kwargs...)
        test_robust_reps(50, algtype, 250, T; kwargs...)
        test_robust_reps(20, algtype, 500, T; kwargs...)
        test_robust_reps(10, algtype, 1000, T; kwargs...)
    end
end


@testset verbose=true "Algorithms" begin
    @testset verbose=true "Newton" begin
        algtype = Newton
        test_simple(algtype)
        test_robust(algtype, Float64)
        test_robust(algtype, Float32)
        test_robust(algtype, Float16; force_f16=true)
    end

    @testset verbose=true "DirectProjection" begin
        algtype = DirectProjection
        test_simple(algtype)
        # test_robust(algtype, Float64)
        # test_robust(algtype, Float32)
        # test_robust(algtype, Float16)
    end

    @testset verbose=true "AlternatingProjections" begin
        algtype = AlternatingProjections
        test_simple(algtype)

        # T = Float64
        # @testset "Robust Test $(nameof(T))" begin
        #     test_robust_reps(100, algtype, 10, T)
        #     test_robust_reps(100, algtype, 25, T)
        #     test_robust_reps(100, algtype, 50, T)
        #     test_robust_reps(100, algtype, 100, T)
        # end

        # T = Float32
        # @testset "Robust Test $(nameof(T))" begin
        #     test_robust_reps(100, algtype, 10, T)
        #     test_robust_reps(100, algtype, 25, T)
        #     test_robust_reps(100, algtype, 50, T)
        #     test_robust_reps(100, algtype, 100, T)
        # end
    end
end
|
