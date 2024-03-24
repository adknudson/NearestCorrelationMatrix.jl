using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix: alg_name
using NearestCorrelationMatrix.Internals
using LinearAlgebra: isposdef, Symmetric
using JuMP, COSMO

include("test_macros.jl");

function test_robust_reps(algtype::Type, nreps, size, T, test_pd; kwargs...)
    @testset "$(size)×$(size)" begin
        for _ in 1:nreps
            r0 = rand_negdef(T, size)
            prob = NCMProblem(r0)
            alg = autotune(algtype, prob)
            cache = init(prob, alg; kwargs...)
            sol = solve!(cache)

            if test_pd
                @test isposdef(sol.X) == true
            else
                @test_iscorrelation sol.X
            end
        end
    end
end

function test_robust_reps(alg::NCMAlgorithm, nreps, size, T, test_pd; kwargs...)
    @testset "$(size)×$(size)" begin
        for _ in 1:nreps
            r0 = rand_negdef(T, size)
            prob = NCMProblem(r0)
            cache = init(prob, alg; kwargs...)
            sol = solve!(cache)

            if test_pd
                @test isposdef(sol.X) == true
            else
                @test_iscorrelation sol.X
            end
        end
    end
end

function test_robust(algtype::Type, T; cutoff=Inf, test_pd=false, kwargs...)
    @testset "$algtype - $T" begin
        cutoff < 10 && return nothing
        test_robust_reps(algtype, 100, 10, T, test_pd; kwargs...)
        cutoff < 25 && return nothing
        test_robust_reps(algtype, 100, 25, T, test_pd; kwargs...)
        cutoff < 50 && return nothing
        test_robust_reps(algtype, 100, 50, T, test_pd; kwargs...)
        cutoff < 100 && return nothing
        test_robust_reps(algtype, 50, 100, T, test_pd; kwargs...)
        cutoff < 250 && return nothing
        test_robust_reps(algtype, 50, 250, T, test_pd; kwargs...)
        cutoff < 500 && return nothing
        test_robust_reps(algtype, 20, 500, T, test_pd; kwargs...)
        cutoff < 1000 && return nothing
        test_robust_reps(algtype, 10, 1000, T, test_pd; kwargs...)
    end
end

function test_robust(alg::NCMAlgorithm, T; cutoff=Inf, test_pd=false, kwargs...)
    @testset "$(alg_name(alg)) - $T" begin
        cutoff < 10 && return nothing
        test_robust_reps(alg, 100, 10, T, test_pd; kwargs...)
        cutoff < 25 && return nothing
        test_robust_reps(alg, 100, 25, T, test_pd; kwargs...)
        cutoff < 50 && return nothing
        test_robust_reps(alg, 100, 50, T, test_pd; kwargs...)
        cutoff < 100 && return nothing
        test_robust_reps(alg, 50, 100, T, test_pd; kwargs...)
        cutoff < 250 && return nothing
        test_robust_reps(alg, 50, 250, T, test_pd; kwargs...)
        cutoff < 500 && return nothing
        test_robust_reps(alg, 20, 500, T, test_pd; kwargs...)
        cutoff < 1000 && return nothing
        test_robust_reps(alg, 10, 1000, T, test_pd; kwargs...)
    end
end

# These must succeed at all costs
@testset verbose = true "Robustness - PosSemiDef" begin
    algtype = Newton
    test_robust(algtype, Float64; cutoff=1000)
    test_robust(algtype, Float32; cutoff=1000)
    test_robust(algtype, Float16; cutoff=1000, force_f16=true)

    algtype = DirectProjection
    test_robust(algtype, Float64; cutoff=1000)
    test_robust(algtype, Float32; cutoff=1000)
    test_robust(algtype, Float16; cutoff=1000, force_f16=true)

    algtype = AlternatingProjections
    test_robust(algtype, Float64; cutoff=250)
    test_robust(algtype, Float32; cutoff=250)
    test_robust(algtype, Float16; cutoff=250, force_f16=true)

    alg = JuMPAlgorithm(optimizer_with_attributes(
        COSMO.Optimizer, MOI.Silent() => true, "rho" => 1.0
    ))
    test_robust(alg, Float64; cutoff=1000)
end

@testset verbose = true "Robustness - PosDef" begin
    algtype = Newton
    test_robust(algtype, Float64; cutoff=1000, test_pd=true)
    test_robust(algtype, Float32; cutoff=1000, test_pd=true)
    test_robust(algtype, Float16; cutoff=1000, test_pd=true, force_f16=true)

    algtype = DirectProjection
    test_robust(algtype, Float64; cutoff=1000, test_pd=true)
    test_robust(algtype, Float32; cutoff=1000, test_pd=true)
    test_robust(algtype, Float16; cutoff=1000, test_pd=true, force_f16=true)

    algtype = AlternatingProjections
    test_robust(algtype, Float64; cutoff=250, test_pd=true, ensure_pd=true)
    test_robust(algtype, Float32; cutoff=250, test_pd=true, ensure_pd=true)
    test_robust(algtype, Float16; cutoff=250, test_pd=true, ensure_pd=true, force_f16=true)

    alg = JuMPAlgorithm(optimizer_with_attributes(
        COSMO.Optimizer, MOI.Silent() => true, "rho" => 1.0
    ))
    test_robust(algtype, Float64; cutoff=1000, test_pd=true, ensure_pd=true)
end
