using Test
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals
using LinearAlgebra: isposdef, Symmetric


include("test_macros.jl")


function test_robust_reps(nreps, algtype, size, T, test_pd; kwargs...)
    @testset "$(size)×$(size)" begin
        for _ in 1:nreps
            r0 = rand_negdef(T, size)
            prob = NCMProblem(r0)
            alg = autotune(algtype, prob)
            cache = init(prob, alg; kwargs...)
            sol = solve!(cache)

            @test iscorrelation(sol.X) == true

            if test_pd
                @test isposdef(sol.X) == true
            end
        end
    end
end


function test_robust(algtype, T; cutoff=Inf, test_pd=false, kwargs...)
    @testset "$algtype - $T" begin
        if cutoff ≥ 10
            test_robust_reps(100, algtype, 10, T, test_pd; kwargs...)
        end

        if cutoff ≥ 25
            test_robust_reps(100, algtype, 25, T, test_pd; kwargs...)
        end

        if cutoff ≥ 50
            test_robust_reps(100, algtype, 50, T, test_pd; kwargs...)
        end

        if cutoff ≥ 100
            test_robust_reps(50, algtype, 100, T, test_pd; kwargs...)
        end

        if cutoff ≥ 250
            test_robust_reps(50, algtype, 250, T, test_pd; kwargs...)
        end

        if cutoff ≥ 500
            test_robust_reps(20, algtype, 500, T, test_pd; kwargs...)
        end

        if cutoff ≥ 1000
            test_robust_reps(10, algtype, 1000, T, test_pd; kwargs...)
        end
    end
end


@testset verbose=true failfast=true "Robustness - PosSemiDef" begin
    test_robust(Newton, Float64; cutoff=1000)
    test_robust(Newton, Float32; cutoff=1000)
    test_robust(Newton, Float16; cutoff=1000, force_f16=true)

    test_robust(DirectProjection, Float64; cutoff=1000)
    test_robust(DirectProjection, Float32; cutoff=1000)
    test_robust(DirectProjection, Float16; cutoff=1000, force_f16=true)

    test_robust(AlternatingProjections, Float64; cutoff=250)
    test_robust(AlternatingProjections, Float32; cutoff=250)
    test_robust(AlternatingProjections, Float16; cutoff=250, force_f16=true)
end


@testset verbose=true failfast=true "Robustness - PosDef" begin
    test_robust(Newton, Float64; cutoff=1000, test_pd=true)
    test_robust(Newton, Float32; cutoff=1000, test_pd=true)
    test_robust(Newton, Float16; cutoff=1000, test_pd=true, force_f16=true)

    test_robust(DirectProjection, Float64; cutoff=1000, test_pd=true)
    test_robust(DirectProjection, Float32; cutoff=1000, test_pd=true)
    test_robust(DirectProjection, Float16; cutoff=1000, test_pd=true, force_f16=true)

    test_robust(AlternatingProjections, Float64; cutoff=250, test_pd=true, ensure_pd=true)
    test_robust(AlternatingProjections, Float32; cutoff=250, test_pd=true, ensure_pd=true)
    test_robust(AlternatingProjections, Float16; cutoff=250, test_pd=true, ensure_pd=true, force_f16=true)
end
