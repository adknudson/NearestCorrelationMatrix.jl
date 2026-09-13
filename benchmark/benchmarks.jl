using BenchmarkTools
using Random, LinearAlgebra
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: symmetrize!

Random.seed!(0x00c0ffee)

const SUITE = BenchmarkGroup()

function rand_negdef(n::Int)
    while true
        A = 2.0 * rand(Float64, n, n) .- 1.0
        symmetrize!(A)
        A[diagind(A)] .= 1.0
        !isposdef(A) && return A
    end
    return zeros(Float64, 0, 0)
end

function create_benchmarkable(n, alg; evals, samples, seconds)
    return @benchmarkable nearest_cor(A, $alg) evals = evals samples = samples seconds = seconds setup = (A = rand_negdef($n))
end

function create_benchmark_group(n::Int; evals = 5, samples = 100, seconds = 60)
    grp = BenchmarkGroup()
    grp["Nw"] = create_benchmarkable(n, Newton; evals, samples, seconds)
    grp["AP"] = create_benchmarkable(n, AlternatingProjections; evals, samples, seconds)
    grp["AA"] = create_benchmarkable(n, AcceleratedAP; evals, samples, seconds)
    grp["Di"] = create_benchmarkable(n, DirectProjection; evals, samples, seconds)
    return grp
end

SUITE["n=10"] = create_benchmark_group(10; samples = 30)
SUITE["n=100"] = create_benchmark_group(100; samples = 30)
SUITE["n=1000"] = create_benchmark_group(1000; samples = 30, seconds = 180)
