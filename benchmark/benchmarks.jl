using BenchmarkTools
using Random, LinearAlgebra
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: rand_negdef

Random.seed!(0x00c0ffee)

const SUITE = BenchmarkGroup()

function create_benchmarkable(n, alg; evals, samples, seconds)
    return @benchmarkable nearest_cor(A, alg) evals = evals samples = samples seconds = seconds setup = (A = rand_negdef(n))
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
