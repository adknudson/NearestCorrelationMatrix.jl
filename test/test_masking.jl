using Test
using LinearAlgebra
using InteractiveUtils
using NearestCorrelationMatrix
import NearestCorrelationMatrix as NCM

include("Datasets.jl")
using .Datasets

include("CustomTestMacros.jl")
using .CustomTestMacros

masking_algs = filter(alg -> NCM.supports_mask(alg), subtypes(NCMAlgorithm))
supported_types = (Float64, Float32, Float16)

for algtype in masking_algs, T in supported_types
    @testset "$(NCM.alg_name(algtype)) - $T" begin
        A, m = usgs13()
        A = convert(Matrix{T}, A)
        X = copy(A)
        prob = NCMProblem(X; mask = m)
        alg = autotune(algtype, prob)
        solver = init(prob, alg)
        @test solver.mask !== nothing
        @test Base.mightalias(solver.A_orig, A) == false
        sol = solve!(solver)
        @test isapprox(sol.X, sol.X')
        @test isapprox(diag(sol.X), ones(T, size(sol.X, 1)))
        @test all(x -> prevfloat(-one(T)) <= x <= nextfloat(one(T)), sol.X)
        evals = eigvals(sol.X)
        @test all(>=(-sqrt(sqrt(eps(T)))), evals)
        @test sol.X[m] == A[m]
    end
end
