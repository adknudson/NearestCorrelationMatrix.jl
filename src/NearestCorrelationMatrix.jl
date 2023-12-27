module NearestCorrelationMatrix

using LinearAlgebra: diag, diagm, diagind, dot, eigen, issymmetric, isposdef, norm
using LinearAlgebra: Diagonal, Symmetric
using Statistics: clampcor
using PrecompileTools


abstract type NearestCorrelationAlgorithm end


default_alg() = Newton()


export 
    nearest_cor, 
    nearest_cor!, 
    Newton, 
    AlternatingProjection,
    DirectProjection,
    default_alg


include("common.jl")
include("nearest_correlation.jl")
include("newton_method.jl")
include("alternating_projection.jl")
include("direct_projection.jl")

include("Experimental/Experimental.jl")


@setup_workload begin
    function make_data(T, n)
        x = 2 * rand(T, n, n) .- one(T)
        _set_diag!(x, one(T))
        _copytolower!(x)
        return x
    end

    f32 = make_data(Float32, 5)
    f64 = make_data(Float64, 5)

    @compile_workload begin
        alg = Newton(τ = 1e-10, tol=1e-8)
        
        nearest_cor(f32, alg)
        nearest_cor(f64, alg)

        alg = AlternatingProjection()

        nearest_cor(f32, alg)
        nearest_cor(f64, alg)

        alg = DirectProjection()

        nearest_cor(f32, alg)
        nearest_cor(f64, alg)
    end
end

end
