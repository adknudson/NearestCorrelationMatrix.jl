module NearestCorrelationMatrix

using LinearAlgebra
using Statistics: clampcor
using SnoopPrecompile


abstract type NearestCorrelationAlgorithm end


default_alg() = Newton()


export nearest_cor, nearest_cor!, Newton, AlternatingProjection


include("common.jl")
include("nearest_correlation.jl")
include("newton_method.jl")
include("alternating_projection.jl")


@precompile_setup begin
    function make_data(T, n)
        x = 2 * rand(T, n, n) .- one(T)
        _set_diag!(x, one(T))
        _make_symmetric!(x)
        return x
    end

    f32 = make_data(Float32, 5)
    f64 = make_data(Float64, 5)

    @precompile_all_calls begin
        alg = Newton(τ = 1e-10, tol=1e-8)
        
        nearest_cor(f32, alg)
        nearest_cor(f64, alg)

        alg = AlternatingProjection()

        nearest_cor(f32, alg)
        nearest_cor(f64, alg)
    end
end

end
