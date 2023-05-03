module NearestCorrelationMatrix

using LinearAlgebra
using Statistics: clampcor
using SnoopPrecompile


abstract type NearestCorrelationAlgorithm end


default_alg() = Newton()


export nearest_cor, nearest_cor!, Newton



include("common.jl")
include("nearest_correlation.jl")
include("newton_method.jl")


@precompile_setup begin
    f16_inplace = rand(Float16, 10, 10)
    f32_inplace = rand(Float32, 10, 10)
    f64_inplace = rand(Float64, 10, 10)

    f16 = rand(Float16, 10, 10)
    f32 = rand(Float32, 10, 10)
    f64 = rand(Float64, 10, 10)

    @precompile_all_calls begin
        Newton(τ = 1e-10, tol=1e-8)
        default_alg()

        nearest_cor!(f16_inplace)
        nearest_cor!(f32_inplace)
        nearest_cor!(f64_inplace)

        nearest_cor(f16)
        nearest_cor(f32)
        nearest_cor(f64)
    end
end

end
