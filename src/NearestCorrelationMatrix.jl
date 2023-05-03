module NearestCorrelationMatrix

using LinearAlgebra
using Statistics: clampcor
using SnoopPrecompile


export cor_nearest_posdef, cor_fast_posdef, cor_fast_posdef!


include("common.jl")
include("nearest_correlation.jl")
include("fast_correlation.jl")


@precompile_setup begin
    f16 = rand(Float16, 10, 10)
    f32 = rand(Float32, 10, 10)
    f64 = rand(Float64, 10, 10)

    @precompile_all_calls begin
        cor_nearest_posdef(f16)
        cor_nearest_posdef(f32)
        cor_nearest_posdef(f64)
        cor_fast_posdef!(f16)
        cor_fast_posdef!(f32)
        cor_fast_posdef!(f64)
        cor_fast_posdef(f16)
        cor_fast_posdef(f32)
        cor_fast_posdef(f64)
    end
end

end
