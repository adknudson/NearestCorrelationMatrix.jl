module NearestCorrelationMatrix

using LinearAlgebra
using Statistics: clampcor
using SnoopPrecompile


abstract type NearestCorrelationAlgorithm end


"""
    nearest_cor!(R::AbstractMatrix{<:AbstractFloat}, alg::NearestCorrelationAlgorithm)

Return the nearest positive definite correlation matrix to `R`. This method updates `R` in place.
"""
function nearest_cor! end



"""
    nearest_cor(R::AbstractMatrix{<:AbstractFloat}, alg::NearestCorrelationAlgorithm=Newton())

Return the nearest positive definite correlation matrix to `R`.

# Examples
```jldoctest
julia> import LinearAlgebra: isposdef

julia> r = [1.00 0.82 0.56 0.44; 0.82 1.00 0.28 0.85; 0.56 0.28 1.00 0.22; 0.44 0.85 0.22 1.00]
4×4 Matrix{Float64}:
 1.0   0.82  0.56  0.44
 0.82  1.0   0.28  0.85
 0.56  0.28  1.0   0.22
 0.44  0.85  0.22  1.0

julia> isposdef(r)
false

julia> p = nearest_cor(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(p)
true
```
"""
function nearest_cor end



nearest_cor!(X, alg::NearestCorrelationAlgorithm) = _nearest_cor!(X, alg)
nearest_cor!(X) = nearest_cor!(X, Newton())

function nearest_cor!(X::AbstractMatrix{Float16}, alg::NearestCorrelationAlgorithm)
    @warn "Float16s are converted to Float32s before computing the nearest correlation" maxlog=1
    R = _nearest_cor!(Float32.(X), alg)
    X .= Float16.(R)
    return X
end

nearest_cor(X, alg::NearestCorrelationAlgorithm) = nearest_cor!(copy(X), alg)
nearest_cor(X) = nearest_cor(X, Newton())


export nearest_cor, nearest_cor!, Newton



include("common.jl")
include("newton_method.jl")


@precompile_setup begin
    f16 = rand(Float16, 10, 10)
    f32 = rand(Float32, 10, 10)
    f64 = rand(Float64, 10, 10)
    
    @precompile_all_calls begin
        alg = Newton(τ = 1e-10, tol=1e-8)

        nearest_cor!(f16)
        nearest_cor!(f32)
        nearest_cor!(f64)

        nearest_cor!(f16, alg)
        nearest_cor!(f32, alg)
        nearest_cor!(f64, alg)

        nearest_cor(f16)
        nearest_cor(f32)
        nearest_cor(f64)

        nearest_cor(f16, alg)
        nearest_cor(f32, alg)
        nearest_cor(f64, alg)
    end
end

end
