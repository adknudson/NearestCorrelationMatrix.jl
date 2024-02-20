"""
    nearest_cor!(R::Matrix{<:AbstractFloat}, alg::NearestCorrelationAlgorithm)

Return the nearest positive definite correlation matrix to `R`. This method updates `R` in place.
"""
function nearest_cor! end

nearest_cor!(X::Matrix{Float64}, alg::NearestCorrelationAlgorithm) = _nearest_cor!(X, alg)
nearest_cor!(X::Matrix{Float32}, alg::NearestCorrelationAlgorithm) = _nearest_cor!(X, alg)

function nearest_cor!(X::Matrix{Float16}, alg::NearestCorrelationAlgorithm)
    @warn "Float16s are converted to Float32s before computing the nearest correlation." maxlog=1

    R = Float32.(X)
    nearest_cor!(R, alg)
    X .= Float16.(R)

    return X
end

function nearest_cor!(X::Matrix{BigFloat}, alg::NearestCorrelationAlgorithm)
    @warn "BigFloats are not fully supported by LinearAlgebra. Converting to Float64s." maxlog=1

    R = Float64.(X)
    nearest_cor!(R, alg)
    X .= BigFloat.(R)

    return X
end

nearest_cor!(X) = nearest_cor!(X, default_alg())



"""
    nearest_cor(R::AbstractMatrix{<:AbstractFloat}, alg::NearestCorrelationAlgorithm)

Return the nearest positive definite correlation matrix to `R`.

# Examples
```jldoctest
julia> import LinearAlgebra: isposdef

julia> r = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
];

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

nearest_cor(X, alg::NearestCorrelationAlgorithm) = nearest_cor!(copy(X), alg)
nearest_cor(X) = nearest_cor(X, default_alg())
