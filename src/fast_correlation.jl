function _fast_pca!(X::Matrix{T}, λ::Vector{T}, P::Matrix{T}) where {T<:AbstractFloat}
    n = length(λ)
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        X .= zeros(T, n, n)
    elseif r == n
        # do nothing
    elseif r == 1 
        X .= (λ[1] * λ[1]) * (P[:,1] * P[:,1]')   
    elseif r ≤ s
        P₁   = @view P[:, 1:r]
        λ₁   = sqrt.(λ[1:r])
        P₁λ₁ = P₁ .* λ₁'
        X   .= P₁λ₁ * P₁λ₁'
    else
        P₂   = @view P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* λ₂'
        X   .= X .+ P₂λ₂ * P₂λ₂'
    end

    return X
end



"""
    cor_fast_posdef!(R::Matrix{<:AbstractFloat} [, τ::Real=1e-6])

Same as [`cor_fastPD`](@ref), but saves space by overwriting the input `R`
instead of creating a copy.

See also: [`cor_fast_posdef`](@ref), [`cor_nearest_posdef`](@ref)

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

julia> cor_fast_posdef!(r)


julia> isposdef(r)
true
```
"""
function cor_fast_posdef!(R::AbstractMatrix{T}, τ::Real=1e-6) where {T<:Union{Float32, Float64}}
    τ  = max(eps(T), T(τ))
    
    R .= Symmetric(R, :U)
    R[diagind(R)] .= (one(T) - τ)

    λ, P = eigen(R)
    reverse!(λ)
    reverse!(P, dims=2)

    _fast_pca!(R, λ, P)

    R[diagind(R)] .+= τ
    _cov2cor!(R)
    return R
end

function cor_fast_posdef!(R::AbstractMatrix{Float16}, τ::Real=1e-6)
    @warn "Float16s are converted to Float32s before computing a near correlation" maxlog=1
    Q = Float32.(R)
    cor_fast_posdef!(Q, τ)
    R .= Float16.(Q)
    return R
end



"""
    cor_fast_posdef(R::Matrix{<:AbstractFloat} [, τ::Real=1e-6])

Return a positive definite correlation matrix that is close to `R`. `τ` is a
tuning parameter that controls the minimum eigenvalue of the resulting matrix.
`τ` can be set to zero if only a positive semidefinite matrix is needed.

See also: [`cor_fast_posdef!`](@ref), [`cor_nearest_posdef`](@ref)

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

julia> p = cor_fast_posdef(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(p)
true
```
"""
cor_fast_posdef(R::AbstractMatrix{Float32}, τ::Real=1e-6) = cor_fast_posdef!(copy(R), τ)
cor_fast_posdef(R::AbstractMatrix{Float64}, τ::Real=1e-6) = cor_fast_posdef!(copy(R), τ)

function cor_fast_posdef(R::AbstractMatrix{Float16}, τ::Real=1e-6)
    @warn "Float16s are converted to Float32s before computing a near correlation" maxlog=1
    Q = Float32.(R)
    cor_fast_posdef!(Q, τ)
    return Float16.(Q)
end
