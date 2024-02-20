"""
    DirectProjection(; τ=1e-6)

Single step projection of the input matrix into the set of correlation matrices. Useful when
a "close" correlation matrix is needed without concern for it being the most optimal.

# Parameters
- `τ`: a tuning parameter controlling the smallest eigenvalue of the resulting matrix
"""
Base.@kwdef struct DirectProjection <: NearestCorrelationAlgorithm
    τ::Float64 = 1e-6
end



_eigsym_reversed(X::AbstractMatrix{Float16}) = _eigen_reversed(X)
_eigsym_reversed(X::AbstractMatrix{Float32}) = _eigen_reversed(Symmetric(X))
_eigsym_reversed(X::AbstractMatrix{Float64}) = _eigen_reversed(Symmetric(X))



function _nearest_cor!(X::AbstractMatrix{T}, alg::DirectProjection) where {T<:AbstractFloat}
    n = _prep_matrix!(X)
    τ  = T(alg.τ)

    if τ > zero(T)
        X[diagind(X)] .-= τ
    end

    λ, P = _eigsym_reversed(X)

    r = count(>(0), λ)
    s = n - r

    if r == 0
        X .= zeros(T, n, n)
    elseif r == n
        # do nothing
    elseif r == 1
        X .= (λ[1] * λ[1]) * (P[:,1] * transpose(P[:,1]))
    elseif r ≤ s
        P₁   = @view P[:, 1:r]
        λ₁   = sqrt.(λ[1:r])
        P₁λ₁ = P₁ .* transpose(λ₁)
        X   .= P₁λ₁ * transpose(P₁λ₁)
    else
        P₂   = @view P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* transpose(λ₂)
        X   .= X .+ P₂λ₂ * transpose(P₂λ₂)
    end

    X[diagind(X)] .+= τ
    _cov2cor!(X)
    return X
end
