"""
    DirectProjection(; τ=sqrt(eps()))

Single step projection of the input matrix into the set of correlation matrices. Useful when
a "close" correlation matrix is needed without concern for it being the most optimal.

# Parameters
- `τ`: a tuning parameter controlling the smallest eigenvalue of the resulting matrix
"""
Base.@kwdef struct DirectProjection <: NearestCorrelationAlgorithm
    τ::Real = 1e-6
end


function _nearest_cor!(X::AbstractMatrix{T}, alg::DirectProjection) where {T<:AbstractFloat}
    checkmat!(X)
    tau = max(T(alg.τ), eps(T))
    X[diagind(X)] .-= tau
    project_psd!(X)
    X[diagind(X)] .+= tau
    cov2cor!(X)
    return X
end
