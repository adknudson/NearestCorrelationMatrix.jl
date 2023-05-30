struct AlternatingProjection <: NearestCorrelationAlgorithm 
    maxiter::Int
    tol::Float64

    function AlternatingProjection(; maxiter=100, tol=1e-7)
        return new(maxiter, tol)
    end
end


"""
    _project_psd(A::AbstractMatrix{T}) where {T<:AbstractFloat}

Project onto the positive semidefinite matrices.
"""
function _project_psd(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    λ, Q = eigen(Symmetric(A))
    return Q * Diagonal(max.(λ, zero(T))) * Q'
end

# eigen(Symmetric(Matrix{Float16})) returns a decomposition with Float32 eltype
# eigen(Matrix{Float16}) returns a decomposition with Float16 eltype
function _project_psd(A::AbstractMatrix{Float16})
    λ, Q = eigen(A)
    return Q * Diagonal(max.(λ, zero(Float16))) * Q'
end




function _project_symmetric(A::AbstractMatrix{T}, W::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    W05 = sqrt(W)
    return inv(W05) * _project_psd(Symmetric(W05 * A * W05)) * inv(W05)
end


function _project_unitdiag(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    Y = copy(X)
    Y[diagind(Y)] .= one(T)
    return Y
end


function _nearest_cor!(A::Matrix{T}, alg::AlternatingProjection) where {T<:AbstractFloat}
    n = _prep_matrix!(A)

    tol = T(alg.tol)

    W = Diagonal(ones(T, n))
    Sk = zeros(T, n, n)
    Rk = zeros(T, n, n)
    Xk = zeros(T, n, n)
    Yk = copy(A)

    i = 0
    converged = false
    conv = typemax(T)

    while i < alg.maxiter && !converged
        Rk .= Yk - Sk
        Xk .= _project_symmetric(Rk, W)
        Sk .= Xk - Rk
        Yk .= _project_unitdiag(Xk)

        conv = norm(Yk - Xk) / norm(Yk)
        converged = conv ≤ tol && isposdef(Yk)
        i += 1
    end

    A .= Yk
    _cov2cor!(Yk)
    copyto!(A, Yk)
    return A
end