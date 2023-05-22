"""
    Newton

`τ` is a tuning parameter that controls the minimum eigenvalue of the resulting matrix, and can be 
set to zero if only a positive semidefinite matrix is needed.

# Parameters
- `τ`: a tuning parameter controlling the smallest eigenvalue of the resulting matrix
- `tol`: the tolerance used as a stopping condition during iterations
- `tol_cg`: the tolerance used in the conjugate gradient method
- `tol_ls`: the tolerance used in the line search method
- `iter_outer`: the max number of Newton steps
- `iter_inner`: the max number of refinements during the Newton step
- `iter_cg`: the max number of iterations in the conjugate gradient method
"""
struct Newton <: NearestCorrelationAlgorithm
    τ::Float64
    tol::Float64
    tol_cg::Float64
    tol_ls::Float64
    iter_outer::Int
    iter_inner::Int
    iter_cg::Int

    function Newton(; τ=1e-6, tol=1e-3, tol_cg=1e-2, tol_ls=1e-4, iter_outer=200, iter_inner=20, iter_cg=200)
        return new(
            max(zero(Float64), float(τ)),
            max(eps(Float64), float(tol)),
            tol_cg,
            tol_ls,
            iter_outer,
            iter_inner,
            iter_cg,
        )
    end
end



function _nearest_cor!(R::Matrix{T}, alg::Newton) where {T<:AbstractFloat}
    n = _prep_matrix!(R)
    
    # Setup 
    onehalf    = T(0.5)
    τ          = T(alg.τ)
    iter_outer = alg.iter_outer
    iter_inner = alg.iter_inner
    iter_cg    = alg.iter_cg
    tol_cg     = T(alg.tol_cg)
    tol_ls     = T(alg.tol_ls)
    err_tol    = T(alg.tol)
    inner_eps  = T(1e-6)

    b = ones(T, n)
    if τ > zero(T)
        b .-= τ
        R[diagind(R)] .-= τ
    end
    b₀ = copy(b)

    y    = zeros(T, n)  # [n,1]
    x₀   = copy(y)      # [n,1]
    X    = copy(R)      # [n,n]
    λ, P = eigen(Symmetric(X)) # [n,1], [n,n]
    reverse!(λ)         # [n,1]
    reverse!(P; dims=2) # [n,n]

    f₀, Fy = _gradient(y, λ, P, b₀) # [1], [n,1]
    f      = f₀      # [1]
    b     .= b₀ - Fy # [n,1]

    _pca!(X, b₀, λ, P) # [n,n]
    
    val_R    = onehalf * norm(R)^2
    val_dual = val_R - f₀
    val_obj  = onehalf * norm(X - R)^2
    gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
    
    norm_b  = norm(b)
    norm_b0 = norm(b₀) + 1
    norm_b_rel = norm_b / norm_b0
    
    k = 0
    c = zeros(T, n)
    d = zeros(T, n)
    
    while (gap > err_tol) && (norm_b_rel > err_tol) && (k < iter_outer)
        Ω₀ = _create_omega_matrix(λ) # [r,s]

        _precondition_matrix!(c, Ω₀, P)                           # [n,1]
        _pre_conjugate_gradient!(d, b, c, Ω₀, P, tol_cg, iter_cg) # [n,1]

        slope = dot(Fy - b₀, d)    # [1]
        y    .= x₀ + d                 # [n,1]
        X    .= R + diagm(y)           # [n,n]
        λ, P = eigen(Symmetric(X))     # [n,1], [n,n]
        reverse!(λ)                    # [n,1]
        reverse!(P; dims=2)            # [n,n]
        f, Fy = _gradient(y, λ, P, b₀) # [1], [n,1]

        k_inner = 0
        while (k_inner ≤ iter_inner) && (f > f₀ + tol_ls * slope * onehalf^k_inner + inner_eps)
            k_inner += 1
            y    .= x₀ + d * onehalf^k_inner # [n,1]
            X    .= R + diagm(y)             # [n,n]
            λ, P = eigen(Symmetric(X))       # [n,1], [n,n]
            reverse!(λ)                      # [n,1]
            reverse!(P; dims=2)              # [n,n]
            f, Fy = _gradient(y, λ, P, b₀)   # [1], [n,1]
        end

        x₀  = copy(y) # [n,1]
        f₀  = f       # [1]

        _pca!(X, b₀, λ, P)    # [n,n]
        val_dual = val_R - f₀ # [1]
        val_obj  = onehalf * norm(X - R)^2
        gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        b        = b₀ - Fy
        norm_b   = norm(b)
        norm_b_rel = norm_b / norm_b0

        k += 1
    end

    X[diagind(X)] .+= τ
    _cov2cor!(X)
    copyto!(R, X)
    return R
end



"""
    _gradient(y::Vector{T}, λ₀::Vector{T}, P::Matrix{T}, b₀::Vector{T}) where {T<:AbstractFloat}

Return f(yₖ) and ∇f(yₖ) where

```math
f(y) = \\frac{1}{2} \\Vert (A + diag(y))_+ \\Vert_{F}^{2} - e^{T}y
```

and 

```math
\\nabla f(y) = Diag((A + diag(y))_+) - e
```
"""
function _gradient(y::Vector{T}, λ₀::Vector{T}, P::Matrix{T}, b₀::Vector{T}) where {T<:AbstractFloat}
    r = sum(λ₀ .> 0)
    λ = copy(λ₀)
    n = length(y)

    r == 0 && return zero(T), zeros(T, n)
    
    λ[λ .< 0] .= zero(T)
    Fy = vec(sum((P .* λ') .* P, dims=2))
    f  = T(0.5) * dot(λ, λ) - dot(b₀, y)

    return f, Fy
end



"""
    _pca!(X::Matrix{T}, b::Vector{T}, λ::Vector{T}, P::Matrix{T}) where {T<:AbstractFloat}
"""
function _pca!(X::Matrix{T}, b::Vector{T}, λ::Vector{T}, P::Matrix{T}) where {T<:AbstractFloat}
    n = length(b)
    r = sum(>(0), λ)
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
        P₁λ₁ = P₁ .* λ₁' # each row of P₁ times λ₁
        X   .= P₁λ₁ * P₁λ₁'
    else
        P₂   = @view P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* λ₂' # each row of P₂ times λ₂
        X   .= X .+ P₂λ₂ * P₂λ₂'
    end

    d  = diag(X)
    d .= max.(d, b)
    X[diagind(X)] .= d
    d .= sqrt.(b ./ d)
    d₂ = d * d'
    X .= X .* d₂
    X
end



"""
    _pre_conjugate_gradient!(p::Vector{T}, b::Vector{T}, c::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}, tol::Real, num_iter::Int) where {T<:AbstractFloat}

Preconditioned conjugate gradient method to solve Vₖdₖ = -∇f(yₖ)
"""
function _pre_conjugate_gradient!(p::Vector{T}, b::Vector{T}, c::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}, tol::Real, num_iter::Int) where {T<:AbstractFloat}
    fill!(p, zero(T))

    n = length(p)
    ϵ_b = T(tol) * norm(b) # [1]
    r   = copy(b)          # [n,1]
    z   = r ./ c           # [n,1]
    d   = copy(z)          # [n,1]
    rz1 = sum(r .* z)      # [1]
    rz2 = one(T)           # [1]
    w   = zeros(T, n)      # [n,1]

    for k in 1:num_iter
        if k > 1
            d .= z + d * (rz1 / rz2)
        end

        _jacobian!(w, d, Ω₀, P, n)

        denom = sum(d .* w)
        
        denom ≤ 0 && return d / norm(d)
        
        a = rz1 / denom
        p .+= a*d
        r .-= a*w
        
        norm(r) ≤ ϵ_b && return p
        
        z .= r ./ c
        rz2, rz1 = rz1, dot(r, z)
    end
    
    return p
end



"""
    _precondition_matrix!(c::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}) where {T<:AbstractFloat}

Create the preconditioner matrix used in solving the linear system `Vₖdₖ = -∇f(yₖ)` in the conjugate gradient method. Stores the result in `c`
"""
function _precondition_matrix!(c::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}) where {T<:AbstractFloat}
    r, s = size(Ω₀)
    n = length(c)

    if r == 0 || r == n 
        fill!(c, one(T))
        return c
    end

    H  = transpose(P .* P)
    H₁ = @view H[1:r,:]
    H₂ = @view H[r+1:n,:]

    if r < s
        H12 = H₁' * Ω₀
        c .= sum(H₁, dims=1)'.^2 + 2 * sum(H12 .* H₂', dims=2)
    else
        H12 = (1.0 .- Ω₀) * H₂
        c .= sum(H, dims=1)'.^2 - sum(H₂, dims=1)'.^2 - 2 * sum(H₁ .* H12, dims=1)'
    end

    ϵ = T(1e-8)
    c[c .< ϵ] .= ϵ
    return c
end



"""
    _create_omega_matrix(λ::Vector{T}) where {T<:AbstractFloat}

The Omega matrix is used in creating the preconditioner matrix.
"""
function _create_omega_matrix(λ::Vector{T}) where {T<:AbstractFloat}
    n = length(λ)
    r = sum(>(0), λ)
    s = n - r

    r == 0 && return zeros(T, 0, 0)
    r == n && return ones(T, n, n)
    
    M = zeros(T, r, s)
    λᵣ = @view λ[1:r]
    λₛ = @view λ[r+1:n]
    @inbounds for j in eachindex(λₛ), i in eachindex(λᵣ)
        M[i,j] = λᵣ[i] / (λᵣ[i] - λₛ[j])
    end

    return M
end



"""
    _jacobian!(w::Vector{T}, x::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}, n::Int) where {T<:AbstractFloat}

Create the Generalized Jacobian matrix for the Newton direction step, and store in `w`
"""
function _jacobian!(w::Vector{T}, x::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}, n::Int) where {T<:AbstractFloat}
    r, s = size(Ω₀)
    perturbation = T(1e-10)

    if r == 0
        fill!(w, zero(T))
        return w
    end

    if r == n
        copyto!(w, x .* (1 + perturbation))
        return w
    end

    P₁ = @view P[:, 1:r]
    P₂ = @view P[:, r+1:n]

    if r < s
        H₁ = diagm(x) * P₁
        Ω  = Ω₀ .* (H₁' * P₂)

        HT₁ = P₁ * P₁' * H₁ + P₂ * Ω'
        HT₂ = P₁ * Ω

        w .= sum(P .* [HT₁ HT₂], dims=2) + x .* perturbation
        return w
    else
        H₂ = diagm(x) * P₂
        Ω  = (1 .- Ω₀) .* (P₁' * H₂)

        HT₁ = P₂ * Ω'
        HT₂ = P₂ * H₂' * P₂ + P₁ * Ω

        w .= x .* (1 + perturbation) - sum(P .* [HT₁ HT₂], dims=2)
        return w
    end
end
