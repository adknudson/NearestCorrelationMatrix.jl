"""
    Newton(; kwargs...)

# Parameters
- `tau`: a tuning parameter controlling the smallest eigenvalue of the resulting matrix
- `tol_cg`: the tolerance used in the conjugate gradient method
- `tol_ls`: the tolerance used in the line search method
- `iter_cg`: the max number of iterations in the conjugate gradient method
- `iter_ls`: the max number of refinements during the Newton step
"""
struct Newton{A,K} <: NCMAlgorithm
    tau::Real
    tol_cg::Real
    tol_ls::Real
    iter_cg::Int
    iter_ls::Int
    args::A
    kwargs::K
end

function Newton(
    args...;
    tau::Real=sqrt(eps()),
    tol_cg::Real = 1e-2,
    tol_ls::Real = 1e-4,
    iter_cg::Int = 200,
    iter_ls::Int = 20,
    kwargs...
)
    return Newton(tau, tol_cg, tol_ls, iter_cg, iter_ls, args, kwargs)
end

default_iters(::Newton, A) = max(size(A,1), 10)


function CommonSolve.solve!(solver::NCMSolver, alg::Newton)
    R = solver.A
    n = size(R, 1)
    T = eltype(R)

    # Setup
    tau        = max(alg.tau, 0)
    tol_cg     = alg.tol_cg
    tol_ls     = alg.tol_ls
    iter_cg    = alg.iter_cg
    iter_ls    = alg.iter_ls
    err_tol    = solver.reltol
    inner_eps  = 1e-6

    b = ones(T, n)
    b .-= tau
    R[diagind(R)] .-= tau
    b₀ = copy(b)

    y    = zeros(T, n)
    x₀   = copy(y)
    X    = copy(R)
    λ, P = eigen_safe(Symmetric(X))

    f₀, Fy = _gradient(y, λ, P, b₀)
    f      = f₀
    b     .= b₀ - Fy

    _pca!(X, b₀, λ, P)

    val_R    = norm(R)^2 / 2
    val_dual = val_R - f₀
    val_obj  = norm(X - R)^2 / 2
    gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))

    norm_b  = norm(b)
    norm_b0 = norm(b₀) + 1
    norm_b_rel = norm_b / norm_b0

    k = 0
    c = zeros(T, n)
    d = zeros(T, n)

    while (gap > err_tol) && (norm_b_rel > err_tol) && (k < solver.maxiters)
        Ω₀ = _create_omega_matrix(λ)

        _precondition_matrix!(c, Ω₀, P)
        _pre_conjugate_gradient!(d, b, c, Ω₀, P, tol_cg, iter_cg)

        slope = dot(Fy - b₀, d)
        y    .= x₀ + d
        X    .= R + diagm(y)
        λ, P = eigen_safe(Symmetric(X))
        f, Fy = _gradient(y, λ, P, b₀)

        k_inner = 0
        while (k_inner ≤ iter_ls) && (f > f₀ + tol_ls * slope / 2^k_inner + inner_eps)
            k_inner += 1
            y    .= x₀ + d / 2^k_inner
            X    .= R + diagm(y)
            λ, P = eigen_safe(Symmetric(X))
            f, Fy = _gradient(y, λ, P, b₀)
        end

        x₀  = copy(y)
        f₀  = f

        _pca!(X, b₀, λ, P)
        val_dual = val_R - f₀
        val_obj  = norm(X - R)^2 / 2
        gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        b        = b₀ - Fy
        norm_b   = norm(b)
        norm_b_rel = norm_b / norm_b0

        k += 1
    end

    X[diagind(X)] .+= tau
    cov2cor!(X)

    return build_ncm_solution(alg, X, gap, solver; iters=k)
end



"""
    _gradient(y::Vector{T}, λ₀::Vector{T}, P::Matrix{T}, b₀::Vector{T}) where {T<:AbstractFloat}

Return f(yₖ) and ∇f(yₖ) where

f(y) = ½‖(A + diag(y))₊‖² - eᵀy

and

∇f(y) = Diag((A + diag(y))₊) - e
"""
function _gradient(
    y::AbstractVector{T},
    λ₀::AbstractVector{T},
    P::AbstractMatrix{T},
    b₀::AbstractVector{T}
) where {T<:AbstractFloat}
    r = sum(λ₀ .> 0)
    λ = copy(λ₀)
    n = length(y)

    r == 0 && return zero(T), zeros(T, n)

    λ[λ .< 0] .= zero(T)
    Fy = vec(sum((P .* transpose(λ)) .* P, dims=2))
    f  = T(0.5) * dot(λ, λ) - dot(b₀, y)

    return f, Fy
end



"""
    _pca!(X::Matrix{T}, b::Vector{T}, λ::Vector{T}, P::Matrix{T}) where {T<:AbstractFloat}
"""
function _pca!(
    X::AbstractMatrix{T},
    b::AbstractVector{T},
    λ::AbstractVector{T},
    P::AbstractMatrix{T}
) where {T<:AbstractFloat}
    project_psd!(X, λ, P)
    d  = diag(X)
    d .= max.(d, b)
    X[diagind(X)] .= d
    d .= sqrt.(b ./ d)
    d₂ = d * transpose(d)
    X .= X .* d₂
    return X
end



"""
    _pre_conjugate_gradient!(p::Vector{T}, b::Vector{T}, c::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}, tol::Real, num_iter::Int) where {T<:AbstractFloat}

Preconditioned conjugate gradient method to solve Vₖdₖ = -∇f(yₖ)
"""
function _pre_conjugate_gradient!(
    p::AbstractVector{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    Ω₀::AbstractMatrix{T},
    P::AbstractMatrix{T},
    tol::Real,
    num_iter::Int
) where {T<:AbstractFloat}
    fill!(p, zero(T))

    n = length(p)
    ϵ_b = T(tol) * norm(b)
    r   = copy(b)
    z   = r ./ c
    d   = copy(z)
    rz1 = sum(r .* z)
    rz2 = one(T)
    w   = zeros(T, n)

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
function _precondition_matrix!(
    c::AbstractVector{T},
    Ω₀::AbstractMatrix{T},
    P::AbstractMatrix{T}
) where {T<:AbstractFloat}
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
        H12 = transpose(H₁) * Ω₀
        c .= transpose(sum(H₁, dims=1)).^2 + 2 * sum(H12 .* transpose(H₂), dims=2)
    else
        H12 = (1.0 .- Ω₀) * H₂
        c .= transpose(sum(H, dims=1)).^2 - transpose(sum(H₂, dims=1)).^2 - 2 * transpose(sum(H₁ .* H12, dims=1))
    end

    ϵ = T(1e-8)
    c[c .< ϵ] .= ϵ
    return c
end



"""
    _create_omega_matrix(λ::Vector{T}) where {T<:AbstractFloat}

The Omega matrix is used in creating the preconditioner matrix.
"""
function _create_omega_matrix(λ::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(λ)
    r = sum(>(0), λ)
    s = n - r

    r == 0 && return zeros(T, 0, 0)
    r == n && return ones(T, n, n)

    M = zeros(T, r, s)
    λᵣ = @view λ[1:r]
    λₛ = @view λ[r+1:n]
    for j in eachindex(λₛ), i in eachindex(λᵣ)
        @inbounds M[i,j] = λᵣ[i] / (λᵣ[i] - λₛ[j])
    end

    return M
end



"""
    _jacobian!(w::Vector{T}, x::Vector{T}, Ω₀::Matrix{T}, P::Matrix{T}, n::Int) where {T<:AbstractFloat}

Create the Generalized Jacobian matrix for the Newton direction step, and store in `w`
"""
function _jacobian!(
    w::AbstractVector{T},
    x::AbstractVector{T},
    Ω₀::AbstractMatrix{T},
    P::AbstractMatrix{T},
    n::Int
) where {T<:AbstractFloat}
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
        Ω  = Ω₀ .* (transpose(H₁) * P₂)

        HT₁ = P₁ * transpose(P₁) * H₁ + P₂ * transpose(Ω)
        HT₂ = P₁ * Ω

        w .= sum(P .* [HT₁ HT₂], dims=2) + x .* perturbation
        return w
    else
        H₂ = diagm(x) * P₂
        Ω  = (1 .- Ω₀) .* (transpose(P₁) * H₂)

        HT₁ = P₂ * transpose(Ω)
        HT₂ = P₂ * transpose(H₂) * P₂ + P₁ * Ω

        w .= x .* (1 + perturbation) - sum(P .* [HT₁ HT₂], dims=2)
        return w
    end
end
