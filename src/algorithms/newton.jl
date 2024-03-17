# https://www.polyu.edu.hk/ama/profile/dfsun/

"""
    NewtonNew(; tau, tol_cg, tol_ls, iter_cg, iter_ls)

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
    tau::Real = eps(),
    tol_cg::Real = 1e-2,
    tol_ls::Real = 1e-4,
    iter_cg::Int = 200,
    iter_ls::Int = 20,
    kwargs...
)
    return Newton(tau, tol_cg, tol_ls, iter_cg, iter_ls, args, kwargs)
end


autotune(::Type{Newton}, prob::NCMProblem) = autotune(Newton, prob.A)
autotune(::Type{Newton}, A::AbstractMatrix{Float64}) = Newton(tau=1e-12)
autotune(::Type{Newton}, A::AbstractMatrix{Float32}) = Newton(tau=1e-8)
autotune(::Type{Newton}, A::AbstractMatrix{Float16}) = Newton(tau=1e-4)


default_iters(::Newton, A) = size(A,1)

modifies_in_place(::Newton) = false


function CommonSolve.solve!(solver::NCMSolver, alg::Newton; kwargs...)
    G = Symmetric(solver.A)
    n = size(G, 1)
    T = eltype(G)

    tau = max(alg.tau, zero(T))
    error_tol = max(eps(T), solver.reltol)

    b = ones(T, n) .- tau
    G[diagind(G)] .-= tau

    # original diagonal
    b0 = copy(b)
    # initial dual solution
    y = zeros(T, n)
    # gradient of the dual
    ∇fy = zeros(T, n)
    # previous dual solution
    x0 = copy(y)
    # diagonal preconditioner
    v = ones(T, n)
    # step direction
    d = zeros(T, n)
    # initial primal solution
    X = G + Diagonal(y)
    # full omega matrix
    Ω = Matrix{T}(undef, n, n)

    λ, P = eigen_sym(X)
    f0 = dual_gradient!(∇fy, y, λ, P, b0)
    f = f0
    b .= b0 - ∇fy

    val_G = norm(G)^2 / 2
    val_dual = val_G - f0

    primal_feasible_solution!(X, λ, P, b0)

    val_primal = norm(X - G)^2 / 2
    gap = (val_primal - val_dual) / (1 + abs(val_primal) + abs(val_dual))

    norm_b = norm(b)
    norm_b0 = norm(b0) + 1
    norm_b_rel = norm_b / norm_b0

    k = 0
    while gap > error_tol && norm_b_rel > error_tol && k < solver.maxiters
        W = omega_matrix(λ)

        precondition_matrix!(v, W, P, Ω)
        preconditioned_cg!(d, b, v, W, P, alg.tol_cg, alg.iter_cg)

        slope = dot(∇fy - b0, d)
        y .= x0 + d
        X .= G + Diagonal(y)
        λ, P = eigen_sym(X)
        f = dual_gradient!(∇fy, y, λ, P, b0)

        m = 0
        while (m < alg.iter_ls) && (f > f0 + alg.tol_ls * slope / 2^m + 1e-6)
            m += 1
            y .= x0 + d / 2^m
            X .= G + Diagonal(y)
            λ, P = eigen_sym(X)
            f = dual_gradient!(∇fy, y, λ, P, b0)
        end

        x0 .= y
        f0 = f

        val_dual = val_G - f0
        primal_feasible_solution!(X, λ, P, b0)

        val_primal = norm(X - G)^2 / 2
        gap = (val_primal - val_dual) / (1 + abs(val_primal) + abs(val_dual))

        b .= b0 - ∇fy
        norm_b = norm(b)
        norm_b_rel = norm_b / norm_b0

        k += 1
    end

    G[diagind(G)] .+= tau # restore the diagonal of A
    X[diagind(X)] .+= tau

    cov2cor!(X)

    return build_ncm_solution(alg, X, gap, solver; iters=k)
end



"""
Compute

``∇f(y) = diag((A + diag(y))₊) - e``

and

``f(y) = ½‖(A + diag(y))₊‖² - eᵀy``

- `y`: current dual solution
- `λ`: vector of eigenvalues
- `P`: eigenvectors
- `b`: a vector of (1-τ)'s
"""
function dual_gradient!(∇fy, y, λ, P, b)
    r = count(>(0), λ)
    Pr = @view P[:, begin:r]
    λr = @view λ[begin:r]

    fy = dot(λr, λr) / 2 - dot(b, y)
    ∇fy .= diag(Pr * Diagonal(λr) * Pr')

    return fy
end


"""
Compute the primal feasible solution using PCA

- `X`: current primal solution
- `λ`: vector of eigenvalues
- `P`: eigenvectors
- `b`: a vector of (1-τ)'s
"""
function primal_feasible_solution!(X, λ, P, b)
    r = count(>(0), λ)
    n = size(X, 1)
    s = n - r

    if r == 0
        fill!(X, zero(eltype(X)))
    elseif r == 1
        P1 = @view P[:,1]
        λ1 = λ[1]
		mul!(X, P1, P1', λ1, 0)
    elseif r ≤ s
        Pr = @view P[:, begin:r]
		λr = sqrt(Diagonal(λ[begin:r]))
		Q = Pr * λr
        mul!(X, Q, Q')
    elseif r < n
        Ps = @view P[:, r+1:end]
		λs = sqrt(Diagonal(-λ[r+1:end]))
		Q = Ps * λs
        mul!(X, Q, Q', 1, 1)
    end

    # scales `X` diagonal to `b` without changing PSD property
    d = max.(diag(X), b)
    X[diagind(X)] .= d
    d .= sqrt.(b ./ d)
    d2 = d * d'
    X .*= d2

    return X
end

primal_feasible_solution!(X::Symmetric, args...) = primal_feasible_solution!(X.data, args...)


"""
Generates the second block of M(y), the essential part of the first-order difference of `d`

- `λ`: the eigenvalues of `X`
"""
function omega_matrix(λ)
    r = count(>(0), λ)
    n = length(λ)

    r == 0 && return zeros(eltype(λ), 0, 0)
    r == n && return ones(eltype(λ), n, n)

    λr = @view λ[begin:r]
    λs = @view λ[r+1:end]

    @tullio W[i,j] := λr[i] / (λr[i] - λs[j])

    return W
end



function full_omega_matrix!(Ω, W)
    T = eltype(Ω)
    r = size(W, 1)

    fill!(@view(Ω[begin:r,begin:r]), one(T))
    fill!(@view(Ω[r+1:end,r+1:end]), zero(T))
    @view(Ω[begin:r, r+1:end]) .= W
    @view(Ω[r+1:end, begin:r]) .= W'

    return Ω
end


perturb(::Type{T}) where {T<:Real} = sqrt(eps(eltype(T))) / 4
perturb(::AbstractArray{T,N}) where {T,N} = perturb(T)
perturb(::T) where {T<:AbstractFloat} = perturb(T)


"""
Generate the Jacobian product with `d`:

``F'(y)(d) = V(y)d``

- `d`: the step direction in the CG method
- `W`: the matrix returned from `omega_matrix`
- `P`: the eigenvectors of `X`
"""
function jacobian_matrix!(Vd, d, W, P)
    T = eltype(Vd)
    n = length(d)
    r, s = size(W)


    if r == n
        Vd .= (1 + perturb(d)) * d
        return Vd
    end

    if r == 0
        fill!(Vd, zero(T))
        return Vd
    end

    Pr = @view P[:, begin:r]
    Ps = @view P[:, r+1:end]

    Wrs = W .* (Pr' * Diagonal(d) * Ps)
    PW = Pr * Wrs
    hh = 2 * sum(PW .* Ps, dims=2)

    if r < s
        PrPr = Pr * Pr'
        Vd .= (PrPr .* PrPr) * d + hh + perturb(d) * d
    else
        PsPs = Ps * Ps'
        Vd .= d + (PsPs .* PsPs) * d + hh - (2 * d .* diag(PsPs)) + perturb(d) * d
    end

    return Vd
end


"""
Generate the diagonal preconditioner, `v`

- `v`: the diagonal vector to write into
- `W`: the matrix returned from `omega_matrix`
- `P`: the eigenvectors of `X`
- `Ω`: pre-allocated data for the full Omega matrix
"""
function precondition_matrix!(v, W, P, Ω)
    T = eltype(P)
    r = size(W, 1)
    n = size(P, 1)

    if r == 0 || r == n
        fill!(v, one(T))
        return v
    end

    full_omega_matrix!(Ω, W)

    Q = P .* P
    M = Ω * Q

    @tullio v[i] = dot(@view(Q[:,i]), @view(M[:,i]))

    ϵ = sqrt(eps(T))
    replace!(x -> max(x, ϵ), v)

    return v
end


"""
- `p`: the solution vector to write into
- `b`: a vector of (1-τ)'s
- `v`: the diagonal preconditioner
- `W`: the matrix returned from `omega_matrix`
- `P`: the eigenvectors of `X`
"""
function preconditioned_cg!(p, b, v, W, P, tol, maxiter)
    T = eltype(p)
    r = copy(b) # initial residual

    norm_b = norm(b)
    tol_b = tol * norm_b

    fill!(p, zero(T))

    z = r ./ v
    rz1 = dot(r, z)
    rz2 = one(rz1)
    d = copy(z)
    w = similar(d)

    k = 0

    for k = 1:maxiter
        if k > 1
            β = rz1 / rz2
            d .= z + β * d
        end

        jacobian_matrix!(w, d, W, P)
        den = dot(d, w)

        if den ≤ 0
            p .= d / norm(d)
            break
        else
            a = rz1 / den
            p .+= a * d
            r .-= a * w
        end

        z .= r ./ v

        if norm(r) ≤ tol_b
            break
        end

        rz2 = rz1
        rz1 = dot(r, z)
    end

    return p
end
