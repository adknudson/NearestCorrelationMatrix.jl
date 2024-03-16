# https://www.polyu.edu.hk/ama/profile/dfsun/

using LinearAlgebra, Tullio, BlockArrays




eigen_sym(X::AbstractMatrix{T}) where {T<:Real} = eigen(Symmetric(X), sortby=x->-x)
eigen_sym(X::Symmetric{T})      where {T<:Real} = eigen(X,            sortby=x->-x)

function eigen_sym(X::Symmetric{Float16})
    E = eigen(X, sortby=x->-x)
    values = convert(AbstractVector{Float16}, E.values)
    vectors = convert(AbstractMatrix{Float16}, E.vectors)
    return Eigen(values, vectors)
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
Generate the diagonal preconditioner, `c`

- `W`: the matrix returned from `omega_matrix`
- `P`: the eigenvectors of `X`
"""
function precondition_matrix(W, P)
    T = eltype(P)
    r, s = size(W)
    n = size(P, 1)

    r == 0 || r == n && return ones(T, n)

    Ω = PseudoBlockMatrix{T}(undef, [r,s], [r,s])
    view(Ω, Block(1,1)) .= one(T)
    view(Ω, Block(2,2)) .= zero(T)
    view(Ω, Block(1,2)) .= W
    view(Ω, Block(2,1)) .= W'

    Q = P .* P
    M = Ω * Q

    @tullio v[i] := dot(@view(Q[:,i]), @view(M[:,i]))

    ϵ = sqrt(eps(T))
    replace!(x -> max(x, ϵ), v)

    return v
end




"""
- `b`: a vector of (1-τ)'s
- `v`: the diagonal preconditioner
- `W`: the matrix returned from `omega_matrix`
- `P`: the eigenvectors of `X`
"""
function preconditioned_cg(b, v, W, P, tol, maxiter)
    r = copy(b) # initial residual

    norm_b = norm(b)
    tol_b = tol * norm_b

    p = zero(b) # final search direction

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
            α = rz1 / den
            p .+= α * d
            r .-= α * w
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




function newton_cor(
    A::AbstractMatrix{T};
    tau = eps(T),
    tol = sqrt(eps(T)),
    maxiter = 200
) where {T}
    G = Symmetric(A)
    n = size(A, 1)
    tau = max(tau, zero(T))
    error_tol = max(eps(T), tol)

    b = ones(T, n) .- tau
    A[diagind(A)] .-= tau

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

    λ, P = eigen_sym(X)
    f0 = dual_gradient!(∇fy, y, λ, P, b0)

    val_G = norm(G)^2 / 2
    val_dual = val_G - f0

    primal_feasible_solution!(X, λ, P, b0)

    val_primal = norm(X - G)^2 / 2
    gap = (val_primal - val_dual) / (1 + abs(val_primal) + abs(val_dual))

    f = f0
    b .= b0 - ∇fy

    norm_b = norm(b)
    norm_b0 = norm(b0) + 1
    norm_b_rel = norm_b / norm_b0

    W = omega_matrix(λ)

    k = 0
    while gap > error_tol && norm_b_rel > error_tol && k < maxiter
        v .= precondition_matrix(W, P)
        d .= preconditioned_cg(b, v, W, P, 1e-2, 200)

        slope = dot(∇fy - b0, d)
        y .= x0 + d
        X .= G + Diagonal(y)
        λ, P = eigen_sym(X)

        f = dual_gradient!(∇fy, y, λ, P, b0)

        # begin line search loop
        m = 0
        while (m < 20) && (f > f0 + 1e-4 * slope / 2^m + 1e-6)
            m += 1

            y .= x0 + d / 2^m
            X .= G + Diagonal(y)
            λ, P = eigen_sym(X)
            f = dual_gradient!(∇fy, y, λ, P, b0)
        end # end line search loop

        x0 .= y
        f0 = f

        val_dual = val_G - f0
        primal_feasible_solution!(X, λ, P, b0)

        val_primal = norm(X - G)^2 / 2
        gap = (val_primal - val_dual) / (1 + abs(val_primal) + abs(val_dual))

        k += 1
        b .= b0 - ∇fy
        norm_b = norm(b)
        norm_b_rel = norm_b / norm_b0

        W = omega_matrix(λ)
    end

    A[diagind(A)] .+= tau # restore the diagonal of A
    X[diagind(X)] .+= tau

    @info "Ended after $k iterations"

    return X
end
