using LinearAlgebra

export dual_gradient!,
    primal_feasible_solution!, omega_matrix!, precondition_matrix!, preconditioned_cg!

"""
    dual_gradient(∇fy, y, λ, P, b)

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
    primal_feasible_solution(X, λ, P, b)

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
        P1 = @view P[:, 1]
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

function primal_feasible_solution!(X::Symmetric, args...)
    return primal_feasible_solution!(X.data, args...)
end

"""
    omega_matrix!(Ω, λ)

Generates the second block of M(y), the essential part of the first-order difference of `d`

- `λ`: the eigenvalues of `X`
"""
function omega_matrix!(Ω, λ)
    T = eltype(Ω)
    n = length(λ)
    r = count(>(0), λ)
    s = n - r

    λr = @view λ[begin:r]
    λs = @view λ[r+1:end]

    @inbounds for j in 1:r, i in 1:r
        Ω[i, j] = one(T)
    end

    @inbounds for j in r+1:n, i in r+1:n
        Ω[i, j] = zero(T)
    end

    for j in 1:s
        for i in 1:r
            w = λr[i] / (λr[i] - λs[j])
            @inbounds Ω[i, j+r] = w
            @inbounds Ω[j+r, i] = w
        end
    end

    return @view Ω[begin:r, r+1:end]
end


"""
    perturb(x)

Compute a small perturbation value for the given input.
"""
function perturb end
perturb(::Type{T}) where {T} = sqrt(eps(eltype(T))) / 4
perturb(::T) where {T} = perturb(T)
perturb(::AbstractArray{T,N}) where {T,N} = perturb(T)

"""
    jacobian_matrix!(Vd, d, W, P)

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
    hh = 2 * sum(PW .* Ps; dims=2)

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
    precondition_matrix!(v, P, Ω)

Generate the diagonal preconditioner, `v`

- `v`: the diagonal vector to write into
- `P`: the eigenvectors of `X`
- `Ω`: the matrix returned from `omega_matrix`
"""
function precondition_matrix!(v, P, Ω, r)
    T = eltype(P)
    n = size(P, 1)

    if r == 0 || r == n
        fill!(v, one(T))
        return v
    end

    Q = P .* P
    M = Ω * Q

    for i in eachindex(v)
        @inbounds v[i] = dot(@view(Q[:, i]), @view(M[:, i]))
    end

    ϵ = sqrt(eps(T))
    replace!(x -> max(x, ϵ), v)

    return v
end

"""
    preconditioned_cg!(p, b, v, W, P, tol, maxiters)

- `p`: the solution vector to write into
- `b`: a vector of (1-τ)'s
- `v`: the diagonal preconditioner
- `W`: the partial matrix returned from `omega_matrix`
- `P`: the eigenvectors of `X`
"""
function preconditioned_cg!(p, b, v, W, P, tol, maxiters)
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

    for k in 1:maxiters
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
