function preconditioned_newton(
    R::AbstractMatrix{T};
    tol=sqrt(eps(T)),
    linsolve=KrylovJL_MINRES(),
    linesearch=BackTracking()
) where {T}
    A = Symmetric(copy(R))
    A[diagind(A)] .= one(T)
    n = size(A, 1)

    y = zeros(T, n)

    ∇fy = dual_gradient(y, A)
    cache = nothing

    while norm(∇fy) > tol
        e = eigen(A + Diagonal(y), sortby=x -> -x)
        W = weighted_matrix(e.values)

        sol = find_step_direction(y, ∇fy, e.vectors, W, linsolve, cache)
        if isnothing(cache)
            cache = sol.cache
        end
        d = sol.u

        a = find_step_size(A, y, d, linesearch)
        y .+= a.*d
        ∇fy = dual_gradient(y, A)
    end

    X = dual_to_primal(y, A)
    return cov2cor(X)
end



function proj_psd(C::Symmetric{T}) where {T}
    λ, Q = eigen(C)
    replace!(x -> x < 0 ? 0 : x, λ)
    return Symmetric(Q * Diagonal(λ) * Q')
end

function dual_lp(y::AbstractVector{T}, A::Symmetric{T}) where {T}
    return norm(proj_psd(A + Diagonal(y)))^2 / 2 - sum(y)
end

function dual_gradient(y::AbstractVector{T}, A::Symmetric{T}) where {T}
    return diag(proj_psd(A + Diagonal(y))) .- one(T)
end

function dual_to_primal(y::AbstractVector{T}, A::Symmetric{T}) where {T}
    return proj_psd(A + Diagonal(y))
end

function cov2cor(X::Symmetric)
    D = inv(sqrt(Diagonal(X)))
    return Symmetric(D * X * D)
end

function weighted_matrix(λ::AbstractVector{T}) where {T}
    n = length(λ)
    r = count(>(0), λ)
    s = n - r

    λa = @view λ[begin:r]
    λg = @view λ[r+1:end]

    @tullio tau[i,j] := λa[i] / (λa[i] - λg[j])

    return Symmetric([ones(T, r, r) tau; tau' zeros(T, s, s)])
end

function jacobi_preconditioner(P::AbstractMatrix{T}, W::Symmetric{T}, tol::T) where {T}
    n = size(P, 1)

    Q = P .* P
    M = W * Q

    v = zeros(T, n)

    for i = 1:n
        @inbounds v[i] = dot(Q[:,i], M[:,i])
    end

    replace!(x -> max(x, tol), v)

    return v
end

function generalized_jacobian(
    h::AbstractVector{T},
    P::AbstractMatrix{T},
    W::Symmetric{T}
) where {T}
    return diag(P * (W .* (P' * Diagonal(h) * P)) * P')
end

struct VkWrapper{T}
    P::AbstractMatrix{T}
    W::AbstractMatrix{T}
end

function (V::VkWrapper)(du, u, p, t)
    du .= generalized_jacobian(u, V.P, V.W)
end

function make_Vk_op(y, P, W)
    V = VkWrapper(P, W)

    return FunctionOperator(V,
        similar(y),
        similar(y);
        islinear=true,
        isinplace=true,
        isposdef=true,
    )
end

function find_step_direction(
    y::AbstractVector{T},
    ∇fy::AbstractVector{T},
    P::AbstractMatrix{T},
    W::Symmetric{T},
    alg,
    cache=nothing
) where {T}
    V = make_Vk_op(y, P, W)
    D = jacobi_preconditioner(P, W, 1e-8) |> Diagonal |> sqrt

    if isnothing(cache)
        prob = LinearProblem(V, -∇fy)
        cache = init(prob, alg, y; Pl = D, Pr = D)
    else
        cache.A = V
        cache.b = -∇fy
        cache.Pl = D
        cache.Pr = D
    end

    sol = solve!(cache)

    return sol
end

struct LSFWrapper{T}
    A::AbstractMatrix{T}
end

(f::LSFWrapper)(u) = dual_lp(u, f.A)

struct LSGWrapper{T}
    A::AbstractMatrix{T}
end

function (g::LSGWrapper)(du, u)
    du .= dual_gradient(u, g.A)
    return du
end

struct LSFGWrapper{T}
    A::AbstractMatrix{T}
end

function (fg::LSFGWrapper)(du, u)
    C = proj_psd(fg.A + Diagonal(u))
    fx = norm(C)^2 / 2 - sum(u)
    du .= diag(C) .- 1
    return fx
end


function make_linesearch_funcs(A::Symmetric{T}) where {T}
    f   = LSFWrapper(A)
    g!  = LSGWrapper(A)
    fg! = LSFGWrapper(A)

    return (f, g!, fg!)
end

function find_step_size(
    A::Symmetric{T},
    y::AbstractVector{T},
    d::AbstractVector{T},
    linesearch
) where {T}
    f, g!, fg! = make_linesearch_funcs(A)

    x = copy(y)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)

    function ϕ(a)
        return f(x .+ a.*d)
    end

    function dϕ(a)
        g!(gvec, x .+ a.*d)
        return dot(gvec, d)
    end

    function ϕdϕ(a)
        phi = fg!(gvec, x .+ a.*d)
        dphi = dot(gvec, d)
        return (phi, dphi)
    end

    dϕ_0 = dot(d, gvec)

    a, _ = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
    return a
end
