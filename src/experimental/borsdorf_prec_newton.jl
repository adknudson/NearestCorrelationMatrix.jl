struct NewtonBorsdorf{Tlsolve,Tlsearch,A,K} <: NCMAlgorithm
    linsolve::Tlsolve
    linsearch::Tlsearch
    args::A
    kwargs::K
end

function NewtonBorsdorf(
    args...;
    linsolve=KrylovJL_GMRES(),
    linsearch=BackTracking(),
    kwargs...
)
    return NewtonBorsdorf(linsolve, linsearch, args, kwargs)
end

default_iters(::NewtonBorsdorf, A) = max(size(A,1), 10)


function CommonSolve.solve!(solver::NCMSolver, alg::NewtonBorsdorf)
    A = Symmetric(solver.A)
    n = size(A, 1)
    T = eltype(A)

    y = zeros(T, n)
    W = similar(A)

    ∇fy = dual_gradient(y, A)
    cache = nothing

    iters = 0

    while norm(∇fy) > solver.reltol && iters < solver.maxiters
        e = eigen(A + Diagonal(y), sortby=x -> -x)
        weighted_matrix!(W.data, e.values)

        sol = find_step_direction(y, ∇fy, e.vectors, W, alg.linsolve, cache)
        cache = sol.cache
        d = sol.u

        a = find_step_size(A, y, d, alg.linsearch)
        y .+= a.*d
        ∇fy .= dual_gradient(y, A)

        iters += 1
    end

    X = cov2cor(dual_to_primal(y, A))

    return build_ncm_solution(alg, X, norm(∇fy), solver; iters=iters)
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



function weighted_matrix!(W::AbstractMatrix, λ::AbstractVector{T}) where {T}
    r = count(>(0), λ)

    λa = @view λ[begin:r]
    λg = @view λ[r+1:end]

    @tullio tau[i,j] := λa[i] / (λa[i] - λg[j])

    fill!(@view(W[begin:r,begin:r]), one(T))
    fill!(@view(W[r+1:end,r+1:end]), zero(T))
    @view(W[begin:r, r+1:end]) .= tau
    @view(W[r+1:end, begin:r]) .= tau'

    return W
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
    ::Talg,
    cache::Tcache
) where {T,Talg,Tcache}
    V = make_Vk_op(y, P, W)
    D = jacobi_preconditioner(P, W, 1e-8) |> Diagonal |> sqrt

    cache.A = V
    cache.b = -∇fy
    cache.Pl = D
    cache.Pr = D

    sol = solve!(cache)

    return sol
end

function find_step_direction(
    y::AbstractVector{T},
    ∇fy::AbstractVector{T},
    P::AbstractMatrix{T},
    W::Symmetric{T},
    alg::Talg,
    ::Nothing
) where {T,Talg}
    V = make_Vk_op(y, P, W)
    D = jacobi_preconditioner(P, W, 1e-8) |> Diagonal |> sqrt

    prob = LinearProblem(V, -∇fy)
    cache = init(prob, alg, y; Pl = D, Pr = D)
    sol = solve!(cache)

    return sol
end



struct LSFWrapper{T}
    A::AbstractMatrix{T}
end

(w::LSFWrapper)(u) = dual_lp(u, w.A)

struct LSGWrapper{T}
    A::AbstractMatrix{T}
end

function (w::LSGWrapper)(du, u)
    du .= dual_gradient(u, w.A)
    return du
end

struct LSFGWrapper{T}
    A::AbstractMatrix{T}
end

function (w::LSFGWrapper)(du, u)
    C = proj_psd(w.A + Diagonal(u))
    fx = norm(C)^2 / 2 - sum(u)
    du .= diag(C) .- 1
    return fx
end

struct LSϕWrapper{Tx, Td}
    f::LSFWrapper
    x::Tx
    d::Td
end

(w::LSϕWrapper)(a) = w.f(w.x + a * w.d)

struct LSdϕWrapper{Tgvec, Tx, Td}
    g!::LSGWrapper
    gvec::Tgvec
    x::Tx
    d::Td
end

function (w::LSdϕWrapper)(a)
    w.g!(w.gvec, w.x + a * w.d)
end

struct LSϕdϕWrapper{Tgvec, Tx, Td}
    fg!::LSFGWrapper
    gvec::Tgvec
    x::Tx
    d::Td
end

function (w::LSϕdϕWrapper)(a)
    phi = w.fg!(w.gvec, w.x + a * w.d)
    dphi = dot(w.gvec, w.d)
    return (phi, dphi)
end

function find_step_size(
    A::Symmetric{T},
    y::AbstractVector{T},
    d::AbstractVector{T},
    linesearch::Tls
) where {T, Tls}
    f = LSFWrapper(A)
    g! = LSGWrapper(A)
    fg! = LSFGWrapper(A)

    x = copy(y)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)

    ϕ = LSϕWrapper(f, x, d)
    dϕ = LSdϕWrapper(g!, gvec, x, d)
    ϕdϕ = LSϕdϕWrapper(fg!, gvec, x, d)

    dϕ_0 = dot(d, gvec)

    a, _ = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
    return a
end
