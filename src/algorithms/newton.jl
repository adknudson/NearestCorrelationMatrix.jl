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


autotune(::Type{Newton}, prob::NCMProblem) = _autotune(Newton, prob.A)
_autotune(::Type{Newton}, A::AbstractMatrix{Float64}) = Newton(tau=1e-12)

function _autotune(::Type{Newton}, A::AbstractMatrix{Float32})
    n = size(A, 1)

    tau = if n ≤ 50
        7.5e-6
    elseif n ≤ 100
        1e-5
    elseif n ≤ 500
        5e-5
    elseif n ≤ 1000
        1e-4
    else
        5e-5
    end

    Newton(tau=tau)
end

function _autotune(::Type{Newton}, A::AbstractMatrix{Float16})
    n = size(A, 1)

    tau = if n ≤ 25
        5e-3
    elseif n ≤ 50
        1e-2
    elseif n ≤ 500
        5e-2
    else
        1e-1
    end

    Newton(tau=tau)
end


modifies_in_place(::Newton) = false
supports_symmetric(::Newton) = true
supports_float16(::Newton) = true
supports_parameterless_construction(::Newton) = true


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
