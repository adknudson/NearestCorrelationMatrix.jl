using LinearAlgebra
using Printf

"""
    correlation_matrix(G, b=nothing; tau=0.0, tol=1.0e-6, verbose=true)

Compute the nearest correlation matrix (or nearest positive semi-definite matrix with
diagonal `b`) to a given matrix `G` using the Semismooth Newton-CG method.

# Arguments
- `G`: Real matrix (n x n).
- `b`: Vector of target diagonal values (default: `ones(n)`).
- `tau`: Lower bound on eigenvalues (`X >= tau * I`, default: `0.0`).
- `tol`: Convergence tolerance (default: `1.0e-6`).
- `verbose`: Print iteration progress (default: `true`).

# Returns
- `X`: Optimal primal matrix solution.
- `y`: Optimal dual solution vector.
"""
function correlation_matrix(
        G::AbstractMatrix{T},
        b::Union{AbstractVector{T}, Nothing} = nothing;
        tau::Real = 0.0,
        tol::Real = 1.0e-6,
        verbose::Bool = true
    ) where {T <: Real}
    n, m = size(G)
    n == m || throw(DimensionMismatch("Matrix G must be square."))

    t0 = time()

    # Symmetrize input matrix G and setup target diagonal b0
    G_mod = (G + G') / 2
    b0 = b === nothing ? ones(T, n) : copy(b)

    if tau > 0
        b0 .-= tau
        G_mod -= tau * I
    end

    error_tol = max(1.0e-12, tol)

    if verbose
        println("*********************************************************************************")
        println("*  --- Semismooth Newton-CG method starts  --- ")
        println("*  Developed by Houduo Qi and Defeng Sun  ")
        println("*  Based on the algorithm in `A Quadratically Convergent Newton Method for ")
        println("*  Computing the Nearest Correlation Matrix' ")
        println("*  SIAM J. Matrix Anal. Appl. 28 (2006) 360--385. ")
        println("* --- This version: August 30, 2019 ---- ")
        println("*********************************************************************************")
    end

    Res_b = zeros(T, 300)
    norm_b0 = norm(b0)

    y = zeros(T, n)  # Initial dual point

    k = 0
    f_eval = 0

    Iter_Whole = 500
    Iter_inner = 20  # Max line search steps in Newton method
    maxit = 200       # Max PCG iterations
    iterk = 0
    tol_cg = 1.0e-2   # Relative accuracy for CG

    sigma_1 = 1.0e-4  # Line search tolerance parameter

    prec_time = 0.0
    pcg_time = 0.0
    eig_time = 0.0

    val_G = sum(abs2, G_mod) / 2

    X = G_mod + Diagonal(y)

    t_eig0 = time()
    P, lambda = eigen_descending(X)
    eig_time += time() - t_eig0

    f0, Fy = gradient_eval(y, lambda, P, b0, n)

    Initial_f = val_G - f0

    X = pca(X, lambda, P, b0, n)
    val_obj = sum(abs2, X - G_mod) / 2
    gap = (val_obj - Initial_f) / (1 + abs(Initial_f) + abs(val_obj))

    f = f0
    f_eval += 1
    b_vec = b0 - Fy
    norm_b = norm(b_vec)
    time_used = time() - t0
    eta = norm_b / (1 + norm_b0)
    Omega12 = omega_mat(P, lambda, n)
    x0 = copy(y)

    if verbose
        @printf("\n matrix dimension n = %d", n)
        @printf("\n ---------------------------------------------------------")
        @printf("\n  iter     pobj          dobj            relgap        etaorg        eta      time | cg_its inner_its")
        @printf(
            "\n   0   %- 5.4e     %-5.4e      %-3.2e        %3.2e     %3.2e     %3.1f",
            val_obj, Initial_f, gap, norm_b, eta, time_used
        )
    end

    while (eta > error_tol && k < Iter_Whole)
        size_plus = length(Omega12)

        t_prec0 = time()
        c = size_plus > 0 ? precond_matrix(Omega12, P, n) : ones(T, n)
        prec_time += time() - t_prec0

        if size_plus > 0
            t_pcg0 = time()
            d, flag, relres, iterk = pre_cg(b_vec, tol_cg, maxit, c, Omega12, P, n)
            pcg_time += time() - t_pcg0

            if flag != 0 && verbose
                println("\n *********************************************************************************")
                println("..... Warning: This step is not a completed Newton-CG step .....")
            end
        else
            d = b0 - Fy
            if verbose
                println("\n *********************************************************************************")
                println("..... Warning: This step uses the negative gradient direction .....")
            end
        end

        slope = dot(Fy - b0, d)

        y = x0 + d
        X = G_mod + Diagonal(y)

        t_eig0 = time()
        P, lambda = eigen_descending(X)
        eig_time += time() - t_eig0

        f, Fy = gradient_eval(y, lambda, P, b0, n)

        k_inner = 0
        while k_inner <= Iter_inner && f > f0 + sigma_1 * (0.5^k_inner) * slope + 1.0e-6
            k_inner += 1
            y = x0 + (0.5^k_inner) * d

            X = G_mod + Diagonal(y)

            t_eig0 = time()
            P, lambda = eigen_descending(X)
            eig_time += time() - t_eig0

            f, Fy = gradient_eval(y, lambda, P, b0, n)
        end

        f_eval += k_inner + 1
        x0 = copy(y)
        f0 = f
        val_dual = val_G - f0
        X = pca(X, lambda, P, b0, n)
        val_obj = sum(abs2, X - G_mod) / 2
        gap = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))

        k += 1
        b_vec = b0 - Fy
        norm_b = norm(b_vec)
        eta = norm_b / (1 + norm_b0)
        time_used = time() - t0

        if k <= length(Res_b)
            Res_b[k] = norm_b
        end

        if verbose
            @printf(
                "\n   %d   %- 5.4e     %-5.4e      %-3.2e        %3.2e     %3.2e     %3.1f | %d        %d",
                k, val_obj, val_dual, gap, norm_b, eta, time_used, iterk, k_inner
            )
        end

        Omega12 = omega_mat(P, lambda, n)
    end

    rank_X = count(>(0), lambda)
    Final_f = val_G - f
    if tau > 0
        X .+= tau * I
    end
    time_used = time() - t0

    if verbose
        @printf("\n =====================================================")
        @printf("\n eta = %3.2e, etaorg = %3.2e", eta, norm_b)
        @printf("\n per eig time = %5.4f", eig_time / f_eval)
        @printf("\n time per iteration = %5.4f", k > 0 ? time_used / k : 0.0)
        println()
        println("Newton-CG: Number of Iterations ============ $k")
        println("Newton-CG: Number of Function Evaluations == $f_eval")
        println("Newton-CG: Final Dual Objective Function value ========== $Final_f")
        println("Newton-CG: Final primal Objective Function value ======== $val_obj")
        println("Newton-CG: The final relative duality gap ========================  $gap")
        println("Newton-CG: The rank of the Optimal Solution - tau*I ================= $rank_X")

        println("Newton-CG: computing time for computing preconditioners == $prec_time")
        println("Newton-CG: computing time for linear systems solving (cgs time)            == $pcg_time")
        println("Newton-CG: computing time for  eigenvalue decompositions (calling eig time)== $eig_time")
        println("Newton-CG: computing time used for equal weights calibration ============================== $time_used")
    end

    return X, y
end

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
Compute eigen-decomposition sorted by eigenvalues in descending order.
"""
function eigen_descending(X::AbstractMatrix{T}) where {T <: Real}
    F = eigen(Symmetric(X))
    lambda = reverse(F.values)
    P = reverse(F.vectors, dims = 2)
    return P, lambda
end

"""
Compute dual objective function value `f` and its gradient `Fy`.
"""
function gradient_eval(
        y::AbstractVector{T},
        lambda::AbstractVector{T},
        P::AbstractMatrix{T},
        b0::AbstractVector{T},
        n::Int
    ) where {T <: Real}
    rankX = count(>(0), lambda)
    if rankX == 0
        Fy = zeros(T, n)
        f_val = zero(T)
    else
        P1 = @view P[:, 1:rankX]
        lambdanew = @view lambda[1:rankX]
        Fy = (P1 .^ 2) * lambdanew
        f_val = sum(abs2, lambdanew)
    end
    f = 0.5 * f_val - dot(b0, y)
    return f, Fy
end

"""
Use PCA to project onto the feasible PSD set with diagonal b0.
"""
function pca(
        X::AbstractMatrix{T},
        lambda::AbstractVector{T},
        P::AbstractMatrix{T},
        b0::AbstractVector{T},
        n::Int
    ) where {T <: Real}
    r = count(>(0), lambda)

    if r == 0
        X_out = zeros(T, n, n)
    elseif r == n
        X_out = copy(X)
    elseif r <= div(n, 2)
        lambda1 = sqrt.(@view lambda[1:r])
        P1 = P[:, 1:r] .* lambda1'
        X_out = P1 * P1'
    else
        lambda2 = sqrt.(-(@view lambda[(r + 1):n]))
        P2 = P[:, (r + 1):n] .* lambda2'
        X_out = X + P2 * P2'
    end

    # Scaling step to ensure positive semi-definiteness with diagonal b0
    d = diag(X_out)
    d .= max.(b0, d)
    for i in 1:n
        X_out[i, i] = d[i]
    end
    d .= sqrt.(b0 ./ d)
    X_out .= d .* X_out .* d'

    return X_out
end

"""
Compute first-order difference matrix Omega12.
"""
function omega_mat(P::AbstractMatrix{T}, lambda::AbstractVector{T}, n::Int) where {T <: Real}
    r = count(>(0), lambda)
    if r > 0
        if r == n
            return ones(T, n, n)
        else
            s = n - r
            dp = @view lambda[1:r]
            dn = @view lambda[(r + 1):n]
            return dp ./ (dp .+ abs.(dn'))
        end
    else
        return Matrix{T}(undef, 0, 0)
    end
end

"""
Preconditioned Conjugate Gradient algorithm (Hestenes & Stiefel).
"""
function pre_cg(
        b::AbstractVector{T},
        tol::Real,
        maxit::Int,
        c::AbstractVector{T},
        Omega12::AbstractMatrix{T},
        P::AbstractMatrix{T},
        n::Int
    ) where {T <: Real}
    r = copy(b)
    n2b = norm(b)
    tolb = tol * n2b
    p = zeros(T, n)
    flag = 1
    iterk = 0
    relres = 1000.0
    z = r ./ c
    rz1 = dot(r, z)
    rz2 = one(T)
    d = copy(z)

    for k in 1:maxit
        if k > 1
            beta = rz1 / rz2
            d .= z .+ beta .* d
        end
        w = jacobian_matrix(d, Omega12, P, n)
        denom = dot(d, w)
        iterk = k
        relres = norm(r) / n2b
        if denom <= 0
            p .= d ./ norm(d)
            break
        else
            alpha = rz1 / denom
            p .+= alpha .* d
            r .-= alpha .* w
        end
        z .= r ./ c
        if norm(r) <= tolb
            iterk = k
            relres = norm(r) / n2b
            flag = 0
            break
        end
        rz2 = rz1
        rz1 = dot(r, z)
    end

    return p, flag, relres, iterk
end

"""
Compute Jacobian matrix-vector product F'(y)(x).
"""
function jacobian_matrix(
        x::AbstractVector{T},
        Omega12::AbstractMatrix{T},
        P::AbstractMatrix{T},
        n::Int
    ) where {T <: Real}
    r, s = size(Omega12)
    if r > 0 && r < n
        P1 = @view P[:, 1:r]
        P2 = @view P[:, (r + 1):n]
        O12 = Omega12 .* (P1' * (P2 .* x))
        PO = (P1 * O12) .* P2
        hh = 2.0 .* vec(sum(PO, dims = 2))
        if r < div(n, 2)
            PP1 = P1 * P1'
            PP11 = PP1 .^ 2
            return PP11 * x .+ hh .+ 1.0e-10 .* x
        else
            PP2 = P2 * P2'
            PP22 = PP2 .^ 2
            return x .+ PP22 * x .+ hh .- 2.0 .* x .* diag(PP2) .+ 1.0e-10 .* x
        end
    elseif r == n
        return (1.0 + 1.0e-10) .* x
    else
        return zeros(T, n)
    end
end

"""
Compute diagonal preconditioner.
"""
function precond_matrix(
        Omega12::AbstractMatrix{T},
        P::AbstractMatrix{T},
        n::Int
    ) where {T <: Real}
    r, s = size(Omega12)
    c = ones(T, n)

    if r > 0
        H = (P .^ 2)'  # H[k, i] = P[i, k]^2
        if r < div(n, 2)
            H1 = @view H[1:r, :]
            H2 = @view H[(r + 1):n, :]
            H12 = H1' * Omega12  # n x s

            for i in 1:n
                sum_H1 = sum(@view H1[:, i])
                dot_H12 = dot(@view(H12[i, :]), @view(H2[:, i]))
                val = sum_H1^2 + 2.0 * dot_H12
                c[i] = max(1.0e-8, val)
            end
        elseif r < n
            Omega12_comp = ones(T, r, s) .- Omega12
            H1 = @view H[1:r, :]
            H2 = @view H[(r + 1):n, :]
            H12 = Omega12_comp * H2  # r x n

            for i in 1:n
                sum_H2 = sum(@view H2[:, i])
                dot_H12 = dot(@view(H1[:, i]), @view(H12[:, i]))
                alpha = sum(@view H[:, i])
                val = alpha^2 - (sum_H2^2 + 2.0 * dot_H12)
                c[i] = max(1.0e-8, val)
            end
        end
    end

    return c
end
