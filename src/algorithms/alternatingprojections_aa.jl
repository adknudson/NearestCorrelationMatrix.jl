"""
    AlternatingProjectionsAA(; tau=0, m=2)

The Anderson-accelerated alternating projections algorithm, implementing Algorithm 5 of the
reference with the *full* Dykstra state ``z = (Y, S)`` (treated as a vector in ``R^{2n^2}``).
At each step the map ``\\hat g(z) = (Y', S')`` performs one Dykstra iteration —
``R = Y - S``, ``X = P_S(R)``, ``S' = X - R``, ``Y' = P_E(X)`` — and Anderson acceleration
(Algorithm 2) is applied to the fixed-point residual ``f(z) = \\hat g(z) - z``.

- `tau`: eigenvalue bound used in the PSD projection. When a fixed-element mask is present it
  plays the role of the §3.2 bound: the Dykstra step projects onto ``S_τ = {λ_min ≥ τ}`` so that
  the ``O(tol)`` diagonal/fixed-element corrections do not destroy positive definiteness. For
  PD output with exact fixed elements at a tight tolerance, use `tau ≈ 1e-8`. When unmasked (or
  `tau = 0`) this is the plain PSD cone.
- `m`: Anderson memory (number of history vectors). `m = 0` disables acceleration and reduces to
  plain Dykstra's algorithm.

The returned matrix has an exact unit diagonal and exact fixed elements, and is positive
semi-definite up to ``O(tol)``.
"""
struct AlternatingProjectionsAA{A, K} <: NCMAlgorithm
    tau::Real
    m::Int
    args::A
    kwargs::K
end

function AlternatingProjectionsAA(args...; tau::Real = 0, m::Int = 2, kwargs...)
    return AlternatingProjectionsAA(tau, m, args, kwargs)
end

default_iters(::AlternatingProjectionsAA, A) = clamp(size(A, 1), 20, 200)
modifies_in_place(::AlternatingProjectionsAA) = true
supports_float16(::AlternatingProjectionsAA) = true
supports_symmetric(::AlternatingProjectionsAA) = false
supports_parameterless_construction(::Type{AlternatingProjectionsAA}) = true
supports_mask(::AlternatingProjectionsAA) = true

function autotune(::Type{AlternatingProjectionsAA}, prob::NCMProblem)
    return AlternatingProjectionsAA(; tau = eps(eltype(prob.A)), m = 2)
end

function CommonSolve.solve!(solver::NCMSolver, alg::AlternatingProjectionsAA; kwargs...)
    A = solver.A
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("Input matrix A must be square."))

    T = eltype(A)
    m = alg.m
    tol = solver.reltol
    maxiter = solver.maxiters
    mask = solver.mask
    tau = convert(T, alg.tau)

    n2 = n * n
    dim = 2 * n2

    # Dykstra state z = (Y, S), initialized to z_0 = (A, 0).
    Y = copy(A)
    S = zeros(T, n, n)

    # Working matrices for one evaluation of the map ǧ(z) = (Y', S').
    R = similar(Y)
    X = similar(Y)
    Ymap = similar(Y)
    Smap = similar(Y)
    tmp = similar(Y)   # scratch for the n×n residual halves

    # Pre-allocated Anderson history (only when m > 0). Following Algorithm 2 over the full
    # state z = (Y, S), we keep sliding windows of the state differences Δz_k = z_k − z_{k−1}
    # and residual differences Δf_k = f_k − f_{k−1} (the stable "differences" form of AA).
    if m > 0
        DZ = Matrix{T}(undef, dim, m)   # state-difference history Δz (columns)
        DF = Matrix{T}(undef, dim, m)   # residual-difference history Δf (columns)
        f = Vector{T}(undef, dim)       # current residual f_k
        f_prev = zeros(T, dim)         # previous residual f_{k−1} (set in iter 1 before first use)
        zvec = Vector{T}(undef, dim)    # vec of the current state z_k
        zprev = zeros(T, dim)          # vec of the previous state z_{k−1} (set in iter 1 before first use)
        dz = Vector{T}(undef, dim)      # Δz_k
        df = Vector{T}(undef, dim)      # Δf_k
        delta = Vector{T}(undef, dim)   # accelerated step result z_{k+1}
    end
    m_eff = 0

    rel_err = Inf
    iter = 0

    while iter < maxiter
        iter += 1

        # ǧ(z): one Dykstra step followed by P_E (unit diagonal + fixed elements).
        @. R = Y - S
        X .= mask === nothing ? project_s(R) : Symmetric(project_psd(R, tau))
        @. Smap = X - R
        copyto!(Ymap, X)
        setdiag!(Ymap, one(T))
        if mask !== nothing
            project_f!(Ymap, solver.A_orig, mask)
        end

        # Paper (Algorithm 5, step 5) termination test: ‖Y_k − X_k‖₂ / ‖Y_k‖₂. We measure it on the
        # map image Ymap = P_E(X), which always has a unit diagonal (so ‖Ymap‖₂ ≥ 1 even at iter 0,
        # where the state Y = A need not).
        rel_err = norm(Ymap .- X, 2) / norm(Ymap, 2)
        if solver.verbose
            println("Iter $iter: rel_err = $rel_err")
        end
        if rel_err <= tol
            break   # result is Ymap (the last P_E image)
        end

        if m == 0
            # Plain Picard / Dykstra step.
            Y .= Ymap
            S .= Smap
        else
            # Residual f_k = vec(Y' − Y) ⊕ vec(S' − S) and the current state z_k = (Y, S).
            # vec() gives a lazy 1D view of the contiguous n×n halves (no allocation), so the
            # broadcast matches the flat 1D slices — broadcasting a 2D array into a 1D slice
            # would fail on shape, and copyto! misbehaves across shapes.
            @. tmp = Ymap - Y
            f[1:n2] .= vec(tmp)
            @. tmp = Smap - S
            f[(n2 + 1):dim] .= vec(tmp)
            zvec[1:n2] .= vec(Y)
            zvec[(n2 + 1):dim] .= vec(S)

            if iter > 1
                # Differences from the previously visited state/residual.
                @. dz = zvec - zprev
                @. df = f - f_prev

                # Push (Δz_k, Δf_k) into the sliding window of size m.
                if m_eff < m
                    m_eff += 1
                    DZ[:, m_eff] .= dz
                    DF[:, m_eff] .= df
                else
                    DZ[:, 1:(m - 1)] .= @view DZ[:, 2:m]
                    DF[:, 1:(m - 1)] .= @view DF[:, 2:m]
                    DZ[:, m] .= dz
                    DF[:, m] .= df
                end

                # γ = argmin ‖DF_k γ − f_k‖₂  (Algorithm 2, step 8).
                DZ_view = @view DZ[:, 1:m_eff]
                DF_view = @view DF[:, 1:m_eff]
                gamma = DF_view \ f

                # z_{k+1} = z_k + f_k − DZ_k γ − DF_k γ  (Algorithm 2, step 9). The two
                # products are BLAS matrix-vector multiplies and must be computed eagerly — they
                # cannot sit under `@.`, which would rewrite `*` into element-wise `.*`.
                dzg = DZ_view * gamma
                dfg = DF_view * gamma
                @. delta = zvec + f - dzg - dfg
                Y .= reshape((@view delta[1:n2]), n, n)
                S .= reshape((@view delta[(n2 + 1):dim]), n, n)
            else
                # First step: plain Picard / Dykstra (z_1 = z_0 + f_0 = ǧ(z_0)).
                Y .= Ymap
                S .= Smap
            end

            copyto!(zprev, zvec)
            copyto!(f_prev, f)
        end
    end

    if rel_err > tol && solver.verbose
        println(
            "AlternatingProjectionsAA did not converge within $maxiter iterations " *
                "(rel_err = $rel_err)."
        )
    end

    return build_ncm_solution(alg, Ymap, rel_err, solver; iters = iter)
end
