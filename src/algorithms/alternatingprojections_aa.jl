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

    # Initialize working matrices
    X = copy(A)
    Y = copy(A)
    S = zeros(T, n, n)
    R = similar(A)
    G = similar(A)

    # Pre-allocate memory for Anderson Acceleration history
    vec_dim = n * n
    if m > 0
        DF = Matrix{T}(undef, vec_dim, m)
        DG = Matrix{T}(undef, vec_dim, m)
        f_old = Vector{T}(undef, vec_dim)
        g_old = Vector{T}(undef, vec_dim)
    end
    m_eff = 0

    iter = 0
    converged = false
    rel_err = 0.0

    while iter < maxiter
        iter += 1

        # R = Y - S
        @. R = Y - S

        # X = P_S(R) : Project onto Positive Semidefinite Cone S+
        X .= project_s(R)

        # S = X - R : Update Dykstra correction
        @. S = X - R

        # G = P_U(X) : Project onto Unit Diagonal U
        copyto!(G, X)
        for i in 1:n
            G[i, i] = one(T)
        end

        # Relative residual error check
        rel_err = norm(X .- G, 2) / max(one(T), norm(X, 2))

        if solver.verbose
            println("Iter $iter: rel_err = $rel_err")
        end

        if rel_err <= tol
            converged = true
            break
        end

        # Anderson Acceleration update step
        if m > 0
            # Residual F = G - Y
            f = vec(G .- Y)
            g = vec(G)

            if iter > 1
                df = f .- f_old
                dg = g .- g_old

                # Update memory history buffers without dynamic re-allocations
                if m_eff < m
                    m_eff += 1
                    DF[:, m_eff] .= df
                    DG[:, m_eff] .= dg
                else
                    # Shift left
                    DF[:, 1:m-1] .= @view DF[:, 2:m]
                    DF[:, m] .= df
                    DG[:, 1:m-1] .= @view DG[:, 2:m]
                    DG[:, m] .= dg
                end

                # Solve linear least-squares: min || DF * gamma - f ||_2
                DF_view = @view DF[:, 1:m_eff]
                DG_view = @view DG[:, 1:m_eff]

                gamma = DF_view \ f

                # Compute accelerated iterate g_acc = g - DG * gamma
                g_acc = g .- DG_view * gamma

                # Reshape back to n x n matrix
                Y .= reshape(g_acc, n, n)

                # Enforce symmetry and unit diagonal on Y for numerical stability
                Y .= (Y .+ Y') ./ 2
                for i in 1:n
                    Y[i, i] = one(T)
                end
            else
                Y .= G
            end

            copyto!(f_old, f)
            copyto!(g_old, g)
        else
            Y .= G
        end
    end

    # Ensure output X strictly satisfies unit diagonal and exact symmetry
    for i in 1:n
        X[i, i] = one(T)
    end
    X .= (X .+ X') ./ 2

    return build_ncm_solution(alg, X, rel_err, solver; iters = iter)
end
