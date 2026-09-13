"""
    AcceleratedAP(; tau=0, m=2)

The alternating projections algorithm with Anderson acceleration applied. Should converge
in roughly half the number of steps of the standard alternating projections algorithm.
"""
struct AcceleratedAP{A, K} <: NCMAlgorithm
    tau::Real
    m::Int
    args::A
    kwargs::K
end

function AcceleratedAP(args...; tau::Real = 0, m::Int = 2, kwargs...)
    return AcceleratedAP(tau, m, args, kwargs)
end

default_iters(::AcceleratedAP, A) = clamp(size(A, 1), 20, 200)
modifies_in_place(::AcceleratedAP) = true
supports_float16(::AcceleratedAP) = true
supports_symmetric(::AcceleratedAP) = false
supports_parameterless_construction(::Type{<:AcceleratedAP}) = true
supports_mask(::Type{<:AcceleratedAP}) = true

function autotune(::Type{<:AcceleratedAP}, prob::NCMProblem)
    return AcceleratedAP(; tau = sqrt(eps(eltype(prob.A))), m = 2)
end

function CommonSolve.solve!(solver::NCMSolver, alg::AcceleratedAP; kwargs...)
    A = solver.A
    n = size(A, 1)

    T = eltype(A)
    m = alg.m
    tau = convert(T, alg.tau)
    mask = solver.mask
    A_orig = solver.A_orig

    # Initialize working matrices
    X = copy(A)
    Y = copy(A)
    S = zeros(T, n, n)
    R = similar(A)
    G = similar(A)
    scratch = similar(A)

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
    resid = Inf

    while iter < solver.maxiters
        iter += 1

        R .= Y .- S
        project_psd!(X, R, tau, scratch)
        S .= X .- R
        copyto!(G, X)

        if mask !== nothing
            project_fixed!(G, A_orig, mask)
        end

        # project unit after projecting fixed to ensure that the unit diagonal is preserved
        project_unit!(G)

        resid = norm(X .- G) / norm(X)

        if resid <= solver.reltol
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
                    DF[:, 1:(m - 1)] .= @view DF[:, 2:m]
                    DF[:, m] .= df
                    DG[:, 1:(m - 1)] .= @view DG[:, 2:m]
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

    if mask !== nothing
        project_fixed!(X, A_orig, mask)
    end
    project_unit!(X)

    return build_ncm_solution(alg, X, resid, solver; iters = iter)
end
