struct Rescaling <: NCMAlgorithm
    δ::Real
end

Rescaling() = Rescaling(sqrt(eps()))

modifies_in_place(::Rescaling) = false
supports_symmetric(::Rescaling) = true
supports_float16(::Rescaling) = true
supports_parameterless_construction(::Type{Rescaling}) = true

function CommonSolve.solve!(solver::NCMSolver, alg::Rescaling; kwargs...)
    R0 = solver.A
    d = size(R0, 1)
    T = float(eltype(R0))
    R = Matrix{T}(R0)

    if any(x -> !isfinite(x), R)
        X = Matrix{T}(LinearAlgebra.I, d, d)
        return build_ncm_solution(alg, X, Inf, solver)
    end

    R = Matrix(LinearAlgebra.Symmetric((R + R') / 2))
    @inbounds for j in 1:d
        R[j, j] = one(T)
    end

    δ = convert(T, alg.δ)
    λmin = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(R))
    if λmin > δ
        return build_ncm_solution(alg, R, 0, solver)
    end

    # The eigenvalues of (1 - λ) R + λ I are (1 - λ) λᵢ + λ, so this λ is
    # the smallest shrinkage that puts the smallest one at δ.
    λ = (δ - λmin) / (one(T) - λmin)
    R = (one(T) - λ) .* R .+ λ .* Matrix{T}(LinearAlgebra.I, d, d)
    @inbounds for j in 1:d
        R[j, j] = one(T)
    end

    return build_ncm_solution(alg, R, δ, solver; iters = 1)
end
