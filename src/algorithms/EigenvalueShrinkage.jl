"""
    EigenvalueShrinkage(δ = sqrt(eps()))

Fast, one-step heuristic that repairs a symmetric matrix into a positive-definite
correlation matrix by shrinking its eigenvalues towards the identity.

The input is symmetrized and given a unit diagonal, then the smallest shrinkage
intensity `λ` is found so that the convex combination

    (1 - λ) R + λI

has smallest eigenvalue at least `δ`. Because the eigenvalues of `(1 - λ) R + λ I`
are `(1 - λ) λᵢ + λ`, this intensity is

    λ = (δ - λmin) / (1 - λmin)

where `λmin` is the smallest eigenvalue of the (unit-diagonal, symmetrized) input.
Every eigenvalue is mapped affinely to `(1 - λ) λᵢ + λ`: eigenvalues below 1 are
lifted, eigenvalues above 1 are pushed down, and the smallest eigenvalue `λmin` is
lifted exactly to `δ`. No eigenvalue is left unchanged unless it equals 1. The unit
diagonal is preserved exactly (a convex combination of `R` and `I`), and the result
is strictly positive definite.

# Fields
- `δ`: minimum eigenvalue enforced on the output. Defaults to `sqrt(eps())`, a value
  just above machine precision, so the result is positive definite and well conditioned.

# Notes
- Shrinkage heuristic, not the true nearest correlation matrix: the eigenvectors of the
  input are kept fixed, so the Frobenius distance to the input is not minimized. Use
  [`Newton`](@ref) or [`AlternatingProjections`](@ref) for an optimal solution.
- Useful when a valid, well-conditioned correlation matrix is needed quickly without
  concern for optimality - e.g. to seed an optimizer or to repair pairwise estimates in
  high dimensions.
"""
struct EigenvalueShrinkage <: NCMAlgorithm
    δ::Real
    EigenvalueShrinkage(δ::Real) = new(float(δ))
end

EigenvalueShrinkage() = EigenvalueShrinkage(sqrt(eps()))

modifies_in_place(::EigenvalueShrinkage) = false
supports_symmetric(::EigenvalueShrinkage) = true
supports_float16(::EigenvalueShrinkage) = true
supports_parameterless_construction(::Type{EigenvalueShrinkage}) = true

function CommonSolve.solve!(solver::NCMSolver, alg::EigenvalueShrinkage; kwargs...)
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
