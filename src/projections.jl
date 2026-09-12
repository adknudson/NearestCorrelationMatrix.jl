"""
    project_unit!(X)

Projects the matrix `X` onto the set of symmetric matrices with unit diagonal.

## Arguments

- `X`: The matrix to perform the projection on.
"""
function project_unit!(X)
    v = one(eltype(X))
    Xs = parent(X)
    for i in diagind(Xs)
        Xs[i] = v
    end
    return X
end

"""
    project_psd!(X, A, δ, Z)

Projects the matrix `A` onto the set of symmetric positive [semi]definite matrices and
stores the result in `X`. The minimum eigenvalue is constrained to `ϵ`. The matrix `Z` is a
scratch space for computing matrix multiplication.

## Details

Let the symmetric matrix `A ∈ ℝⁿˣⁿ` have the spectral decomposition `A = Q diag(λᵢ) Qᵀ` and
let `δ ≥ 0`. Then the unique matrix nearest to `A` with the smallest eigenvalue at least `δ`
is given by

``Q diag(τᵢ) Qᵀ, τᵢ = max(λᵢ, δ)``

## Arguments

- `X`: The matrix to store the projection in.
- `A`: The matrix to perform the projection on.
- `δ`: The minimum eigenvalue allowed in the spectral decomposition.
- `Z`: A scratch space for the matrix multiplication.
"""
function project_psd!(X, A, δ, Z)
    λ, P = eigen_sym(A)
    for i in eachindex(λ)
        λ[i] = max(λ[i], δ)
    end
    Λ = Diagonal(λ)
    Xs = parent(X)
    mul!(Z, P, Λ)   # P * Λ  -> Z
    mul!(Xs, Z, P') # Z * P' -> X
    return X
end

"""
    project_fixed!(X, A, mask)

Projects the elements of `A` onto `X` for the elements where `mask` is `true`.

## Arguments

- `X`: The matrix to store the projection in.
- `A`: The matrix of elements to preserve.
- `mask`: A `BitMatrix` inddicating which elements should remain fixed.
"""
function project_fixed!(X, A, mask)
    Xs = parent(X)
    Xs[mask] .= A[mask]
    return X
end
