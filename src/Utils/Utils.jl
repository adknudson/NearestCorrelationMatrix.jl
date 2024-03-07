module Utils

using LinearAlgebra


export
    issquare,
    require_square,
    diagonals_are_one,
    constrained_to_pm_one,
    isprecorrelation,
    iscorrelation,
    clampcor,
    clampcor!,
    setdiag!,
    symmetric!,
    corconstrain!,
    cov2cor,
    cov2cor!,
    cor2cov,
    cor2cov!,
    checkmat!,
    eigen_safe,
    project_psd,
    project_psd!



function issquare(X::AbstractMatrix)
    m, n = size(X)
    return m == n
end

function require_square(X::AbstractMatrix)
    issquare(X) || throw_square()
end

@noinline throw_square() = throw(DimensionMismatch("Matrix required to be square"))


function diagonals_are_one(X::AbstractMatrix{T}) where {T<:Real}
    return all(==(one(T)), diag(X))
end


function constrained_to_pm_one(X::AbstractMatrix{T}) where {T<:Real}
    return all(x -> -one(T) ≤ x ≤ one(T), X)
end

"""
    isprecorrelation(X)

Test that a matrix passes all the pre-qualifications to be a correlation matrix.

A pre-correlation matrix must:

- be square
- be symmetric
- be constrained to ±1
- have diagonals equal to 1
"""
function isprecorrelation(X::AbstractMatrix{T}) where {T}
    issquare(X)              || return false
    issymmetric(X)           || return false
    diagonals_are_one(X)     || return false
    constrained_to_pm_one(X) || return false
    return true
end

"""
    iscorrelation(X)

Test that a matrix passes all the qualifications to be a correlation matrix including being
positive (semi) definite.

A correlation matrix must:

- be square
- be symmetric
- be constrained to ±1
- have diagonals equal to 1
- be positive definite
"""
function iscorrelation(X::AbstractMatrix{T}) where {T<:Real}
    return isprecorrelation(X) && isposdef(X)
end



clampcor(x::Real) = clamp(x, -one(x), one(x))

function clampcor!(X::AbstractArray{T,N}) where {T<:Real, N}
    @inbounds for i in eachindex(X)
        X[i] = clampcor(X[i])
    end
    return X
end



"""
    setdiag!(X, v)

Set the diagonal elements of `X` to `v`.
"""
function setdiag!(X::AbstractMatrix{T}, v::T) where {T}
    require_square(X)

    for i in diagind(X)
        X[i] = v
    end

    return X
end


function char_uplo(uplo::Symbol)
    if uplo === :U
        return 'U'
    elseif uplo === :L
        return 'L'
    else
        throw_uplo()
    end
end

function sym_uplo(uplo::Char)
    if uplo == 'U'
        return :U
    elseif uplo == 'L'
        return :L
    else
        throw_uplo()
    end
end

@noinline throw_uplo() = throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))


function symmetric!(X::AbstractMatrix, uplo::Symbol=:U)
    if uplo === :U
        _copytolower!(X)
    elseif uplo === :L
        _copytoupper!(X)
    else
        throw_uplo()
    end

    return X
end

symmetric!(X::Symmetric, ::Symbol=:U) = X
symmetric!(X::Diagonal,  ::Symbol=:U) = X



function _copytolower!(X::AbstractMatrix)
    require_square(X)

    nr, nc = size(X)
    for j in 1:nc-1
        for i in j+1:nr
            @inbounds X[i,j] = X[j,i]
        end
    end
    return X
end

function _copytoupper!(X::AbstractMatrix)
    require_square(X)

    nr, nc = size(X)
    for j in 1:nc-1
        for i in j+1:nr
            @inbounds X[j,i] = X[i,j]
        end
    end
    return X
end



function corconstrain!(X::AbstractMatrix{T}, uplo::Symbol=:U) where {T<:Real}
    clampcor!(X)
    setdiag!(X, one(T))
    symmetric!(X, uplo)
    return X
end

function corconstrain!(X::Symmetric{T}, ::Symbol=:U) where {T<:Real}
    clampcor!(X.data)
    setdiag!(X, one(T))
    symmetric!(X.data, sym_uplo(X.uplo))
    return X
end

function corconstrain!(X::Diagonal{T}, ::Symbol=:U) where {T<:Real}
    fill!(X.diag, one(T))
    return X
end



function cov2cor(X::AbstractMatrix{T}) where {T<:Real}
    D = sqrt(inv(Diagonal(X)))
    return corconstrain!(D * X * D)
end

function cov2cor!(X::AbstractMatrix{T}) where {T<:Real}
    D = sqrt(inv(Diagonal(X)))
    lmul!(D, X)
    rmul!(X, D)
    setdiag!(X, one(T))
    symmetric!(X)
    return X
end

function cov2cor!(X::Symmetric{T}) where {T<:Real}
    symmetric!(X.data, sym_uplo(X.uplo))
    cov2cor!(X.data)
    return X
end

cor2cov(C::AbstractMatrix{T}, s::AbstractVector{T}) where {T<:Real} = cor2cov!(copy(C), s)

function cor2cov!(C::AbstractMatrix{T}, s::AbstractVector{T}) where {T<:Real}
    for i in CartesianIndices(size(C))
        @inbounds C[i] *= s[i[1]] * s[i[2]]
    end
    return C
end

function cor2cov!(C::Symmetric{T}, s::AbstractVector{T}) where {T<:Real}
    symmetric!(C.data)
    cor2cov!(C.data, s)
    return C
end


"""
    checkmat!(X)

Check the properties of the input matrix and prepare it for the nearest correlation algorithm.
This checks that the matrix is square (required) and symmetric (optional). If the matrix is
not symmetric, then it is replaced with `(X + X') / 2`.
"""
function checkmat!(X::AbstractMatrix{T}; warn::Bool=false) where {T<:Real}
    require_square(X)

    if !issymmetric(X)
        warn && @warn "The input matrix is not symmetric. Replacing with (X + X') / 2"
        X .= (X + X') / 2
    end

    return X
end




"""
    eigen_safe(X)

Compute the eigen docomposition of `X` with eigenvalues in descending order. This method is
also type stable in the sense that if the eltype of the input is `T`, then the eltype of the
output is also `T`.
"""
function eigen_safe(X::AbstractMatrix{T}) where {T<:Real}
    E = eigen(X, sortby=x->-x)

    TE = eltype(E)
    TE != T && error("Eigen eltype does not match the input eltype")
    TE <: Complex && error("Eigen decomposition resulted in complex values")

    return E
end

# symmetric matrices are guaranteed to have real eigenvalues, so no checks required.
function eigen_safe(X::Symmetric{T}) where {T<:Real}
    return eigen(X, sortby=x->-x)
end

#=
Unlike eigen(::Matrix{Float16}) which returns Float16, the returned eltype of
eigen(::Symmetric{Float16}) is Float32. We do this extra conversion for less surprising
eltype results.
=#
function eigen_safe(X::Symmetric{Float16})
    E = eigen(X, sortby=x->-x)
    values = convert(AbstractVector{Float16}, E.values)
    vectors = convert(AbstractMatrix{Float16}, E.vectors)
    return Eigen(values, vectors)
end



"""
    project_psd!(X, λ, P)

Project `X` onto the cone of positive semi-definite matrices. This will modify `X` in place.

- `X` is the input matrix
- `λ` is a vector of the eigenvalues of `X` sorted in descending order
- `P` are the corresponding eigenvectors to `λ`
"""
function project_psd!(X::AbstractMatrix{T}, λ::AbstractVector{T}, P::AbstractMatrix{T}) where {T}
	n = length(λ)
	r = count(>(0), λ)

	if r == n
		nothing
    elseif r == 0
		fill!(X, zero(T))
	elseif r == 1
		P1 = @view P[:,1]
        λ1 = λ[1]
		mul!(X, P1, P1', λ1, 0)
	elseif 2r ≤ n
		Pr = @view P[:, begin:r]
		λr = sqrt(Diagonal(λ[begin:r]))
		Q = Pr * λr
        mul!(X, Q, Q')
	else
		Ps = @view P[:, r+1:end]
		λs = sqrt(Diagonal(-λ[r+1:end]))
		Q = Ps * λs
        mul!(X, Q, Q', 1, 1)
	end

	return X
end

"""
    project_psd!(X)

Project `X` onto the cone of positive semi-definite matrices. This will modify `X` in place.
"""
function project_psd!(X::AbstractMatrix{T}) where {T<:Real}
    λ, P = eigen_safe(X)
    return project_psd!(X, λ, P)
end

function project_psd!(X::Symmetric{T}) where {T<:Real}
    λ, P = eigen_safe(X)
    symmetric!(X.data, sym_uplo(X.uplo))
    project_psd!(X.data, λ, P)
    return X
end

"""
    project_psd(X)

The projection of `X` onto the cone of positive semi-definite matrices.
"""
project_psd(X) = project_psd!(copy(X))

"""
    project_psd(X, λ, P)

The projection of `X` onto the cone of positive semi-definite matrices.

- `X` is the input matrix
- `λ` is a vector of the eigenvalues of `X` sorted in descending order
- `P` are the corresponding eigenvectors to `λ`
"""
function project_psd(X::AbstractMatrix{T}, λ::AbstractVector{T}, P::AbstractMatrix{T}) where {T<:Real}
    return project_psd!(copy(X), λ, P)
end

end
