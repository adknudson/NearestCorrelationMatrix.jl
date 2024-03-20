using LinearAlgebra

export
    get_negdef_matrix,
    rand_negdef,
    clampcor,
    clampcor!,
    setdiag!,
    sym_uplo,
    char_uplo,
    symmetric!,
    corconstrain!,
    cov2cor,
    cov2cor!,
    cor2cov,
    cor2cov!,
    eigen_sym,
    project_psd!,
    project_psd



"""
    get_negdef_matrix(Type)

Get a negative definite matrix for testing.
"""
function get_negdef_matrix(::Type{T}) where {T<:AbstractFloat}
    r = [
        1.0     -0.2188  -0.79     0.7773
       -0.2188   1.0      0.2559  -0.5977
       -0.79     0.2559   1.0      0.2266
        0.7773  -0.5977   0.2266   1.0
    ]

    return convert(AbstractMatrix{T}, r)
end


function rand_negdef(::Type{T}, n) where{T<:AbstractFloat}
    while true
        r = rand(T, n, n)
        symmetric!(r)
        r[diagind(r)] .= one(T)

        !isposdef(r) && return r
    end
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



eigen_sym(X::Symmetric{T}) where {T<:Real} = eigen(X, sortby=x->-x)

function eigen_sym(X::Symmetric{Float16})
    E = eigen(X, sortby=x->-x)
    values = convert(AbstractVector{Float16}, E.values)
    vectors = convert(AbstractMatrix{Float16}, E.vectors)
    return Eigen(values, vectors)
end

eigen_sym(X::AbstractMatrix{T}) where {T<:Real} = eigen_sym(Symmetric(X))



function project_psd!(X::AbstractMatrix{T}, ϵ::T=zero(T)) where T
    ϵ = max(ϵ, zero(T))
    λ, P = eigen_sym(X)
    replace!(x -> max(x, ϵ), λ)
    X .= P * Diagonal(λ) * P'
    return X
end

function project_psd!(X::Symmetric{T}, ϵ::T=zero(T)) where T
    ϵ = max(ϵ, zero(T))
    λ, P = eigen_sym(X)
    replace!(x -> max(x, ϵ), λ)
    X.data .= P * Diagonal(λ) * P'
    return X
end

function project_psd(X::AbstractMatrix{T}, ϵ::T=zero(T)) where T
    return project_psd!(copy(X), ϵ)
end
