export default_negdef, rand_negdef

"""
    default_negdef([T=Float64])

Returns a 4×4 invalid correlation matrix and an optional mask.
"""
function default_negdef(::Type{T}) where {T <: AbstractFloat}
    # !WARNING! The data in this generator must never be edited to ensure consistency with
    # future releases.

    A = Float64[
        1.0 -0.2188 -0.79 0.7773
        -0.2188 1.0 0.2559 -0.5977
        -0.79 0.2559 1.0 0.2266
        0.7773 -0.5977 0.2266 1.0
    ]

    mask = [
        1 1 0 0
        1 1 0 0
        0 0 1 1
        0 0 1 1
    ]

    X = convert(AbstractMatrix{T}, A)
    m = convert(BitMatrix, mask)

    return X, m
end

default_negdef() = default_negdef(Float64)

"""
    rand_negdef([T=Float64], n; max_attempts=100)

Generates a random pseudo-correlation matrix that is negative definite.
"""
function rand_negdef(::Type{T}, n::Int; max_attempts::Int = 100) where {T <: AbstractFloat}
    attempts = 0
    while attempts < max_attempts
        r = 2 * rand(T, n, n) .- one(T)
        symmetrize!(r)
        r[diagind(r)] .= one(T)
        !isposdef(r) && return r
        attempts += 1
    end
    return Matrix{T}(undef, 0, 0)
end

rand_negdef(n::Int; max_attempts::Int = 100) = rand_negdef(Float64, n; max_attempts)
