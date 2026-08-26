using LinearAlgebra

"""
    get_negdef_matrix(Type)

Gets a 4×4 invalid correlation matrix for testing.
"""
function get_negdef_matrix(::Type{T}) where {T}
    r = [
        1.0 -0.2188 -0.79 0.7773
        -0.2188 1.0 0.2559 -0.5977
        -0.79 0.2559 1.0 0.2266
        0.7773 -0.5977 0.2266 1.0
    ]

    return convert(AbstractMatrix{T}, r)
end

"""
    rand_negdef(T, n)

Generate a random negative definite matrix of size `n × n` with eltype ``T``.
"""
function rand_negdef(::Type{T}, n) where {T}
    while true
        r = 2 * rand(T, n, n) .- one(T)
        symmetric!(r)
        r[diagind(r)] .= one(T)

        !isposdef(r) && return r
    end
    return
end
