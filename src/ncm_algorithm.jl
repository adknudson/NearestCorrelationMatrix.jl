"""
    NCMAlgorithm
"""
abstract type NCMAlgorithm end


init_cacheval(::NCMAlgorithm, args...) = nothing


"""
    default_tol(::Type)

Get the default tolerance for a given type.
"""
default_tol(::Type{Any}) = 0
default_tol(::Type{T}) where {T} = sqrt(eps(T))
default_tol(::Type{Complex{T}}) where {T} = sqrt(eps(T))
default_tol(::Type{<:Rational}) = 0
default_tol(::Type{<:Integer}) = 0



"""
    default_iters(alg, A)
"""
default_iters(::NCMAlgorithm, ::Any) = 0


"""
    default_alias_A(alg, A)

Whether to alias the matrix `A` or use a copy by default. When `true`, algorithms that update
in place can be faster by reusing the memory, but care must be taken as the original input
will be modified. Default is `true` if the algorithm is known not to modify `A`, otherwise
is `false`.
"""
default_alias_A(::Any,          ::Any) = false
default_alias_A(::NCMAlgorithm, ::Any) = false


"""
    default_alg(A)

Get the default algorithm for a given input matrix.
"""
default_alg(::Any) = AlternatingProjections()
