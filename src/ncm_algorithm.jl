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
    modifies_in_place(alg)

Trait for if an algorithm modifies the input in place or not. `true` by default.
"""
modifies_in_place(::Any         ) = true
modifies_in_place(::NCMAlgorithm) = true



"""
    default_alias_A(alg, A)

Whether to alias the matrix `A` or use a copy by default. When `true`, algorithms that update
in place can be faster by reusing the memory, but care must be taken as the original input
will be modified. Default is `true` if the algorithm is known not to modify `A`, otherwise
is `false`.
"""
default_alias_A(alg::Any,          ::Any) = !modifies_in_place(alg)
default_alias_A(alg::NCMAlgorithm, ::Any) = !modifies_in_place(alg)


"""
    default_alg(A)

Get the default algorithm for a given input matrix.
"""
default_alg(::Any) = AlternatingProjections()
