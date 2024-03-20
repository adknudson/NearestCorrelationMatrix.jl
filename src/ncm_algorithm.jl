"""
    NCMAlgorithm

The abstract type of all NCM problem solving algorithms.
"""
abstract type NCMAlgorithm end


"""
    alg_name(alg)

Get the simple name for the NCM algorithm type.
"""
alg_name(::Type{T}) where {T<:NCMAlgorithm} = (isempty(T.parameters) ? T : T.name.wrapper)
alg_name(alg::NCMAlgorithm) = alg_name(typeof(alg))


"""
    autotune(algtype, prob)

Initialize an algorithm that is tuned to the NCM problem.

# Examples

```julia-repl
julia> r = Float32[
     1.0     -0.2188  -0.79     0.7773
    -0.2188   1.0      0.2559  -0.5977
    -0.79     0.2559   1.0      0.2266
     0.7773  -0.5977   0.2266   1.0
];

julia> prob = NCMProblem(r);

julia> alg = autotune(Newton, prob);
```
"""
autotune(algtype::Type{<:NCMAlgorithm}, ::NCMProblem) = construct_algorithm(algtype)


"""
    init_cacheval(alg, args...)
"""
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
default_iters(::NCMAlgorithm, A::Any) = size(A, 1)


"""
    default_alias_A(alg, A)

Whether to alias the matrix `A` or use a copy by default. When `true`, algorithms that update
in place can be faster by reusing the memory, but care must be taken as the original input
will be modified. Default is `true` if the algorithm is known not to modify `A`, otherwise
is `false`.
"""
default_alias_A(alg::NCMAlgorithm, ::Any) = !modifies_in_place(alg)


"""
    modifies_in_place(alg)

Trait for if an algorithm modifies the input in place or not. `true` by default.
"""
modifies_in_place(::NCMAlgorithm) = true


"""
    supports_float16(alg)

Trait for if an algorithm supports matrices with Float16 values. There are often numerical
instabilities with Float16 values, so the default is `false`.
"""
supports_float16(::NCMAlgorithm) = false


"""
    supports_symmetric(alg)

Trait for if an algorithm supports `LinearAlgebra.Symmetric` matrix type (default is `false`).
If `false`, then a copy using the upper or lower matrix is used instead.
"""
supports_symmetric(::NCMAlgorithm) = false


"""
    supports_parameterless_construction(alg)

Trait for if an algorithm can be constructed without any parameters (default is `false`).
"""
supports_parameterless_construction(::Type{NCMAlgorithm}) = false


"""
    construct_algorithm(algtype)

Construct the algorithm without ant parameters. Throws an error if the algtype does not
support parameterless construction.
"""
function construct_algorithm(algtype::Type{NCMAlgorithm})
    if supports_parameterless_construction(algtype)
        return algtype()
    end

    throw(MethodError(algtype, "$algtype does not support parameterless construction."))
end
