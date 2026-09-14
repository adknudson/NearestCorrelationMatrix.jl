"""
    nearest_cor!(A, alg=nothing; kwargs...)

Return the nearest positive definite correlation matrix to ``A``.
This method will overwrite ``A``.

This is a "batteries included" method, and is designed to just work with as little thought
as possible. Unlike `solve!`, this method will return the nearest correlation matrix
directly instead of a `NCMSolution` object. Additionally the solution is checked to be
positive definite, and corrected if it is not.

When a fixed-element mask is passed (via the `mask` keyword or on the problem), the result has
an exact unit diagonal and retains the masked elements exactly. Because repairing positive
definiteness perturbs every entry, strict PD and exact fixed-element feasibility cannot both be
guaranteed: the fixed elements take precedence, and the result is positive definite up to `√ϵ`.

# Examples

```julia-repl
julia> import LinearAlgebra: isposdef

julia> r = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
];

julia> isposdef(r)
false

julia> nearest_cor!(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(r)
true
```
"""
function nearest_cor!(A, alg; kwargs...)
    # (#41) Pass any potential mask to the problem so that the proper algorithm can be selected
    prob = NCMProblem(A; kwargs...)

    solver = init(
        prob,
        alg;
        alias_A = true,
        fix_sym = true,
        convert_f16 = true,
        ensure_pd = true,
        kwargs...
    )

    sol = solve!(solver)

    copyto!(A, sol.X)
    return A
end

nearest_cor!(A; kwargs...) = nearest_cor!(A, nothing; kwargs...)

"""
    nearest_cor(A, alg; kwargs...)

Return the nearest positive definite correlation matrix to ``A``.

This is a "batteries included" method, and is designed to just work with as little thought
as possible. Unlike `solve!`, this method will return the nearest correlation matrix
directly instead of a `NCMSolution` object. Additionally the solution is checked to be
positive definite, and corrected if it is not.

When a fixed-element mask is passed (via the `mask` keyword or on the problem), the result has
an exact unit diagonal and retains the masked elements exactly. Because repairing positive
definiteness perturbs every entry, strict PD and exact fixed-element feasibility cannot both be
guaranteed: the fixed elements take precedence, and the result is positive definite up to `√ϵ`.

# Examples

```julia-repl
julia> import LinearAlgebra: isposdef

julia> r = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
];

julia> isposdef(r)
false

julia> p = nearest_cor(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(p)
true
```
"""
function nearest_cor(A, alg; kwargs...)
    # (#41) Pass any potential mask to the problem so that the proper algorithm can be selected
    prob = NCMProblem(A; kwargs...)

    # `nearest_cor` cannot simply call `nearest_cor!` with `alias_A=false`. For some reason
    # the `alias_A` keyword is passed properly, but `A` still ends up getting aliased anyway.
    # The solution is to call `init` with `alias_A=false` independently.

    solver = init(
        prob,
        alg;
        alias_A = false,
        fix_sym = true,
        convert_f16 = true,
        ensure_pd = true,
        kwargs...
    )

    sol = solve!(solver)

    return sol.X
end

nearest_cor(A; kwargs...) = nearest_cor(A, nothing; kwargs...)
