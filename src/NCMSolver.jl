"""
    NCMSolver

Common interface for solving NCM problems. Algorithm-specific cache is stored in the
`cacheval` field.

- `A`: The input matrix. Must be square. Should be symmetric.
- `p`: The parameters for the problem. Defaults to `NullParameters`. Currently unused.
- `alg`: The algorithm used by the solver.
- `cacheval`: Algorithm-specific cache.
- `isfresh`: `true` if the cacheval hasn't been set yet.
- `abstol`: The absolute tolerance. Defaults to `√(eps(eltype(A)))`.
- `reltol`: The relative tolerance. Defaults to `√(eps(eltype(A)))`.
- `maxiters`: The number of iterations allowed. Defaults to `size(A,1)`
- `ensure_pd`: Checks (and corrects) that the resulting matrix is positive definite.
  Defaults to `false`.
- `min_eigenvalue`: The minimum eigenvalue to enforce when `ensure_pd` == true.
- `verbose`: Whether to print extra information. Defaults to `false`.
- `mask`: The fixed-element mask, or `nothing` if unmasked.
- `A_orig`: The original values of A to be used when a mask is supplied.
"""
mutable struct NCMSolver{TA, P, Talg, Tc, Ttol, Tm}
    A::TA           # the input matrix
    p::P            # parameters
    alg::Talg       # ncm algorithm
    cacheval::Tc    # store algorithm cache here
    isfresh::Bool   # false => cacheval is set wrt A, true => update cacheval wrt A
    abstol::Ttol    # absolute tolerance for convergence
    reltol::Ttol    # relative tolerance for convergence
    maxiters::Int   # maximum number of iterations
    ensure_pd::Bool # ensures that the resulting matrix is positive definite
    min_eigenvalue::Union{Nothing, Real} # the minimum eigenvalue to enforce
    verbose::Bool   # whether to print extra information
    mask::Tm        # fixed-element mask, or nothing
    A_orig::TA      # a copy of A, or an alias of A if no mask is given
end

"""
    init(prob, alg, args...; kwargs...)

Initialize the solver with the given algorithm.

## Keyword Arguments

- `mask`: An optional fixed-element mask. For every upper-triangular position ``(i, j)`` with
  ``i < j`` where ``mask[i, j]`` is truthy, the solution must retain the value ``A[i, j]``. The
  mask is normalized and symmetrized. Requires an algorithm that supports `supports_mask`.
  Defaults to `nothing` (no elements held fixed).
- `alias_A`: Whether to alias the matrix ``A`` or use a copy by default. When `true`,
  algorithms that operate in place can save memory by reusing ``A``. Defaults to `true` if
  the algorithm is known not to modify ``A``, and `false` otherwise.
- `abstol`: The absolute tolerance. Defaults to `√(eps(eltype(A)))`.
- `reltol`: The relative tolerance. Defaults to `√(eps(eltype(A)))`.
- `maxiters`: The number of iterations allowed. Defaults to `size(A,1)`
- `fix_sym`: If `true`, then makes the input matrix symmetric if it is not already. Defaults
   to `false`, and init will fail if the input matrix is not symmetric.
- `uplo`: If `fix_sym` is `true`, then the upper (`:U`) or lower (`:L`) triangle of the
  input is used to make a symmetric matrix. Defaults to `:U`.
- `convert_f16`: If the algorithm does not support `Float16` values, then the input matrix
  will be converted to an `AbstractMatrix{Float32}`. Defaults to `false`, and init will
  fail if the algorithm does not support `Float16`.
- `force_f16`: If `true`, then the algorithm will be forced to use the input matrix, even
  if the algorithm doesn't fully support `Float16` values in a stable way.
- `ensure_pd`: Checks (and corrects) that the resulting matrix is positive definite.
  Defaults to `false`.
- `min_eigenvalue`: The minimum eigenvalue to enforce when `ensure_pd` is `true`. Defaults to
  `nothing`, in which case it is either unused or set to a reasonable value depending on the
  problem parameters.
- `verbose`: Whether to print extra information. Defaults to `false`.
"""
function CommonSolve.init(
        prob::NCMProblem,
        alg::NCMAlgorithm,
        args...;
        mask = nothing,
        alias_A = default_alias_A(alg, prob.A),
        abstol = default_tol(real(eltype(prob.A))),
        reltol = default_tol(real(eltype(prob.A))),
        maxiters::Int = default_iters(alg, prob.A),
        fix_sym::Bool = false,
        uplo::Symbol = :U,
        convert_f16::Bool = false,
        force_f16::Bool = false,
        ensure_pd::Bool = false,
        min_eigenvalue = nothing,
        verbose::Bool = false,
        kwargs...
    )
    # Resolve the effective mask first: an explicit `mask=` kwarg overrides any mask set on the
    # problem; otherwise fall back to the problem's mask, then finally default to nothing.
    mask = if mask !== nothing
        verbose && println("Using fixed-element mask")
        mask
    elseif prob.mask !== nothing
        verbose && println("Using fixed-element mask from the problem")
        prob.mask
    else
        nothing
    end

    # Ensure that the mask is normalized to a BitMatrix or Nothing
    mask = normalize_mask(mask)

    # A non-empty effective mask must be enforced by an algorithm that supports it. Check the
    # *effective* mask (not just the kwarg) so a mask set on the problem is not silently ignored
    # by an unsupported algorithm.
    if mask !== nothing && !supports_mask(alg)
        error(
            "$(alg_name(alg)) does not support a fixed-element mask. Use an algorithm that " *
                "implements `supports_mask` (e.g. AlternatingProjections), or clear the mask " *
                "by passing `mask=nothing`."
        )
    end

    A = prob.A
    p = prob.p

    T = eltype(A)

    A = if alias_A
        verbose && println("Aliasing A")
        A
    elseif A isa Symmetric
        if supports_symmetric(alg)
            verbose && println("Creating a Symmetric copy of A")
            copy(A)
        else
            verbose && println(
                "$(alg_name(alg)) does not support Symmetric types. " *
                    "Creating a symmetric copy of A.data"
            )
            Matrix(A)
        end
    elseif A isa Matrix
        verbose && println("Creating a copy of A")
        copy(A)
    else
        verbose && println("Creating a deep copy of A")
        deepcopy(A)
    end

    if !issymmetric(A)
        if fix_sym
            if supports_symmetric(alg)
                verbose &&
                    println(
                    "Input matrix is not symmetric. Creating a Symmetric view " *
                        "of the $(uplo == :U ? "upper" : "lower") part of the matrix"
                )
                A = Symmetric(A, uplo)
            else
                verbose && println(
                    "Input matrix is not symmetric. Copying the " *
                        "$(uplo == :U ? "upper" : "lower") part of the matrix"
                )
                symmetrize!(A, uplo)
            end
        else
            error(
                "Input matrix is not symmetric. Pass the argument `fix_sym=true`, or ensure " *
                    "that your input matrix is symmetric before solving."
            )
        end
    end

    if T === Float16 && !supports_float16(alg)
        if convert_f16
            verbose &&
                println(
                "Input matrix has eltype Float16, which $(alg_name(alg)) does " *
                    "not support. Converting to `AbstractMatrix{Float32}`"
            )
            A = convert(AbstractMatrix{Float32}, A)
        elseif force_f16
            verbose &&
                println(
                "Input matrix has eltype Float16, which $(alg_name(alg)) does " *
                    "not support. `force_f16=true` so using input matrix anyway."
            )
        else
            error(
                "Input matrix has eltype Float16, which $(alg_name(alg)) does not support. " *
                    "Pass either the argument `convert_f16=true` or `force_f16=true`, or convert " *
                    "your input matrix to an `AbstractMatrix{Float32}` before solving."
            )
        end
    end

    # Capture the original (transformed) input as the value source for fixed elements. With
    # alias_A=true the algorithms overwrite A in place, so the values must be saved here at
    # init time. Only allocated for masked problems.
    A_orig = mask === nothing ? A : copy(A)

    # Guard against type mismatch for user-specified reltol/abstol
    reltol = real(T)(reltol)
    reltol = max(reltol, sqrt(eps(T)))
    abstol = real(T)(abstol)
    abstol = max(abstol, eps(T))

    min_eigenvalue = if min_eigenvalue === nothing
        if ensure_pd
            if mask === nothing
                # no mask, can default to sqrt(eps(T))
                sqrt(eps(T))
            else
                # be more conservative about the min eigenvalue when there is a mask
                sqrt(sqrt(eps(T)))
            end
        else
            # no checks for PD -> min_eigenvalue is not used
            nothing
        end
    else
        # user explicitly set min_eigenvalue. Just ensure that it is Real
        real(T)(min_eigenvalue)
    end

    cacheval = init_cacheval(alg, A; maxiters = maxiters, abstol = abstol, reltol = reltol, verbose = verbose)
    isfresh = true
    Tc = typeof(cacheval)

    solver = NCMSolver{typeof(A), typeof(p), typeof(alg), Tc, typeof(reltol), Union{Nothing, typeof(mask)}}(
        A, p, alg, cacheval, isfresh, abstol, reltol, maxiters, ensure_pd, min_eigenvalue, verbose, mask, A_orig
    )

    return solver
end

"""
    init(prob, algtype, args...; kwargs...)

Initialize the solver, and autotune the algorithm to the problem.
"""
function CommonSolve.init(prob::NCMProblem, algtype::Type{<:NCMAlgorithm}, args...; kwargs...)
    alg = autotune(algtype, prob)
    return init(prob, alg, args...; kwargs...)
end

"""
    init(prob, args...; kwargs...)

Initialize the solver with the default algorithm autotuned to the problem.
"""
function CommonSolve.init(prob::NCMProblem, args...; kwargs...)
    return init(prob, nothing, args...; kwargs...)
end

"""
    init(prob, algtype::Nothing, args...; kwargs...)

Initialize the solver with the default algorithm autotuned to the problem.
"""
function CommonSolve.init(prob::NCMProblem, ::Nothing, args...; kwargs...)
    algtype = default_algtype(prob; kwargs...)
    return init(prob, algtype, args...; kwargs...)
end

"""
    default_algtype(prob; kwargs...)

Get the default algorithm type for a given input matrix.
"""
function default_algtype(prob::NCMProblem; mask = nothing, kwargs...)
    mask = if mask !== nothing
        mask
    elseif prob.mask !== nothing
        prob.mask
    else
        nothing
    end

    return mask === nothing ? Newton : AcceleratedAP
end

"""
    normalize_mask(mask)

Normalizes a mask by converting it to a BitMatrix, or leaves as `nothing`. If a matrix is
given, the resulting BitMatrix is forced to be symmetric. If A[i,j] or A[j,i] is true, then
the resulting BitMatrix will have a true value in both positions. The diagonal elements are
always set to false, since the algorithm should handle setting the diagonal elements to 1.
"""
function normalize_mask(B::BitMatrix)
    n = require_square(B)

    for i in 1:(n - 1), j in (i + 1):n
        b = B[i, j] || B[j, i]
        B[i, j] = b
        B[j, i] = b
    end

    for ii in diagind(B)
        B[ii] = false
    end

    return B
end

normalize_mask(B::AbstractMatrix{T}) where {T} = normalize_mask(BitMatrix(B))
normalize_mask(::Nothing) = nothing
