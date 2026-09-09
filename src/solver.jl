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
- `verbose`: Whether to print extra information. Defaults to `false`.
- `mask`: The fixed-element mask (a normalized `Bool` matrix), or `nothing` if unmasked.
- `A_orig`: A copy of the (transformed) input holding the fixed-element values, used as the
  value source for `project_f!`. `nothing` when there is no mask.
"""
mutable struct NCMSolver{TA, P, Talg, Tc, Ttol, Tm, TAorig}
    A::TA           # the input matrix
    p::P            # parameters
    alg::Talg       # ncm algorithm
    cacheval::Tc    # store algorithm cache here
    isfresh::Bool   # false => cacheval is set wrt A, true => update cacheval wrt A
    abstol::Ttol    # absolute tolerance for convergence
    reltol::Ttol    # relative tolerance for convergence
    maxiters::Int   # maximum number of iterations
    ensure_pd::Bool # ensures that the resulting matrix is positive definite
    verbose::Bool   # whether to print extra information
    mask::Tm        # fixed-element mask (a Bool matrix), or nothing
    A_orig::TAorig  # copy of the (transformed) input holding the fixed-element values; nothing if unmasked
end

"""
    default_algtype(prob)

Get the default algorithm type for a given input matrix.
"""
default_algtype(::NCMProblem) = Newton

"""
    init(prob, alg, args...; kwargs...)

Initialize the solver with the given algorithm.

## Keyword Arguments

- `alias_A`: Whether to alias the matrix ``A`` or use a copy by default. When `true`,
  algorithms that operate in place can save memory by reusing ``A``. Defaults to `true` if
  the algorithm is known not to modify ``A``, and `false` otherwise.
- `mask`: An optional fixed-element mask. For every upper-triangular position ``(i, j)`` with
  ``i < j`` where ``mask[i, j]`` is truthy, the solution must retain the value ``A[i, j]``. The
  mask is normalized and symmetrized. Requires an algorithm that supports `supports_mask`.
  Defaults to `nothing` (no elements held fixed).
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
        verbose::Bool = false,
        kwargs...
    )
    # Resolve the effective mask first: an explicit `mask=` kwarg overrides any mask set on the
    # problem; otherwise fall back to the problem's (already-normalized) mask, then to nothing.
    mask = if mask !== nothing
        verbose && println("Using fixed-element mask")
        normalize_mask(prob.A, mask)
    elseif prob.mask !== nothing
        verbose && println("Using fixed-element mask from the problem")
        prob.mask
    else
        nothing
    end

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

    @unpack A, p = prob

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
            symmetric!(copy(A.data), sym_uplo(A.uplo))
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
                symmetric!(A, uplo)
            end
        else
            error(
                "Input matrix is not symmetric. Pass the argument `fix_sym=true`, or ensure " *
                    "that your input matrix is symmetric before solving."
            )
        end
    end

    if eltype(A) === Float16 && !supports_float16(alg)
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
    A_orig = mask === nothing ? nothing : copy(A)

    # Guard against type mismatch for user-specified reltol/abstol
    reltol = real(eltype(A))(reltol)
    abstol = real(eltype(A))(abstol)

    cacheval = init_cacheval(alg, A, maxiters, abstol, reltol, verbose)
    isfresh = true
    Tc = typeof(cacheval)

    solver = NCMSolver{typeof(A), typeof(p), typeof(alg), Tc, typeof(reltol), Union{Nothing, Matrix{Bool}}, typeof(A_orig)}(
        A, p, alg, cacheval, isfresh, abstol, reltol, maxiters, ensure_pd, verbose, mask, A_orig
    )

    return solver
end

"""
    init(prob, algtype, args...; kwargs...)

Initialize the solver, and autotune the algorithm to the problem.
"""
function CommonSolve.init(
        prob::NCMProblem, algtype::Type{<:NCMAlgorithm}, args...; kwargs...
    )
    return init(prob, autotune(algtype, prob), args...; kwargs...)
end

"""
    init(prob, args...; kwargs...)

Initialize the solver with the default algorithm autotuned to the problem.
"""
function CommonSolve.init(prob::NCMProblem, args...; kwargs...)
    return init(prob, nothing, args...; kwargs...)
end

"""
    init(prob, nothing, args...; kwargs...)

Initialize the solver with the default algorithm autotuned to the problem.
"""
function CommonSolve.init(prob::NCMProblem, ::Nothing, args...; kwargs...)
    return init(prob, default_algtype(prob), args...; kwargs...)
end
