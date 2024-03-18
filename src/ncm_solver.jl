"""
    NCMSolver

Common interface for solving NCM problems. Algorithm-specific cache is stored in the
`cacheval` field.

- `A`: the input matrix
- `p`: the parameters (currently unused)
- `alg`: the NCM algorithm to use
- `cacheval`: algorithm-specific cache
- `isfresh`: `true` if the cacheval hasn't been set yet
- `abstol`:
- `reltol`:
- `maxiters`:
- `verbose`:
"""
mutable struct NCMSolver{TA, P, Talg, Tc, Ttol}
    A::TA         # the input matrix
    p::P          # parameters
    alg::Talg     # ncm algorithm
    cacheval::Tc  # store algorithm cache here
    isfresh::Bool # false => cacheval is set wrt A, true => update cacheval wrt A
    abstol::Ttol  # absolute tolerance for convergence
    reltol::Ttol  # relative tolerance for convergence
    maxiters::Int # maximum number of iterations
    ensure_pd::Bool # ensures that the resulting matrix is positive definite
    verbose::Bool
end


"""
    init(prob, alg)

Initialize the solver with the given algorithm.

## Arguments

- `alias_A`
- `abstol`
- `reltol`
- `maxiters`
- `fix_sym`
- `convert_f16`
- `force_f16`
- `uplo`
- `ensure_pd`
- `verbose`
"""
function CommonSolve.init(prob::NCMProblem, alg::NCMAlgorithm, args...;
    alias_A = default_alias_A(alg, prob.A),
    abstol = default_tol(real(eltype(prob.A))),
    reltol = default_tol(real(eltype(prob.A))),
    maxiters::Int = default_iters(alg, prob.A),
    fix_sym::Bool=false,
    convert_f16::Bool=false,
    force_f16::Bool=false,
    uplo::Symbol=:U,
    ensure_pd::Bool=false,
    verbose::Bool = false,
    kwargs...
)
    @unpack A, p = prob

    A = if alias_A
        verbose && println("Aliasing A")
        A
    elseif A isa Symmetric
        if supports_symmetric(alg)
            verbose && println("Creating a Symmetric copy of A")
            copy(A)
        else
            verbose && println("$(alg_name(alg)) does not support Symmetric types. " *
            "Creating a symmetric copy of A.data")
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
                verbose && println("Input matrix is not symmetric. Creating a Symmetric view " *
                "of the $(uplo==:U ? "upper" : "lower") part of the matrix")
                A = Symmetric(A, uplo)
            else
                verbose && println("Input matrix is not symmetric. Copying the " *
                "$(uplo==:U ? "upper" : "lower") part of the matrix")
                symmetric!(A, uplo)
            end
        else
            error("Input matrix is not symmetric. Pass the argument `fix_sym=true`, or ensure " *
            "that your input matrix is symmetric before solving.")
        end
    end


    if eltype(A) === Float16 && !supports_float16(alg)
        if convert_f16
            verbose && println("Input matrix has eltype Float16, which $(alg_name(alg)) does " *
            "not support. Converting to `AbstractMatrix{Float32}`")
            A = convert(AbstractMatrix{Float32}, A)
        elseif force_f16
            verbose && println("Input matrix has eltype Float16, which $(alg_name(alg)) does " *
            "not support. `force_f16=true` so using input matrix anyway.")
        else
            error("Input matrix has eltype Float16, which $(alg_name(alg)) does not support. " *
            "Pass either the argument `convert_f16=true` or `force_f16=true`, or convert " *
            "your input matrix to an `AbstractMatrix{Float32}` before solving.")
        end
    end


    # Guard against type mismatch for user-specified reltol/abstol
    reltol = real(eltype(A))(reltol)
    abstol = real(eltype(A))(abstol)

    cacheval = init_cacheval(alg, A, maxiters, abstol, reltol, verbose)
    isfresh = true
    Tc = typeof(cacheval)

    solver = NCMSolver{typeof(A), typeof(p), typeof(alg), Tc, typeof(reltol)}(
        A, p, alg, cacheval, isfresh, abstol, reltol, maxiters, ensure_pd, verbose)

    return solver
end


"""
    default_algtype(prob)

Get the default algorithm type for a given input matrix.
"""
default_algtype(::NCMProblem) = Newton


"""
    init(prob, algtype)

Initialize the algorithm and the solver.
"""
function CommonSolve.init(prob::NCMProblem, algtype::Type{<:NCMAlgorithm}, args...; kwargs...)
    algType = default_algtype(prob)
    alg = autotune(algType, prob)
    return init(prob, alg, args...; kwargs...)
end


"""
    init(prob)

Initialize the solver with the default algorithm.
"""
function CommonSolve.init(prob::NCMProblem, args...; kwargs...)
    return init(prob, nothing, args...; kwargs...)
end

"""
    init(prob, nothing)

Initialize the solver with the default algorithm.
"""
function CommonSolve.init(prob::NCMProblem, ::Nothing, args...; kwargs...)
    algType = default_algtype(prob)
    alg = autotune(algType, prob)
    return init(prob, alg, args...; kwargs...)
end
