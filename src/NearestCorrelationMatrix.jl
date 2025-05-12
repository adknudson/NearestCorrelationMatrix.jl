module NearestCorrelationMatrix

using LinearAlgebra
using CommonSolve: CommonSolve, init, solve, solve!
using UnPack

include("internals/Internals.jl")
using .Internals

struct NullParameters end

include("problem.jl")
include("algorithm.jl")
include("solver.jl")
include("solution.jl")

include("simple_interface.jl")

include("algorithms/newton.jl")
include("algorithms/directprojection.jl")
include("algorithms/alternatingprojections.jl")
include("algorithms/accelerated_alternatingprojections.jl")
include("algorithms/extension_algs.jl")

export
    # domain types
    NCMProblem,
    NCMSolver,
    NCMAlgorithm,
    NCMSolution,
    NullParameters,
    # helpers
    autotune,
    # common solve interface
    init,
    solve,
    solve!,
    # simple interface
    nearest_cor,
    nearest_cor!,
    # algorithms
    Newton,
    AlternatingProjections,
    AcceleratedAlternatingProjections,
    DirectProjection,
    JuMPAlgorithm

end
