module NearestCorrelationMatrix

using LinearAlgebra
using CommonSolve: CommonSolve, init, solve, solve!

include("internals/Internals.jl")
using .Internals


struct NullParameters end


include("NCMProblem.jl")
include("NCMAlgorithm.jl")
include("NCMSolver.jl")
include("NCMSolution.jl")

include("simple_interface.jl")

include("algorithms/Newton.jl")
include("algorithms/DirectProjection.jl")
include("algorithms/AlternatingProjections.jl")
include("algorithms/AcceleratedAP.jl")
include("algorithms/JuMPAlgorithm.jl")

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
    AcceleratedAP,
    AlternatingProjections,
    DirectProjection,
    JuMPAlgorithm,
    Newton

end
