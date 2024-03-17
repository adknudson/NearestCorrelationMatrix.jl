module NearestCorrelationMatrix


import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    include("submodules/Internals.jl")
    using .Internals

    using LinearAlgebra
    using CommonSolve: CommonSolve, init, solve, solve!
    using Tullio
    using UnPack
end


struct NullParameters end


include("ncm_problem.jl")
include("ncm_algorithm.jl")
include("ncm_solver.jl")
include("ncm_solution.jl")

include("legacy.jl")

include("algorithms/newton.jl")
include("algorithms/alternatingprojection.jl")
include("algorithms/directprojection.jl")


export
    NCMProblem,
    NCMSolver,
    NCMAlgorithm,
    NCMSolution,
    NullParameters,
    init, solve, solve!,
    nearest_cor, nearest_cor!,
    autotune,
    Newton,
    AlternatingProjections,
    DirectProjection

end
