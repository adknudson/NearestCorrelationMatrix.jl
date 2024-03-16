module NearestCorrelationMatrix


import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    include("submodules/Internals.jl")
    using .Internals

    using BlockArrays
    using LinearAlgebra
    using LinearSolve
    using LineSearches
    using CommonSolve: CommonSolve, init, solve, solve!, step!
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

include("experimental/newton_revise.jl")
include("experimental/borsdorf_prec_newton.jl")

export
    NCMProblem,
    NCMSolver,
    NCMAlgorithm,
    NCMSolution,
    NullParameters,
    init, solve, solve!, step!,
    nearest_cor, nearest_cor!,
    AlternatingProjections,
    DirectProjection,
    Newton,
    NewtonNew,
    NewtonBorsdorf

end
