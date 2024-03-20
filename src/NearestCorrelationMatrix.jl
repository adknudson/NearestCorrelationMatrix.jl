module NearestCorrelationMatrix


import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    include("internals/Internals.jl")
    using .Internals

    using LinearAlgebra
    using CommonSolve: CommonSolve, init, solve, solve!
    using UnPack
end


struct NullParameters end


include("ncm_problem.jl")
include("ncm_algorithm.jl")
include("ncm_solver.jl")
include("ncm_solution.jl")

include("simple_interface.jl")

include("algorithms/newton.jl")
include("algorithms/directprojection.jl")
include("algorithms/alternatingprojections.jl")
include("algorithms/extension_algs.jl")


PrecompileTools.@compile_workload begin
        A = rand(Float64, 4, 4)
        nearest_cor(A, Newton(); fix_sym=true)
        nearest_cor(A, AlternatingProjections(); fix_sym=true)
        nearest_cor(A, DirectProjection(); fix_sym=true)
end


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
    DirectProjection,
    JuMPAlgorithm


end
