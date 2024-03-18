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

include("simple_interface.jl")

include("algorithms/newton.jl")
include("algorithms/alternatingprojections.jl")
include("algorithms/directprojection.jl")


using PrecompileTools: @compile_workload

@compile_workload begin
    for T in (Float16, Float32, Float64)
        A = rand(T, 4, 4)
        nearest_cor(A, Newton())
        nearest_cor(A, AlternatingProjections())
        nearest_cor(A, DirectProjection())
    end
end


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
