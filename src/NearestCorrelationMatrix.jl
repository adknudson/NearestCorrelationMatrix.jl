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
include("algorithms/directprojection.jl")
include("algorithms/alternatingprojections.jl")


using PrecompileTools: @compile_workload

@compile_workload begin
    for T in (Float64, Float32)
        A = rand(T, 4, 4)
        nearest_cor(A, Newton(); fix_sym=true)
        nearest_cor(A, AlternatingProjections(); fix_sym=true)
        nearest_cor(A, DirectProjection(); fix_sym=true)
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
