using Test
using Aqua


include("test_macros.jl")

include("Internals.jl")
include("CommonSolveApi.jl")
include("SimpleApi.jl")
include("Algorithms.jl")
# include("Robustness.jl")


Aqua.test_all(NearestCorrelationMatrix)
