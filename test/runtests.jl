using Test, Aqua


include("Internals.jl")
include("SimpleApi.jl")


Aqua.test_all(NearestCorrelationMatrix)
