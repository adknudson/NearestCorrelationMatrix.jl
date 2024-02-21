using Test, Aqua


include("Internals.jl")
include("PublicApi.jl")


Aqua.test_all(NearestCorrelationMatrix)
