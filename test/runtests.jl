using Test, Aqua


include("Internals.jl")

include("LegacyApi.jl")


Aqua.test_all(NearestCorrelationMatrix)
