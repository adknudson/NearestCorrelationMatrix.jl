using Test
using NearestCorrelationMatrix
using JuliaFormatter

if VERSION >= v"1.6"
    @test JuliaFormatter.format(NearestCorrelationMatrix; verbose=false, overwrite=false)
end
