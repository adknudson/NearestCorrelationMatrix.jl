using Test
using Aqua


"""
    @test_isdefined expr

Test if the expression evaluates successfully, or results in an `UndefVarError`. Any other
exception will be rethrown.
"""
macro test_isdefined(ex)
    return quote
        try
            $(esc(ex))
        catch e
            if e isa UndefVarError
                @test false
            else
                rethrow()
            end
        else
            @test true
        end
    end
end


"""
    @test_isimplemented expr

Test if the expression evaluates successfully, or results in a `MethodError`. Any other
exception will be rethrown.
"""
macro test_isimplemented(ex)
    return quote
        try
            $(esc(ex))
        catch e
            if e isa MethodError
                @test false
            else
                rethrow()
            end
        else
            @test true
        end
    end
end


"""
    @test_nothrow expr

Test if the expression evaluates successfully, or throws an exception.
"""
macro test_nothrow(ex)
    return quote
        try
            $(esc(ex))
        catch
            @test false
        else
            @test true
        end
    end
end


include("Internals.jl")
include("CommonSolveApi.jl")
include("SimpleApi.jl")
include("Algorithms.jl")
# include("Robustness.jl")


Aqua.test_all(NearestCorrelationMatrix)
