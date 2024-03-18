using Test, Aqua


macro test_isimplemented(ex)
    quote
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

macro test_isdefined(ex)
    quote
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


include("Internals.jl")
include("CommonSolveApi.jl")
include("SimpleApi.jl")


Aqua.test_all(NearestCorrelationMatrix)
