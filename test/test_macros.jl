using Test
using NearestCorrelationMatrix.Internals
using LinearAlgebra: issymmetric

"""
    @test_isdefined s

Tests whether variable `s` is defined in the current scope.

## Examples

```julia-repl
julia> @test_isdefined newvar
The symbol 'newvar' is not defined
Test Failed at ...
  Expression: false
ERROR: There was an error during testing

julia> newvar = 1

julia> @test_isdefined newvar
Test Passed

julia> function f end
f (generic function with 0 methods)

julia> @test_isdefined f
Test Passed
```
"""
macro test_isdefined(ex)
    msg = "The symbol '$ex' is not defined"
    return quote
        if @isdefined $ex
            @test true
        else
            println($(msg))
            @test false
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

"""
    @test_iscorrelation r

Run all tests for if a matrix is a valid correlation matrix
"""
macro test_iscorrelation(ex)
    return quote
        @test issquare($(esc(ex)))
        @test issymmetric($(esc(ex)))
        @test has_unit_diagonal($(esc(ex)))
        @test constrained_to_pm1($(esc(ex)))
        @test ispossemidef($(esc(ex)))
    end
end
