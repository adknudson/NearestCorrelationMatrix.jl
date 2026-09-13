module CustomTestMacros

using Test
using LinearAlgebra

export
    @test_iscorrelation,
    @test_isdefined,
    @test_isimplemented,
    @test_nothrow

"""
    @test_isdefined s

Tests whether variable `s` is defined in the current scope.
"""
macro test_isdefined(ex)
    if !(ex isa Symbol)
        throw(ArgumentError("@test_isdefined requires a single Symbol (e.g., `@test_isdefined x`), got `$ex`"))
    end

    return quote
        let defined = $(esc(:(@isdefined($ex))))
            if !defined
                println(stderr, "The symbol '", $(string(ex)), "' is not defined.")
            end
            @test defined
        end
    end
end

"""
    @test_isimplemented expr

Tests if the expression evaluates successfully or results in a `MethodError`. Any other
exception will be rethrown.
"""
macro test_isimplemented(ex)
    return quote
        let ok = true
            try
                $(esc(ex))
            catch e
                if e isa MethodError
                    println(stderr, "MethodError caught: ", sprint(showerror, e))
                    ok = false
                else
                    rethrow(e)
                end
            end
            @test ok
        end
    end
end

"""
    @test_nothrow expr

Tests if the expression evaluates without throwing any exceptions.
"""
macro test_nothrow(ex)
    return quote
        let ok = true
            try
                $(esc(ex))
            catch e
                println(stderr, "Expression threw exception: ", sprint(showerror, e))
                ok = false
            end
            @test ok
        end
    end
end

"""
    @test_iscorrelation r

Run all tests for if a matrix is a valid correlation matrix
"""
macro test_iscorrelation(ex)
    return quote
        let A = $(esc(ex))
            # Check 1: Matrix must be square
            @test size(A, 1) == size(A, 2)

            if size(A, 1) == size(A, 2)
                # Check 2: Matrix must be symmetric
                @test isapprox(A, A')

                # Check 3: Diagonal elements must all equal 1
                @test isapprox(diag(A), ones(eltype(A), size(A, 1)))

                # Check 4: Off-diagonal elements must be within [-1, 1]
                @test all(x -> prevfloat(-one(eltype(A))) <= x <= nextfloat(one(eltype(A))), A)

                # Check 5: Matrix must be positive semi-definite
                # (all eigenvalues >= 0 up to floating-point tolerance)
                evals = eigvals(Symmetric(A))
                @test all(>=(-sqrt(eps(eltype(A)))), evals)
            end
        end
    end
end

end
