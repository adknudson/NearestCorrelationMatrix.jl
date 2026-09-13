@testset "Constructors" verbose = true begin
    prob = NCMProblem(rand(4, 4))

    for algtype in internal_algtypes
        @testset "$(NCM.alg_name(algtype))" begin
            @test NCM.supports_parameterless_construction(algtype)

            alg = NCM.construct_algorithm(algtype)
            @test alg isa algtype

            alg = autotune(algtype, prob)
            @test alg isa algtype

            # supports_parameterless_construction works on the type, not the instance
            @test_throws MethodError NCM.supports_parameterless_construction(alg)
        end
    end
end


function test_simple(algtype)
    return @testset "$(NCM.alg_name(algtype))" begin
        r0 = default_negdef(Float64)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        cache = init(prob, alg)
        sol = solve!(cache)

        @test_iscorrelation sol.X

        # Issue #19: Solution must always be Symmetric
        @test sol.X isa Symmetric

        # Handle Symmetric type matrices
        r0 = default_negdef(Float64)
        prob = NCMProblem(Symmetric(r0))
        alg = autotune(algtype, prob)
        @test_nothrow solve(prob, alg)

        # Handle Float16 input matrices
        r0 = default_negdef(Float16)
        prob = NCMProblem(r0)
        alg = autotune(algtype, prob)
        if NCM.supports_float16(alg)
            @test_nothrow solve(prob, alg)
        else
            @test_throws Exception solve(prob, alg)
            @test_nothrow solve(prob, alg; convert_f16 = true)
        end
    end
end

@testset "Convergence Tests" verbose = true begin
    for algtype in internal_algtypes
        test_simple(algtype)
    end
end
