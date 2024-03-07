module NearestCorrelationMatrix

include("submodules/Internals.jl")
using .Internals


using LinearAlgebra
using PrecompileTools


"""
    NearestCorrelationAlgorithm

Defines the abstract type for nearest correlation algorithms.
"""
abstract type NearestCorrelationAlgorithm end


"""
    default_alg()

Returns the default algorithm for finding the nearest correlation matrix.
"""
default_alg() = Newton()


export
    nearest_cor,
    nearest_cor!,
    Newton,
    AlternatingProjection,
    DirectProjection,
    default_alg


# algorithm iterface
include("nearestcor.jl")

# algorithms
include("newton.jl")
include("alternatingprojection.jl")
include("directprojection.jl")


@setup_workload begin
    function make_data(T, n)
        x = 2 * rand(T, n, n) .- one(T)
        setdiag!(x, one(T))
        symmetric!(x)
        return x
    end

    f32 = make_data(Float32, 5)
    f64 = make_data(Float64, 5)

    @compile_workload begin
        alg = Newton(τ = 1e-10, tol=1e-8)

        nearest_cor(f32, alg)
        nearest_cor(f64, alg)

        alg = AlternatingProjection()

        nearest_cor(f32, alg)
        nearest_cor(f64, alg)

        alg = DirectProjection()

        nearest_cor(f32, alg)
        nearest_cor(f64, alg)
    end
end

end
