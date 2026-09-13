using Test, Aqua
using LinearAlgebra
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals
import NearestCorrelationMatrix as NCM
using InteractiveUtils: subtypes

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("Datasets.jl")
using .Datasets

include("macros.jl")

const internal_algtypes = setdiff(subtypes(NCMAlgorithm), (JuMPAlgorithm,))

# Package Quality
@testset "Aqua" begin
    Aqua.test_all(NearestCorrelationMatrix)
end

# Internals
include("test_internals.jl")

# API stability
include("test_api.jl")
include("test_simple_api.jl")

# Algorithm Robustnes
include("test_algorithms.jl")
include("test_real_world.jl")

# Extension Packages
include("test_jump.jl")
