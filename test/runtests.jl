using SafeTestsets

include("datadeps_registration.jl")

# Package Quality
@safetestset "Quality Assurance" include("test_qa.jl")

# Internals
@safetestset "Internal Utilities" include("test_internals.jl")

# API stability
@safetestset "Core API" include("test_api.jl")
@safetestset "Simplified API" include("test_simple_api.jl")

# Algorithm Robustnes
@safetestset "Constructors" include("test_constructors.jl")
@safetestset "Convergence" include("test_convergence.jl")
@safetestset "Real World Data" include("test_real_world.jl")

# Extension Packages
@safetestset "JuMP Extension" include("test_jump.jl")
