using Test
using LinearAlgebra
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: default_negdef

include("CustomTestMacros.jl")
using .CustomTestMacros

@test_isdefined nearest_cor
@test_isdefined nearest_cor!

A = default_negdef()

@test_isimplemented nearest_cor(A)
@test_isimplemented nearest_cor(A, Newton())
@test_isimplemented nearest_cor(A, Newton)

@test nearest_cor(r) isa AbstractMatrix

@test_isimplemented nearest_cor!(A)
@test_isimplemented nearest_cor!(A, Newton())
@test_isimplemented nearest_cor!(A, Newton)

@test nearest_cor!(A) isa AbstractMatrix

# not symmetric input
A = rand(4, 4)
@test_nothrow nearest_cor(A)
@test_nothrow nearest_cor!(A)

# Symmetric type input
A = Symmetric(rand(4, 4))
@test_nothrow nearest_cor(A)
@test_nothrow nearest_cor!(A)

# Float16 input
A = rand(Float16, 4, 4)
@test_nothrow nearest_cor(A)
@test_nothrow nearest_cor!(A)

# (#41) uses an algorithm that supports masking when a mask is given
A, m = default_negdef(; include_mask = true)
@test_nothrow nearest_cor(A; mask = m)
@test_nothrow nearest_cor!(A; mask = m)
