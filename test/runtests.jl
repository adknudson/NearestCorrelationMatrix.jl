using SafeTestsets

@time @safetestset "Quality Assurance" include("qa.jl")
@time @safetestset "Code Formatting" include("format_check.jl")
@time @safetestset "Utilities" include("internals.jl")
@time @safetestset "Common Solve API" include("api.jl")
@time @safetestset "Simple API" include("simple_api.jl")
@time @safetestset "Algorithms" include("algorithms.jl")
@time @safetestset "JuMP Extension" include("jump.jl")
