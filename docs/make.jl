using Documenter
using NearestCorrelationMatrix

makedocs(;
    sitename="NearestCorrelationMatrix.jl",
    modules=[NearestCorrelationMatrix],
    pages = [
        "Home" => "index.md",
        "Tutorials" => Any[
            "Quick Start" => "man/quickstart.md",
            "Even Quicker Start" => "man/evenquickerstart.md",
        ],
        "Basics" => [

        ],
        "Algorithms" => [

        ],
        "Reference" => [
            "Public API" => "lib/public.md"
        ],
        "Developers" => [
            "Developing New Algorithms" => "man/implementing.md",
            "Internals" => "lib/internals.md"
        ],
    ],
)

deploydocs(;
    repo="github.com/adknudson/NearestCorrelationMatrix.jl.git"
)
