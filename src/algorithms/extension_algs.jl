struct JuMPAlgorithm{O,A,K} <: NCMAlgorithm
    optimizer::O
    args::A
    kwargs::K

    function JuMPAlgorithm(optimizer, args...; kwargs...)
        ext = Base.get_extension(@__MODULE__, :NCMJuMPExt)
        if ext === nothing
            error("JuMPAlgorithm requires that JuMP is loaded, i.e. `using JuMP`")
        else
            return new{typeof(optimizer), typeof(args), typeof(kwargs)}(optimizer, args, kwargs)
        end
    end
end
