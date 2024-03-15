using COSMO, JuMP, LinearAlgebra

function jump_cor(C::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(C, 1)
    q = -vec(C)
    r = vec(C)' * vec(C) / 2
    m = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => false))
    @variable(m, X[1:n, 1:n], PSD)
    x = vec(X)
    @objective(m, Min, x' * x / 2 + q' * x + r)

    for i = 1:n
        @constraint(m, X[i, i] == one(T))
    end

    status = JuMP.optimize!(m)

    return JuMP.value(X)
end
