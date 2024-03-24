using LinearAlgebra: Symmetric

export project_s, project_u

"""
    project_s(X, WHalf, WHalfInv)

Project ``X`` onto the set of symmetric positive semi-definite matrices with a W-norm.
"""
function project_s(X, Whalf, Whalfinv)
    Y = Whalfinv * project_psd(Whalf * X * Whalf) * Whalfinv
    return Symmetric(Y)
end

"""
    project_u(X)

Project ``X`` onto the set of symmetric matrices with unit diagonal.
"""
function project_u(X)
    Y = copy(X)
    setdiag!(Y, one(eltype(Y)))
    return Symmetric(Y)
end
