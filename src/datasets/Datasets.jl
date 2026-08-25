module Datasets

using DataDeps
using LinearAlgebra
import MAT



function __init__()
    register(DataDep(
        "bccd16",
        "BCCD16 is an invalid correlation matrix of dimension 3250 constructed from data for 3250 banks in 27 EU member states (EU 27).",
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/bccd16.mat"
    ))
    register(DataDep(
        "cor1399",
        "COR1399 is an invalid correlation matrix of dimension 1399 constructed from stock data.  The matrix was provided by investment company Orbis",
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/cor1399.mat"
    ))
    register(DataDep(
        "cor3120",
        "COR3120 is an invalid correlation matrix of dimension 3120 constructed from stock data.  The matrix was provided by investment company Orbis.",
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/cor3120.mat"
    ))
end



"""
    _vec_to_mat(x::AbstractVector; diag_val::Real = 1)

Reconstructs a full `Symmetric` matrix from a compressed strict upper-triangular
vector `x`. Fills the main diagonal with `diag_val` (default: 1).
"""
function _vec_to_mat(x::AbstractVector{T}; diag_val::Real=1) where {T}
    m = length(x)
    n = round(Int, (1 + sqrt(1 + 8m)) / 2)
    if (n * (n - 1)) ÷ 2 != m
        error("Vector length ($m) does not match strict upper triangle of an $n×$n matrix.")
    end

    # Pre-allocate matrix and fill diagonal
    A = fill(convert(T, diag_val), n, n)

    # Iterator over strict upper-triangular Cartesian indices (i < j)
    idx = 1
    for j in 2:n, i in 1:(j - 1)
        v = x[idx]
        A[i, j] = v
        A[j, i] = v
        idx += 1
    end

    return A
end

"""
    _sym_to_vec(A::AbstractMatrix)

Extracts the strict upper-triangular elements (excluding diagonal)
from a symmetric matrix `A` into a compressed vector.
"""
function _sym_to_vec(A::AbstractMatrix{T}) where {T}
    p, q = size(A)
    p == q || error("Matrix must be square, got ($p, $q).")

    # Allocate vector using the element type of A
    m = (p * (p - 1)) ÷ 2
    x = Vector{T}(undef, m)

    # Fill vector sequentially across column-major upper triangle (i < j)
    idx = 1
    for j in 2:p, i in 1:(j - 1)
        x[idx] = A[i, j]
        idx += 1
    end

    return x
end



"""
BCCD16 is a 3250×3250 invalid correlation matrix constructed from data for banks in 27 EU member states.

Source:
> Peter Benczur, Giuseppina Cannas, Jessica Cariboni, Francesca Di Girolamo, Sara Maccaferri
> and Marco Petracco Giudici, Evaluating the Effectiveness of the New EU Bank Regulatory
> Framework: A Farewell to Bail-Out, Journal of Financial Stability, 2016,
> doi 10.1016/j.jfs.2016.03.001. See Table 1.
"""
function bccd16()
    local_path = joinpath(datadep"bccd16", "bccd16.mat")
    mat_dict = MAT.matread(local_path)
    return mat_dict["A"]
end

"""
BEYU11 is a 12×12 invalid correlation matrix based on tetrachoric correlation estimates.

Source:
> Peter M. Bentler and Ke-Hai Yuan. Positive definiteness via off-diagonal scaling of a
> symmetric indefinite matrix. Psychometrika.76(1):119-123, 2011; see Table 1.
"""
function beyu11()
    x = Float64[
        0.2387, 0.6161, 0.3506, 0.6167, 0.3537, 0.8579, 0.6621, 0.2959, 0.6603, 0.7477,
        0.5173, 0.4637, 0.4093, 0.1803, 0.3537, 0.6758, 0.1931, 0.3826, 0.4705, 0.7364,
        0.3582, 0.7071, 0.1202, 0.5164, 0.6167, 0.5670, 0.1803, 0.4705, 0.7983, 0.2316,
        0.6079, 0.6218, 0.6613, 0.0605, 0.6424, 0.7149, 0.5769, 0.1708, 0.5574, 0.4705,
        0.5140, 0.4705, 0.6090, 0.4705, 0.4371, 0.4705, 0.4047, 0.4512, 0.3582, 0.5140,
        0.3582, 0.4911, 0.3582, 0.4371, 0.3745, 0.7881, 0.1161, 0.5128, 0.2966, 0.4610,
        0.6161, 0.4962, 0.5164, 0.6079, 0.4512, 0.4512
    ]
    return _vec_to_mat(x)
end

"""
BHWI01 is a 5×5 invalid correlation matrix relating to portfolio risk.

Source: second matrix in Section 2 of Vineer Bhansali and Mark B. Wise.
Forecasting portfolio risk in normal and stressed markets.
Journal of Risk, 4(1):91-106, 2001.
"""
function bhwi01()
    x = [-0.5, -0.3, 0.9, -0.25, 0.3, 0.25, -0.7, 0.7, 0.2, 0.75]
    return _vec_to_mat(x)
end

function cor1399()
    local_path = joinpath(datadep"cor1399", "cor1399.mat")
    mat_dict = MAT.matread(local_path)
    return mat_dict["x"] |> vec |> _vec_to_mat
end

"""
COR1399 is a 1399×1399 invalid correlation matrix constructed from stock data.

Source:
> The matrix was provided by investment company Orbis.
"""
function cor1399()
    local_path = joinpath(datadep"cor1399", "cor1399.mat")
    mat_dict = MAT.matread(local_path)
    return mat_dict["x"] |> vec |> _vec_to_mat
end

"""
COR3120 is a 3120×3120 invalid correlation matrix constructed from stock data.

Source:
> The matrix was provided by investment company Orbis.
"""
function cor3120()
    local_path = joinpath(datadep"cor3120", "cor3120.mat")
    mat_dict = MAT.matread(local_path)
    return mat_dict["x"] |> vec |> _vec_to_mat
end

"""
FING97 returns a 7×7 invalid correlation matrix `A` from stress testing. The output `mask`
defines positions that must remain fixed in transforming `A` to a valid correlation matrix.

Source:
> Table 4 of Christopher C. Finger. A methodology to stress correlations. RiskMetrics Monitor, Fourth Quarter:3-11, 1997
"""
function fing97()
    A = Float64[
         1.00  0.18 -0.13 -0.26  0.19 -0.25 -0.12
         0.18  1.00  0.22 -0.14  0.31  0.16  0.09
        -0.13  0.22  1.00  0.06 -0.08  0.04  0.04
        -0.26 -0.14  0.06  1.00  0.85  0.85  0.85
         0.19  0.31 -0.08  0.85  1.00  0.85  0.85
        -0.25  0.16  0.04  0.85  0.85  1.00  0.85
        -0.12  0.09  0.04  0.85  0.85  0.85  1.00
   ]

   mask = [trues(3, 3) falses(3, 4); falses(4, 3) I(4)]

   return A, mask
end

"""
HIGH02 is a 3×3 invalid correlation matrix.

Source:
> p. 334 of Nicholas J. Higham. Computing the nearest correlation matrix---A problem from finance. IMA J. Numer. Anal., 22(3):329-343, 2002.
"""
function high02()
    return Float64[1 1 0; 1 1 1; 0 1 1]
end

"""
MMB13 is a 6×6 invalid correlation matrix from foreign exchange trading data supplied by the Royal Bank of Scotland.

Source:
> page 36 of Aleksei Minabutdinov, Ilia Manaev, and Maxim Bouev.
Finding the Nearest Valid Covariance Matrix:
A FX Market Case. Working paper  Ec-07/13, Department of Economics,
European University at St. Petersburg,
St. Petersburg, Russia, 2013. Revised June 2014.
"""
function mmb13()
    A = Float64[
         0.010712 0.000654  0.002391  0.010059 -0.008321  0.001738
         0.000654 0.000004  0.002917  0.000650  0.002263  0.002913
         0.002391 0.002917  0.013225 -0.000525  0.010834  0.010309
         0.010059 0.000650 -0.000525  0.009409 -0.010584 -0.001175
        -0.008321 0.002263  0.010834 -0.010584  0.019155  0.008571
         0.001738 0.002913  0.010309 -0.001175  0.008571  0.007396
    ]

    d = sqrt.(diag(A))
    A .= A ./ (d * d')
    A[diagind(A)] .= 1.0
    return A
end

"""
    TODO
"""
function tec03()
    return Float64[
         1   -0.55 -0.15 -0.10
        -0.55 1     0.90  0.90
        -0.15 0.90  1     0.90
        -0.10 0.90  0.90  1
    ]
end

end
