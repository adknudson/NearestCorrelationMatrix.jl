module Datasets

using DataDeps
using LinearAlgebra
import MAT

export
    bccd16,
    beyu11,
    bhwi01,
    cor1399,
    cor3120,
    fing97,
    high02,
    mmb13,
    tec03,
    tyda99r1,
    tyda99r2,
    tyda99r3,
    usgs13

"""
    vec_to_mat(x::AbstractVector; diag_val::Real = 1)

Reconstructs a full symmetric matrix from a compressed strict upper-triangular
vector `x`. Fills the main diagonal with `diag_val` (default: 1).
"""
function vec_to_mat(x::AbstractVector{T}; diag_val::Real = 1) where {T}
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
        0.3582, 0.7071, 0.1202, 0.5164, 0.6167, 0.567, 0.1803, 0.4705, 0.7983, 0.2316,
        0.6079, 0.6218, 0.6613, 0.0605, 0.6424, 0.7149, 0.5769, 0.1708, 0.5574, 0.4705,
        0.514, 0.4705, 0.609, 0.4705, 0.4371, 0.4705, 0.4047, 0.4512, 0.3582, 0.514,
        0.3582, 0.4911, 0.3582, 0.4371, 0.3745, 0.7881, 0.1161, 0.5128, 0.2966, 0.461,
        0.6161, 0.4962, 0.5164, 0.6079, 0.4512, 0.4512,
    ]
    return vec_to_mat(x)
end

"""
BHWI01 is a 5×5 invalid correlation matrix relating to portfolio risk.

Source: second matrix in Section 2 of Vineer Bhansali and Mark B. Wise.
Forecasting portfolio risk in normal and stressed markets.
Journal of Risk, 4(1):91-106, 2001.
"""
function bhwi01()
    x = [-0.5, -0.3, 0.9, -0.25, 0.3, 0.25, -0.7, 0.7, 0.2, 0.75]
    return vec_to_mat(x)
end

"""
COR1399 is a 1399×1399 invalid correlation matrix constructed from stock data.

Source:
> The matrix was provided by investment company Orbis.
"""
function cor1399()
    local_path = joinpath(datadep"cor1399", "cor1399.mat")
    mat_dict = MAT.matread(local_path)
    return mat_dict["x"] |> vec |> vec_to_mat
end

"""
COR3120 is a 3120×3120 invalid correlation matrix constructed from stock data.

Source:
> The matrix was provided by investment company Orbis.
"""
function cor3120()
    local_path = joinpath(datadep"cor3120", "cor3120.mat")
    mat_dict = MAT.matread(local_path)
    return mat_dict["x"] |> vec |> vec_to_mat
end

"""
FING97 returns a 7×7 invalid correlation matrix `A` from stress testing. The output `mask`
defines positions that must remain fixed in transforming `A` to a valid correlation matrix.

Source:
> Table 4 of Christopher C. Finger. A methodology to stress correlations. RiskMetrics Monitor, Fourth Quarter:3-11, 1997
"""
function fing97()
    A = Float64[
        1.0 0.18 -0.13 -0.26 0.19 -0.25 -0.12
        0.18 1.0 0.22 -0.14 0.31 0.16 0.09
        -0.13 0.22 1.0 0.06 -0.08 0.04 0.04
        -0.26 -0.14 0.06 1.0 0.85 0.85 0.85
        0.19 0.31 -0.08 0.85 1.0 0.85 0.85
        -0.25 0.16 0.04 0.85 0.85 1.0 0.85
        -0.12 0.09 0.04 0.85 0.85 0.85 1.0
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
        0.010712 0.000654 0.002391 0.010059 -0.008321 0.001738
        0.000654 0.000004 0.002917 0.00065 0.002263 0.002913
        0.002391 0.002917 0.013225 -0.000525 0.010834 0.010309
        0.010059 0.00065 -0.000525 0.009409 -0.010584 -0.001175
        -0.008321 0.002263 0.010834 -0.010584 0.019155 0.008571
        0.001738 0.002913 0.010309 -0.001175 0.008571 0.007396
    ]

    d = sqrt.(diag(A))
    A .= A ./ (d * d')
    A[diagind(A)] .= 1.0
    return A
end

"""
TEC03 is a 4×4 invalid correlation matrix from stress testing.

Source:
> ̂Ω on p.~86 in
Saygun Turkay, Eduardo Epperlein, and Nicos Christofides. Correlation
stress testing for value-at-risk. Journal of Risk, 5(4):75-89, 2003
"""
function tec03()
    return Float64[
        1 -0.55 -0.15 -0.1
        -0.55 1 0.9 0.9
        -0.15 0.9 1 0.9
        -0.1 0.9 0.9 1
    ]
end

"""
TYDA99R1 is an 8×8 invalid correlation matrix from resource allocation modeling.

Source:
> Rajesh Tyagi and Chandrasekhar Das. Grouping customers for better
allocation of resources to serve correlated demands. Computers &
Operations Research, 26(10-11):1041-1058, 1999.
"""
function tyda99r1()
    x = [0.1 -1 0.4 0.8 -0.1 -0.2 0.7 0.4 -0.3 0.8 -0.1 0.4 0.8 -0.3 0 0.3 0.2 0.9 -0.3 -0.5 -0.4 0.3 0.6 0.8 0.1 1 -0.2 0.6]
    return vec_to_mat(vec(x))
end

"""
TYDA99R2 is an 8×8 invalid correlation matrix from resource allocation modeling.

Source:
> Rajesh Tyagi and Chandrasekhar Das. Grouping customers for better
allocation of resources to serve correlated demands. Computers &
Operations Research, 26(10-11):1041-1058, 1999.
"""
function tyda99r2()
    x = [0.1 1 0.4 0.8 0.1 0.2 0.7 0.4 0.3 0.8 0.1 0.4 0.8 0.3 0 0.3 0.2 0.9 0.3 0.5 0.4 0.3 0.6 0.8 0.1 1 0.2 0.6]
    return vec_to_mat(vec(x))
end

"""
TYDA99R3 is an 8×8 invalid correlation matrix from resource allocation modeling.

Source:
> Rajesh Tyagi and Chandrasekhar Das. Grouping customers for better
allocation of resources to serve correlated demands. Computers &
Operations Research, 26(10-11):1041-1058, 1999.
"""
function tyda99r3()
    x = [-0.5 -0.5 0.5 0.5 -0.5 -0.5 -0.5 0.5 0.5 -0.5 0.5 0.5 -0.5 0.5 0.5 0.5 0.5 -0.5 0.5 -0.5 0.5 -0.5 0.5 0.5 -0.5 0.5 0.5 -0.5]
    return vec_to_mat(vec(x))
end

"""
USGS13 is a 94×94 invalid correlation matrix from carbon
dioxide storage assessment units for the Rocky Mountains
region of the USA and was generated during the national
assessment of carbon dioxide storage resources.

The output `mask` defines positions that must remain fixed in transforming `A` to a valid
correlation matrix.

Source:
> U.S. Geological Survey Geologic Carbon Dioxide Storage
Resources Assessment Team. National Assessment of Geologic Carbon
Dioxide Storage Resources---Results (Ver. 1.1, September 2013),
September 2013. Provided by Madalyn Blondes of the U.S. Geological
Survey, email correspondence; permission to use given.
"""
function usgs13()
    local_path = joinpath(datadep"usgs13", "Rocky_Mountain_Region_CORR.mat")
    mat_dict = MAT.matread(local_path)
    A = mat_dict["A"]

    block_sizes = [12, 5, 1, 14, 12, 1, 10, 4, 5, 9, 13, 8]
    total_dim = sum(block_sizes)

    # 1. Pre-allocate a dense BitMatrix full of falses
    mask = falses(total_dim, total_dim)

    # 2. Fill in block diagonal submatrices where A is non-zero
    start_idx = 1
    for sz in block_sizes
        stop_idx = start_idx + sz - 1

        # Submatrix block from A
        B = A[start_idx:stop_idx, start_idx:stop_idx]

        # Set non-zero elements to true in the dense matrix
        mask[start_idx:stop_idx, start_idx:stop_idx] .= B .!= 0

        start_idx = stop_idx + 1
    end

    return A, mask
end

end
