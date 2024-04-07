# Even Quicker Start

The `nearest_cor` and `nearest_cor!` methods are the "batteries included" methods for finding the nearest correlation matrix, and are designed to just work. They also have a final check to ensure that the solution is strictly positive-definite, as often that is required for other algorithms (e.g. Cholesky decomposition). They will also return the solution matrix directly rather than an `NCMSolution`.

```@example ncmprob2
using NearestCorrelationMatrix
G = rand(4, 4)
X = nearest_cor(G)
```

You can still pass in your desired algorithm and optional arguments:

```@example ncmprob2
nearest_cor(G, AlternatingProjections(); uplo=:L)
```