# Public Documentation

Documentation for `NearestCorrelationMatrix.jl`'s public interface/

## Contents

```@contents
Pages = ["public.md"]
Depth = 2:2
```

## Index

```@index
Pages = ["public.md"]
```

## Algorithms

```@docs
NCMAlgorithm
Newton
AlternatingProjections
DirectProjection
JuMPAlgorithm
```

### Algorithm Helpers

```@docs
autotune
```

## Problem Types and Solutions

```@docs
NCMProblem
NCMSolution
NCMSolver
```

## CommonSolve.jl Interface

```@docs
init
solve!
solve
```

## Simplified Interface

```@docs
nearest_cor!
nearest_cor
```

## Traits

```@docs
NearestCorrelationMatrix.modifies_in_place
NearestCorrelationMatrix.supports_float16
NearestCorrelationMatrix.supports_parameterless_construction
NearestCorrelationMatrix.supports_symmetric
```

## NCM Tools

```@docs
NearestCorrelationMatrix.alg_name
NearestCorrelationMatrix.build_ncm_solution
NearestCorrelationMatrix.construct_algorithm
NearestCorrelationMatrix.default_algtype
NearestCorrelationMatrix.default_alias_A
NearestCorrelationMatrix.default_iters
NearestCorrelationMatrix.default_tol
NearestCorrelationMatrix.init_cacheval
```