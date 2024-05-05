# RCPSPSandbox

This repository is a sandbox for the work on the Resource Constrained Project Scheduling Problem (RCPSP).
The repository contains the following modules:

- `instances`
  - Parsing and serializing problem instances.
  - Modifying problem instances.
  - Algorithms for instance-graph computations.
- `solver`
  - Building `docplex` models for the problem instances.
  - Solving the models using IBM ILOG Constraint-Programming Optimizer.
- `bottlenecks`
  - Implementation of the Identification Indicator Relaxing Algorithm and the Schedule Suffix Interval Relaxing Algorithm.
  - Evaluating algorithms for relaxing bottlenecks in solutions.

## Requirements

This repository requires Python 3.11 with installed packages listed in the `requirements.txt` file,
using a virtual environment is recommended.

To run experiments, the docplex library should be fully configured.
This requires the IBM ILOG Constraint-Programming Optimizer, available [online](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-cp-optimizer).

## Running experiments

Experiments can be run from the `experiments.py` script file. The script has the following invocation:

```
usage: experiments.py [-h] [--save_plots] [--addition ADDITION] [--migration MIGRATION]

options:
  -h, --help             show this help message and exit
  --save_plots           Determines whether to save plots to files
  --addition ADDITION    Cost of capacity addition
  --migration MIGRATION  Cost of capacity migration
```
