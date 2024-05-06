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

All implementations are written in Python, utilizing several external libraries.
To run scripts, the following is required:

- Python 3.11 with installed packages listed in the `requirements.txt` file.

- Configured IBM ILOG Constraint-Programming Optimizer.
  IBM ILOG Constraint-Programming Optimizer is part of the IBM ILOG CPLEX Optimization Studio
  commercial software package.
  Community and academic editions are available.
  See [online](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-cp-optimizer).

Python version at least 3.11 is required as we utilize several functionalities introduced in that version.
However, note that the `docplex` library does not fully support this version.
We had no issues with this partial incompatibility.

## Running scripts

A simple example demonstrating the two implemented algorithms
is contained in the `rcpsp_sandbox/example.py` script file.
Input and resulting data of this example is located in the `example` directory.

All experiments are run using the `experiments.py` Python script file,
located in the `rcpsp_sandbox` source directory.
Following is the invocation help:

```
usage: experiments.py [-h] [--save_plots] [--addition ADDITION]
                      [--migration MIGRATION]

options:
  -h, --help             show this help message and exit
  --save_plots           Determines whether to save plots to files
  --addition ADDITION    Cost of capacity addition (default is 5)
  --migration MIGRATION  Cost of capacity migration (default is 1)
```

Running this script computes the experiments.
The script evaluates the algorithms (or loads computed evaluations from data files),
computes statistics, which are then saved to data files,
and finally creates plots of the results.

We provide the computed results in the `data` directory.
In this directory, base instances, modified instances,
and computed evaluations and evaluation KPIs are stored
in directories named accordingly.
Before running the experiments, the `modified_instances` directory has to be extracted
from a zip file named `modified_instances.zip`.

## Project overview

The project is divided into the following sub-packages, each containing several modules:

- `rcpsp_sandbox.instances` --- modules for manipulating problem instances.
- `rcpsp_sandbox.solver` --- modules for solving the problem instances.
- `rcpsp_sandbox.bottlenecks` --- modules implementing the presented algorithms
        and hosting experiment evaluations.

The `rcpsp_sandbox.instances` sub-package contains several modules for manipulating with problem instances.
Those modules are used in the rest of the project, forming a core data infrastructure.
In the `problem_instance` module contains the definition of the `ProblemInstance`
class, representing the problem instance defined in \cref{def:problem-instance}.
The `io` module is used for parsing and serializing problem instance object,
be it in the original PSPLIB file format, or in JSON.
The `problem_modified` module provides the `modify_instance` function,
which for a given problem instance returns a `ProblemModifier` object.
The interface of the object allows the user to modify all aspects of the problem instance.
Most important, it implements the modifications described in \cref{sec:problem-statement/scheduling},
namely, splitting the precedence graph, introducing time-variable resource capacities,
assigning job due dates.
The `algorithms` module implements several algorithms regarding problem instances,
mostly various precedence graph traversals.

The `rcpsp_sandbox.solver` sub-package contains, among others, the `solver` module.
This module facilitates the solving of problem instances via the `Solver` class.
The `Solver` class contains a single `solve` method, which takes in either a problem instance,
of a built model to solve.
The function utilizes the `docplex` library to call the IBM ILOG Constraint-Programming Optimizer
solver, which finds (optimal) solutions to given models.
If a problem instance is given to the `Solver.solve` method,
the solver builds a standard model described in \cref{sec:problem-statement/constraint-programming-model}.
For a finer control over the model, the `model_builder` module provides the `build_model` function.
This function, for a given problem instance, return an initialized `ModelBuilder` object.
The interface of this object allows for the creation of specific models,
introducing only selected constraints, restraining job intervals, or choosing alternate optimization goals.
The `solution` module contains the definition of the `Solution` abstract class,
along with several implementing derived classes and utility functions concerning solutions to the problem instance models.

The `rcpsp_sandbox.bottlenecks` sub-package contains the implementations of the \acl{iira} and \acl{ssira},
and a framework for evaluating the algorithms on problem instances.
The `improvements` module contains the algorithms' implementations
together with implementations of helper functions from \cref{sec:attachments/algorithms-functions-procedures}.
The `evaluations` module contains the `evaluate_algorithms` and the `compute_evaluation_kpis` functions.
Those are central for the experiments: the former runs the algorithms with specified parameters on a given problem instance
and return the sets of computed evaluations, the latter computes the experiment KPIs of the evaluations.

Following is a minimal working example of evaluating both algorithms utilizing the `evaluate_algorithms` function
on a problem instance parsed from the `instance.json` file.
The set of all algorithm parameters is the exact same set used for the experiments conducted in \cref{chap:numerical-experiments}.

```py
import rcpsp_sandbox.instances.io as iio
from rcpsp_sandbox.bottlenecks.evaluations import evaluate_algorithms

instance = iio.parse_json("instance.json", is_extended=True)
evaluations = evaluate_algorithms(instance, [
    (ScheduleSuffixIntervalRelaxingAlgorithm(), {
        "max_iterations": [1, 2, 3],
        "relax_granularity": [1],
        "max_improvement_intervals": [1, 2, 3, 4, 5, 6],
        "interval_sort": ["improvement", "time"]}),
    (IdentificationIndicatorRelaxingAlgorithm(), {
        "metric": ["auau", "mrur"],
        "granularity": [4, 8],
        "convolution_mask": ["pre1", "around", "post"],
        "max_iterations": [1, 2, 3],
        "max_improvement_intervals": [1, 2, 3, 4],
        "capacity_addition": [4, 10]}),
])
```

The `rcpsp_sandbox.manager` module contains the `ExperimentManager` class,
which manages loading and saving of experiment evaluations, KPIs, and problem instances.
It can be passed to the `evaluate_algorithms` function
to attempt loading existing evaluations from files before computing them anew.
