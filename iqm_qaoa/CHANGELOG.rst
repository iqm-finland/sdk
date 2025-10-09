=========
Changelog
=========

Version 1.34.0 (2025-10-08)
===========================

Features
--------

- Improve the greedy function to find a path in the QPU topology graphs by introducing backtracking.

Version 1.33.0 (2025-10-08)
===========================

Bug fixes
--------

- Make dictionary `self._log2hard` of class `Mapping` a lasting attribute, instead of evaluating it lazily for performance reasons.

Version 1.32.0 (2025-10-03)
===========================

Bug fixes
---------

- Skip following mypy imports to iqm-data-definitions until errors are fixed. :issue:`SW-1670`

Version 1.31.0 (2025-09-30)
===========================

Features
--------

- Update dependency on iqm-client

Version 1.30.0 (2025-09-16)
===========================

Features
--------

- Add a couple of options to ``maxcut_generator`` to enable it to generate ``WeightedMaxCutInstance`` with two basic distributions of weights.

Version 1.29.0 (2025-09-12)
===========================

Features
--------

- Update dependency on station-control

Version 1.28.0 (2025-09-12)
===========================

Features
--------

- Update dependency on iqm-client

Version 1.27.0 (2025-09-11)
===========================

Features
--------

- Bump dependencies. :issue:`SW-1663`

Version 1.26.0 (2025-09-05)
===========================

Bug fixes
---------

- Add an option to specify ``vartype`` when instantiating a ``QUBOInstance`` from a ``numpy`` array or a ``networkx`` graph.

Version 1.25.0 (2025-09-05)
===========================

Features
---------

* Adds an optional input parameter to method ``build_qiskit`` of ``Routing`` which builds the circuit so that pairs of identical ``CNOT`` gates are cancelled.

Version 1.24.0 (2025-09-04)
===========================

Bugfix
--------

- Make ``mypy`` type checking stricter by checking a couple extra optional things.
- Make ``ruff`` linting stricter by adding a couple extra things to check.
- Make ``ruff`` linting stricter by removing some per-file ignores.

Version 1.23.0 (2025-09-03)
===========================

Features
--------

- Enable ruff rule for missing annotations and mark exemptions.

Version 1.22.0 (2025-09-01)
===========================

Features
--------

* Creates a list of possible names of edge / node attributes and when a graph is used to instantiate certain problem classes, it goes through the list of names and looks for these attributes in the graph.
* Puts the needed helper functions (and constants) into a separate module ``graph_utils.py``.
* Move the function ``relabel_graph_nodes`` from ``qubo.py`` into ``graph_utils.py``.

Version 1.21.0 (2025-09-01)
===========================

Features
--------

* Allow the ``sample`` method of ``SamplerSimulation`` and ``SamplerResonance`` to pass ``seed`` to the internal ``transpiled_circuit`` function to fix the random component of transpilation.
* Allow the ``estimate`` method of ``EstimatorFromSampler`` to pass seed to the inner ``sample`` method of the provided sampler (or any other keyword arguments).

Bug fixes
---------

* Fixes the example in the module docstring of ``maxcut.py``, which used an outdated name of a solver function.
* Set the default seed of most function / methods to ``None``, so that when the user doesn't provide it, the outputs are random and not deterministic.

Version 1.20.0 (2025-09-01)
===========================

Features
--------

* Add ``mis_generator`` function modeled on ``maxcut_generator`` to generate random instances of the ``MISInstance`` problem instance.
* Add basic unit tests for both ``mis_generator`` and for ``maxcut_generator``.

Version 1.19.0 (2025-08-20)
===========================

Features
--------

- Update dependency on iqm-client

Version 1.18.0 (2025-08-20)
===========================

Bug fixes
---------

- Add explicit cross-component requirements

Version 1.17.0 (2025-08-20)
===========================

Bug fixes
---------

* All methods that take `counts` as input now have a warning not to use the raw output of counts from running a `qiskit` experiment and to reverse the order of the bitstrings instead.
* Add clarification to the samplers that they do this reversing of the bitstrings.

Version 1.16.0 (2025-08-08)
===========================

Bugfix
--------

- Fix where :method:`transpiled_circuit` on STAR devices could result in a transpiler error when `optimization_level` is set too large, in which case the transpiler could place single qubit gates onto the resonator. :issue:`SW-944`.

Version 1.15.0 (2025-08-08)
===========================

Bug Fixes
---------
- Replace calls to ``numpy.random`` with creation of an RNG object and calls to its methods.

Version 1.14.0 (2025-08-01)
===========================

Bug fixes
---------

- Fix `qubo_graph` and `qubo_matrix` methods of `ConstrainedQuadraticInstance` so that they re-compute the internal attribute `_bqm` everytime they're called and therefore they're up to date with the problem instance.
- Add a small test that checks if all the QUBO representations of `ConstrainedQuadraticInstance` agree, i.e., `qubo_graph`, `qubo_matrix` and `bqm`.

Version 1.13.0 (2025-07-28)
===========================

Bug fixes
---------

- Fix type hints in `sn_router`, so that it accepts any `QPU` (it still checks if its layout has 2D integer coordinates).
- Add a test for `sn_router` with a fake backend.

Version 1.12.0 (2025-07-24)
===========================

Bug fixes
---------

- Address type checking flags.
- Change the methods `EstimatorBackend` and `SamplerBackend` and their subclasses to accept only `QUBOQAOA`, to avoid violating Liskov Substitution Principle.

Version 1.11.0 (2025-07-23)
===========================

Bug fixes
---------

- Add optional transpilation step to `SamplerSimulation`, so that it can accept more simulators than just `AerSimulator`, e.g., our `IQMFakeApollo`.

Version 1.10.0 (2025-07-21)
===========================

Bug fixes
---------

- Add a fallback routine to `_get_embedding` which allows it to embed larger problems on QPUs on which it would previously fail, by using a greedy algorithm for finding a Hamiltonian path in a graph.
- Add a small test to check that it works.

Version 1.9.0 (2025-07-21)
==========================

Features
--------

- Add a Jupyter notebook showing how one can control / influence which qubits on the QPU get selected for execution of the circuit.
- Add input ``**kwargs`` to ``transpiled_circuit`` which get passed to the inner ``transpile`` call (from Qiskit).

Version 1.8.0 (2025-07-09)
==========================

Features
--------

- Enable mypy type checking in CI and add temporary type ignores to the source code. :issue:`SW-1615`

Version 1.7.0 (2025-07-09)
==========================

Features
--------

- Normalize all line endings to LF. No functional changes.

Version 1.6.0 (2025-06-25)
==========================

Bug fixes
---------

- Fix ``seed`` not working in ``maxcut_generator`` (it wasn't passed over to random graph generators inside of the function).

Version 1.5.0 (2025-06-23)
==========================

Bug fixes
---------

- Fix `__init__.py` docstring in `star` transpilation submodule.


Version 1.4.0 (2025-06-23)
==========================

Features
--------

- Add a citation of Elisabeth's QAOA paper to the documentation (docstring under `TreeQAOA` class).


Version 1.3.0 (2025-06-20)
==========================

Bug fixes
---------

- Fix link to readme in ``pyproject.toml`` to make project description visible in PyPI.


Version 1.2.0 (2025-06-19)
==========================

Features
--------

- Bump version for an updated repo organization. No functional changes. :issue:`SW-1578`


Version 1.1 (2025-06-06)
========================

* Remove ``exa-core`` dependency.


Version 1.0 (2025-06-06)
========================

* Remove the usage of ``mapomatic`` in ``transpiled_circuit``. The transpiled circuit is now just transpiled, not also placed on the best patch of the QPU.
* Remove ``mapomatic`` dependency.


Version 0.30 (2025-05-21)
=========================

* Cosmetic changes to almost all docstrings, aimed at polishing the generated documentation.
    * Fixing links (to functions / classes / methods) within the library.
    * Adding a few more links to outside libraries.
    * Improving consistency about what is documented.


Version 0.29 (2025-05-15)
=========================

* Add a new problem instance class: weighted maximum independent set ``MaximumWeightISInstance``.
    * Create a new class ``ISInstance`` to serve as parent for ``MISInstance`` and ``MaximumWeightISInstance``, carrying methods common for both subclasses.
* Add a new problem instance class: weighted maxcut ``WeightedMaxCutInstance``.


Version 0.28 (2025-05-09)
=========================

* Add a new jupyter notebook ``Training the QAOA.ipynb`` showcasing different ways to train the QAOA.
* Add the new notebook to the end-to-end testing.


Version 0.27 (2025-05-09)
=========================

* Add an option to optimize the angles by minimizing CVaR.


Version 0.26 (2025-04-29)
=========================

* Add links to the source code to API Reference in documentation.


Version 0.25 (2025-04-29)
=========================

* Add the option to calculate Conditional Value at Risk (CVaR) for all problem classes, given a dictionary of counts.
    * Add a post-processing method that keeps only the best / worst quantile of measurement results, given a dictionary of counts (and a quantile).


Version 0.24 (2025-05-09)
=========================

* Add two new jupyter notebook examples showing how the QAOA library is used.
    * A notebook showing how the library can be used to solve a sparse maxcut problem - ``Sparse Maxcut.ipynb``.
    * A notebook showing how the library can be used to solve a constrained problem (portfolio optimization with a fixed budget) - `Portfolio Optimization.ipynb`.
    * Rename the SK model notebook from ``small_sk_model_example.ipynb`` to ``SK Model and Transpilation.ipynb``.
* Add the three above-mentioned notebooks to the documentation using ``myst-nb``.
* Minor fixes of constructing the ``qiskit`` circuit for star QPU.
    * Correct the usage of ``MoveGate``.
    * Swap ``move_in`` and ``move_out`` when the layers are reversed during circuit construction.
* Add custom drawing method for ``RoutingStar`` (ovewriting the same method of ``Routing``).


Version 0.23 (2025-03-27)
=========================

* ``twine`` version bump.
* Expand testing for swap network helper functions.


Version 0.22 (2025-03-26)
=========================

* Remake the subclasses of ``QPU``.
    * Add a subclass that creates an instance of itself from ``IQMBackend``.
    * Add an option to generate the QPU layout automatically using ``planar_layout`` from ``networkx``.
* Add a check requiring the QPU layout to use integer coordinates when using the swap network transpiler.
* Allow the transpilers to work on any size QPU.
    * The swap network transpiler looks for rectangles within the provided QPU.
    * The greedy transpiler looks for almost circle / square / rectangle in the provided QPU.
    * The hardwired transpiler looks for matches of its specific subgraphs in the provided QPU.


Version 0.21 (2025-02-20)
=========================

* Add Q-score and SK-model end-to-end examples as Jupyter notebooks. These examples can also be used for testing.
* Add comparisons of various transpilation methods as Jupyter notebooks.
* There has been a special ``iqm-qaoa`` account created for IQM Resonance to be used with end-to-end testing.


Version 0.20 (2025-02-20)
=========================

* Rename ``ConstrainedQUBOInstance`` to ``ConstrainedQuadraticInstance`` and make it independent from ``QUBOInstance``, so that now it inherits directly from ``ProblemInstance``.
* Make most functionality of ``ConstrainedQuadraticInstance`` based on ``ConstrainedQuadraticModel`` from the ``dimod`` package.


Version 0.19 (2025-02-18)
=========================

* Add package version information to package documentation


Version 0.18 (2025-02-11)
=========================

* Add two post-processing methods to ``ConstrainedQUBOInstance`` and implement them in ``MISInstance``.


Version 0.17 (2025-02-04)
=========================

* Create a new module ``backends.py`` containing backend classes which now take the role of estimator (of expectation values) and sampler.
* Modify (and add) tests for the backends.
* Remove backend-related functionality from the ``QUBOQAOA`` class.
* Create a new module ``circuits.py`` containing functions that construct (quantum) circuits from a ``QUBOQAOA`` object. Formerly the functions were methods of the ``QUBOQAOA`` class.


Version 0.16 (2025-01-31)
=========================

* Change the way that (optional) initial angles are inputted when ``QUBOQAOA`` is initialized. Previously one variable ``initial_angles`` was used. Now it's possible to use input variables ``gammas`` and ``betas`` instead.
* Add setters for ``self.betas``, ``self.gammas`` and ``self.angles`` of ``QUBOQAOA``.


Version 0.15 (2025-01-24)
=========================

* Generate package documentation with ``sphinx`` and upload it to GitLab Pages for each released version of the package.


Version 0.14 (2025-01-08)
=========================

* Replace local copy of ``mapomatic`` code with ``iqm-mapomatic`` package.


Version 0.13 (2025-01-07)
=========================

* Fix estimator based on QUIMB, adding a warning.


Version 0.12 (2024-12-16)
=========================

* Add a method ``circuit`` to the QUBOQAOA class, which builds the circuit and transpiles it to the HW.
* Implement the "hardwired" transpilation strategy.
* Implement the "sparse"/greedy/Ayse-Martin-Fedor transpilation strategy.
* Implement the swap network transpilation strategy.


Version 0.11 (2024-11-22)
=========================

* Change the implementation of Goemans-Williamson algorithm to improve performance.
* Replace the structure of the problem instance classes to only store the BinaryQuadraticModel representation of the problem and calculate the other representations lazily.


Version 0.10 (2024-11-11)
=========================

* Add TreeQAOA class with tree angle setting scheme.


Version 0.9 (2024-11-05)
========================

* Make classical solvers accept either a nx.Graph or a problem instance.
* Add tests for classical algorithms for maximum independent set and for constraints checker.


Version 0.8 (2024-10-30)
========================

* Refine problem classes, removing duplicate methods.


Version 0.7 (2024-10-23)
========================

* Add first batch of unit tests.


Version 0.6 (2024-10-21)
========================

* Update build tools to latest available versions.


Version 0.5 (2024-10-16)
========================

* Add license file.


Version 0.4 (2024-10-16)
========================

* Downgrade build tools to known working versions.


Version 0.3 (2024-10-16)
========================

* Update `setuptools_scm` configuration to fix package version string generation.


Version 0.2 (2024-10-15)
========================

* Fix release process


Version 0.1 (2024-10-15)
========================

* First public-ish release
