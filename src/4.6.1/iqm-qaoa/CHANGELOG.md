# Changelog

## Version 2.0.1 (2026-08-05)

### Features

- Prepare package for `4.6.1` release. No functional changes.

## Version 2.0.0 (2026-07-07)

### Breaking changes

- The BQM variables are allowed to internally use any custom labels (instead of
    forcing the labels to be consecutive integers starting at 0). So the
    internal dictionaries `orig_to_new_labels` and `new_to_orig_labels` are
    removed. Wherever possible, the variables are now addressed by these labels.
    If a translation to integer labelling is necessary (e.g., when interpreting
    a bitstring as a solution), the order of the variables in
    `BinaryQuadraticModel.variables` is used.
- The helper function `relabel_bqm_cqm_variables` has been removed.
- The helper function `relabel_graph_nodes` has been removed.
- `LogQubit` is changed from an alias of `int` to an alias of `Variable` from
    the `dimod` package (which itself is an alias of `Hashable`).
- The input parameter to `QUBOInstance` (and its subclasses)
    `allow_custom_var_names` is removed.
- The `fix_variables` methods of various `ProblemInstance` subclasses have all
    had their input parameter `original_labels` removed.
- The internal helper functions `_get_s_line` and `_embed_chain` now take a list
    of `Variable` instead of just an integer (representing a number of variables).
- The classical solvers `greedy_max_cut`, `goemans_williamson`, `greedy_mis` and
    `bron_kerbosch` for solving maxcut and MIS problems now only accept the
    respective problem instance (`MaxCutInstance` or `MISInstance`), not a
    `nx.Graph` anymore. This is done because they need to know the ordering
    of variables (accessible through the problem's BQM) to generate solution
    bitstrings correctly.
- Rework the function `ham_graph_to_ham_operator` to `ham_bqm_to_ham_operator`
    and correspondingly change its input signature.

- The class `BaseMapping` can be instantiated (to reduce unnecessary
  boilerplate).
- The class `Mapping` now doesn't take a `BinaryQuadraticModel` on input, but
  only its variables, as it doesn't need any more information anyway.
- **`cnot_list` is now a read-only property** (previously a mutable list
  attribute). It returns `compute_cnots + uncompute_cnots`.
- **Default output of `build_qiskit` changed.** With
  `remove_cnots_first_layer=True` (the new default), the first QAOA layer is
  emitted differently (fewer CNOTs, and composed under a permuted qubit map).
  The resulting statevector is equivalent up to global phase for the `|+⟩`
  initial state, but the emitted circuit is not gate-for-gate identical to the
  previous output. Pass `remove_cnots_first_layer=False` to restore the old
  behaviour.
- **`rz()` after uncomputation now raises.** Workflows that added RZ gates once
  uncomputation had begun (previously permitted) will now raise `RuntimeError`.
- The parity twine network router has been rewritten using the new
  `CircuitSynthesis` formalism, which is more suitable for it.
- The type aliases `Parity`, `LineQubit` and `PTNLayer` have been removed.
- The classes `LineMappingPTN` and `LineRoutingPTN` have been removed.
- The API for estimators/samplers has been changed. All estimators and samplers
  now use a single externally-facing method `estimate` / `sample` which
  dispatches based on the type of the input (`QUBOQAOA` vs. `HUBOQAOA`).
- The `sample` and `estimate` methods have moved from `QUBOQAOA` to the
  parent class `QAOA` and no longer accept `**kwargs`.
- `EstimatorBackend` and `SamplerBackend` are no longer abstract base classes.
  Custom backends should override private methods (e.g., `_estimate_qubo`)
  instead of the public `estimate`/`sample` methods.
- The `train` method of `QAOA` no longer routes kwargs between the inner
  `estimate` and `scipy.minimize`. Now all kwargs go directly to `minimize`.
- Sampler backends (`SamplerSimulation`, `SamplerResonance`) now accept
  transpiler keyword arguments at construction time instead of at each `sample`
  call.
- The helper function `ham_bqm_to_ham_operator` has been modified to also
  include the constant term of the BQM, to be in line with the newly-added
  `ham_bp_to_ham_operator`.

### Bug fixes

- Fix some typing bugs:
  - Relax unnecessarily strict type hints (e.g., `list` -> `Iterable`).
  - Fix incorrect type hints in `two_color_mapper.py` (`LogEdge` -> `HardEdge`).
  - Relax the variable type in the generic `ProblemInstance` to `Any`.
  - Fix incorrect type hints in `ptn.py` (`LogQubit` -> `LineQubit`).
- Some occurances of using the `random` package have been replaced by using
    `numpy.random` instead.
- Type of error corrected (`ValueError` -> `TypeError`) when attempting to
    instantiate `QUBOInstance` with an invalid `qubo_object`.
- Demoted an `AttributeError` in `_extract_problem_info` to a warning.
- The warning at the end of `build_qiskit` method of `CircuitSynthesis` (now
  called `build_qiskit_phase_separator`) about unrealized interactions does not
  get raised when the remaining unrealized interactions all have magnitude of 0.
- Fix LSP violation between `QAOA` and `QUBOQAOA` by parametrizing `QAOA`.
- Switch from pinned `requests` dependency to range.
- Domain changed from meetiqm.com to iqm.tech
- Fix a bug in `EstimatorQUIMB` so that it doesn't crash if there is no constant
  term in the Hamiltonian (instead just considering it to be 0).
- Add an input parameter `seed` to `greedy_max_cut`, thus allowing the user to
    seed the randomness. Previously, a fixed seed was used, which defeats the
    purpose of randomness.
- Move the `train` method from `QUBOQAOA` to `QAOA` because the logic of
  training will be shared among other QAOA types. The custom behavior specific
  to the different QAOA types has been delegated to two simple private methods.
- Instead of relying on experimental / unstable
  `IQMClient.get_observation_quality_metrics`, the initialization of
  `CrystalQPUFromBackend` now pulls the gate fidelities from the Qiskit backend
  given as input.
- The method `ProblemInstance.restore_fixed_variable_bitstring` has been fixed
  to respect the ordering of variable labels defined in
  `ProblemInstance.variables` and `ProblemInstance.original_variables`, instead
  of assuming that the variable labels themselves are integers.
- The `draw_problem` function from `graph_utils.py` has been fixed to respect
  the ordering of variable labels, instead of assuming that the variable labels
  themselves are integers.
- The method `quality` of classes `HUBOInstance` and `QUBOInstance` now only
  accepts bitstrings whose length equals the number of (free) problem variables.

### Features

- The `greedy_mis` solver now allows to input an optional seed to be used to
    pseudo-randomly break ties inside of the solver.

- Add two new classes `ParityMapping` and `CircuitSynthesis` to aid in
  synthesizing quantum circuits, as opposed to routing them with the `Mapping`
  and `Routing` classes (or their sibling / child classes).
  - The `ParityMapping` class keeps track of which hardware qubit carries which
    logical parity of the logical qubits.
  - The `CircuitSynthesis` class allows constructing circuits by adding CNOT and
    RZ gates, either constructing the corresponding Hamiltonian during the
    process or implementing a given Hamiltonian.
  - Note that there is currently no algorithm for constructing the circuit from
    a given Hamiltonian alone. The user needs to construct the circuit
    themselves.
- Add a Jupyter Notebook showcasing the use of the two now classes.

- **CNOT-count reduction on the first QAOA layer.**
  `CircuitSynthesis.build_qiskit` gained a `remove_cnots_first_layer` parameter
  (default `True`). Because the first phase separator acts on the all-`|+⟩`
  state, the compute CNOTs are trivial there, so the layer can be emitted as the
  reversed compute half only — producing a statevector-equivalent circuit with
  fewer two-qubit gates.
- **`remove_cnots` mode for the phase separator.**
  `build_qiskit_phase_separator` gained a `remove_cnots` flag. When set, it
  starts from the fully-computed mapping (`pre_uncomputation_mapping`) and
  replays the compute CNOTs in reverse instead of emitting the full compute +
  uncompute sequence.
- **Explicit compute/uncompute phase separation.** New `begin_uncompute()`
  method ends the computing phase and snapshots the current mapping into
  `pre_uncomputation_mapping`. This makes it possible to write *custom*
  uncomputation routines (arbitrary CNOTs after `begin_uncompute()`), not just
  the mirrored default.
- **New `computing` property** signalling whether synthesis is still in the
  computing phase (`True` until `begin_uncompute()`/`uncompute_parities()` is
  called).
- **Separate CNOT tracking.** Compute-phase and uncompute-phase CNOTs are now
  stored in `compute_cnots` and `uncompute_cnots` respectively.

- Various improvements to the functionality of `CircuitSynthesis`.
  - An attribute `allow_interactions` is added to the dataclass `CNOTStep`. This
    controls whether interaction gates (RZ) may be added directly after the CNOT
    or not. It allows better control over the shape of the `CircuitSynthesis`
    quantum circuits.
  - The `build_qiskit` method is renamed to `build_qiskit_phase_separator` and
    a new `build_qiskit` method is added to build the entire QAOA qiskit
    circuit.

- Various improvements to the functionality of `ParityMapping`.
  - An error is raised if the initial mapping maps to parities of multiple
    qubits, such as e.g., `{log_qb1, log_qb2}`. For the initial mapping, only
    mapping to individual qubits is permitted `{log_qb1}` (or the empty set
    `set()`).
  - A new attribute `parity_transform_matrix` of the class is introduced. This
    keeps track of the mapping as a boolean matrix. It is updated independently
    from the `parity_mapping` attribute.

- Add support for higher-order unconstrained binary optimization (HUBO) problems
  - The HUBO problems are instantiated as `HUBOInstance`, effectively a wrapper
    for `BinaryPolynomial` from `dimod` package.
  - New subclass of `QAOA` for HUBO problems has been added, `HUBOQAOA`.
  - New function for constructing Qiskit circuits for HUBO problems has been
    added, `qiskit_circuit_hubo`. These circuits are not transpiled.
  - A helper function for defining multi-qubit ZZ..Z rotations has been added,
    `rnz`.
  - New transforming function `ham_bp_to_ham_operator` has been added which
    transforms `BinaryPolynomial` into the corresponding `SparsePauliOp` to be
    used with Qiskit.
  - `EstimatorStateVector` and `EstimatorFromSampler` now support both QUBO and
    HUBO problems.
- New utility function `positions_to_pauli_string` for building Pauli Z strings.

- Add optional dependency groups with pinned versions to `iqm-qaoa`:
  - `pin-iqm`: pins inter-IQM package dependencies to the exact versions that
    are tested and released together.
  - `pin-all`: like `pin-iqm`, but additionally pins all transitive third-party
    dependencies to exact locked versions.
  - `notebook-pin-iqm`: includes `notebook` extra dependencies with inter-IQM
    packages pinned to co-released versions.
  - `notebook-pin-all`: like `notebook-pin-iqm`, but additionally pins all
    transitive third-party dependencies.
- The class `ProblemInstance` has new attributes `variables` and
  `original_variables`. Those track the ordering (i.e., mapping of integers) of
  the variable lables of the currently "active" variables of the problem and the
  original variables, before any have been fixed.
  - For `QUBOInstance` and `ISInstance` problem classes, the `variables`
    attribute points to the variables of the internal BQM / CQM. The
    `original_variables` attribute is a snapshot of the BQM / CQM variables at
    the instantiation of the problem class.
  - For problem classes which don't implement `fix_variables` method, the
    attributes `variables` and `original_variables` are not defined.

- Add a method `uncompute_parities` to `CircuitSynthesis`, which adds CNOTs at
  the end of the circuit synthesis quantum circuit to restore the initial
  mapping of the parities to the hardware qubits.
- Add a nested class `CNOTStep` to represent CNOT gates in `CircuitSynthesis`.
- Add the ability to estimate the expectation values of the Hamiltonian for HUBO
  problems, i.e., add an implementation of `_estimate_hubo` to `EstimatorQUIMB`.
- Add a method `fix_constraint_violation_bitstring` for the class
  `MaximumWeightISInstance`, allowing its solutions to be pruned. The pruning
  procedure uses a greedy min-weight vertex cover solver on the graph of
  constraint violations.
- Add a new solver for `MaximumWeightISInstance` called `greedy_mwis`. This
  solver uses a greedy minimum-weight vertex cover solver internally.

## Version 1.40.0

### Bug fixes

- Added a `BaseRouting` abstract base class and made classes `Routing` and
  `RoutingStar` its subclasses.
- Removes a warning about using fewer logical qubits than there are hardware
    qubits.
    This is the case for almost all circuits and it's not a problem at all.
- Fixes a bug where the Sparse transpiler would fail if given a problem with
    some nodes of degree 0 (no neighbors).
- Rename an internal variable in `_get_s_line` to not overshadow built-in Python
    `map`.
- Make the internal `transpile` from Qiskit use the layout created during
    our routing, instead of doing its own "layout" pass.
- Change iteration range in the function `linear_ramp_schedule` to make it
    set the angles correctly.
- Don't apply single-qubit gates unnecessarily on ancilla qubits
    (in `transpiled_circuit`).

### Features

- Improve documentation of `EDGE_ATTR_PRIORITY` and `NODE_ATTR_PRIORITY`.
- Add a boolean input parameter to `sn_router` called `do_line_swapping` to
    determine if the line swap networks strategy should be employed (and
    whether it should be done preferentially).
- Line swapping strategy is implemented, by default as a fallback if the
    grid swapping strategy fails.
- Add an optional boolean `verbose` input to the method `set_tree_angles` of
  `TreeQAOA` to control its output.
- Add the Parity Twine Network routing algorithm.
- Make it so that when using `CrystalQPUFromBackend`, the calibration data of
    the backend is used to construct the hardware (topology) graph,
    so that any missing edges are not used in the transpilation.
- Add a new method `estimate_correlations_z` to all estimators, allowing them
    to be used to estimate the expectation values of `<Z>` and `<ZZ>`
    operators on arbitrary qubits.
- Make the `transpiler` input an enum instead of a string.
- Make it so that when the upper and lower bounds on solution quality are
    calculated (by brute-force), the best/worst bitstrings are also saved,
    representing the problem solution and un-solution (or whatever we can
    call it).
- The bitstrings are saved as a set (which may have more than one element if
    the solution is degenerate).

## Version 1.39.0 (2025-11-19)

### Features

- Make `fix_variables` of certain problem classes accepts the original problem variable names on input if `original_labels==True`.
- Upgrades to the "sparse router". Previously, the router started with an edge-coloring of the problem graph and then it picked the sets of edges corresponding to the two largest colors to define the first two layers of interaction gates in the QAOA phase separator.
  - Now it tracks the `max_iter_color_pairs` largest color pairs (or fewer if there is not that many pairs to choose from). For each of these color pairs, it performs the routing as previously.
  - Once the routings are done, it uses a function `key_best_route` to determine which of these routings is the best one and that is outputted. The default is "number of routing layers", but the user may use their own function with the signature `Callable[[Routing], float | int]` (lower outputs are considered better).
- Removed a print statement from the method `quality` of the class `ConstrainedQuadraticInstance` as it was a bit spammy and not very important actually.

### Bug fixes

- Allow independent set problem instances (`MISInstance` and `MaximumWeightISInstance`) to also use arbitrary problem variable names on initialization.
- Added a `BaseRouting` abstract base class and made classes `Routing` and `RoutingStar` its subclasses.
- Make the dictionary `hard2log` (from `Mapping`) contain also unassigned hardware qubits (with the value `None`).

## Version 1.38.0 (2025-10-23)

### Bug fixes

- When initializing `QUBOQAOA`, the `bqm` of the problem instance (representing the problem as a QUBO) is transformed into a `hamiltonian_bqm`, representing it as a Hamiltonian (i.e., the `hamiltonian_bqm` contains the coefficients in front of the Z and ZZ terms in the corresponding Hamiltonian).

## Version 1.37.0 (2025-10-22)

### Features

- Add the calculation of the "tree angles" according to *Missing Puzzle Pieces in the Performance Landscape of the Quantum Approximate Optimization Algorithm* by Elisabeth Wybo.
- Add testing of the tree angle calculation.

### Bug fixes

- Make the docstrings of `QUBOQAOA` and `QAOA` more descriptive about the input angles.

## Version 1.36.0 (2025-10-14)

### Features

- Add the option to plot a problem graph with a solution highlighted (as a standalone function and methods which call this function).
- Harmonize the way that problems with custom-named variables are treated internally.

## Version 1.35.0 (2025-10-09)

### Features

- Update dependency on iqm-client

## Version 1.34.0 (2025-10-08)

### Features

- Improve the greedy function to find a path in the QPU topology graphs by introducing backtracking.

## Version 1.33.0 (2025-10-08)

### Bug fixes

- Make dictionary `self._log2hard` of class `Mapping` a lasting attribute, instead of evaluating it lazily for performance reasons.

## Version 1.32.0 (2025-10-03)

### Bug fixes

- Skip following mypy imports to iqm-data-definitions until errors are fixed.

## Version 1.31.0 (2025-09-30)

### Features

- Update dependency on iqm-client

## Version 1.30.0 (2025-09-16)

### Features

- Add a couple of options to `maxcut_generator` to enable it to generate `WeightedMaxCutInstance` with two basic distributions of weights.

## Version 1.29.0 (2025-09-12)

### Features

- Update dependency on station-control

## Version 1.28.0 (2025-09-12)

### Features

- Update dependency on iqm-client

## Version 1.27.0 (2025-09-11)

### Features

- Bump dependencies.

## Version 1.26.0 (2025-09-05)

### Bug fixes

- Add an option to specify `vartype` when instantiating a `QUBOInstance` from a `numpy` array or a `networkx` graph.

## Version 1.25.0 (2025-09-05)

### Features

- Adds an optional input parameter to method `build_qiskit` of `Routing` which builds the circuit so that pairs of identical `CNOT` gates are cancelled.

## Version 1.24.0 (2025-09-04)

### Bug fixes

- Make `mypy` type checking stricter by checking a couple extra optional things.
- Make `ruff` linting stricter by adding a couple extra things to check.
- Make `ruff` linting stricter by removing some per-file ignores.

## Version 1.23.0 (2025-09-03)

### Features

- Enable ruff rule for missing annotations and mark exemptions.

## Version 1.22.0 (2025-09-01)

### Features

- Creates a list of possible names of edge / node attributes and when a graph is used to instantiate certain problem classes, it goes through the list of names and looks for these attributes in the graph.
- Puts the needed helper functions (and constants) into a separate module `graph_utils.py`.
- Move the function `relabel_graph_nodes` from `qubo.py` into `graph_utils.py`.

## Version 1.21.0 (2025-09-01)

### Features

- Allow the `sample` method of `SamplerSimulation` and `SamplerResonance` to pass `seed` to the internal `transpiled_circuit` function to fix the random component of transpilation.
- Allow the `estimate` method of `EstimatorFromSampler` to pass seed to the inner `sample` method of the provided sampler (or any other keyword arguments).

### Bug fixes

- Fixes the example in the module docstring of `maxcut.py`, which used an outdated name of a solver function.
- Set the default seed of most function / methods to `None`, so that when the user doesn't provide it, the outputs are random and not deterministic.

## Version 1.20.0 (2025-09-01)

### Features

- Add `mis_generator` function modeled on `maxcut_generator` to generate random instances of the `MISInstance` problem instance.
- Add basic unit tests for both `mis_generator` and for `maxcut_generator`.

## Version 1.19.0 (2025-08-20)

### Features

- Update dependency on iqm-client

## Version 1.18.0 (2025-08-20)

### Bug fixes

- Add explicit cross-component requirements

## Version 1.17.0 (2025-08-20)

### Bug fixes

- All methods that take `counts` as input now have a warning not to use the raw output of counts from running a `qiskit` experiment and to reverse the order of the bitstrings instead.
- Add clarification to the samplers that they do this reversing of the bitstrings.

## Version 1.16.0 (2025-08-08)

### Bug fixes

- Fix where `transpiled_circuit` on STAR devices could result in a transpiler error when `optimization_level` is set too large, in which case the transpiler could place single qubit gates onto the resonator.

## Version 1.15.0 (2025-08-08)

### Bug fixes

- Replace calls to `numpy.random` with creation of an RNG object and calls to its methods.

## Version 1.14.0 (2025-08-01)

### Bug fixes

- Fix `qubo_graph` and `qubo_matrix` methods of `ConstrainedQuadraticInstance` so that they re-compute the internal attribute `_bqm` everytime they're called and therefore they're up to date with the problem instance.
- Add a small test that checks if all the QUBO representations of `ConstrainedQuadraticInstance` agree, i.e., `qubo_graph`, `qubo_matrix` and `bqm`.

## Version 1.13.0 (2025-07-28)

### Bug fixes

- Fix type hints in `sn_router`, so that it accepts any `QPU` (it still checks if its layout has 2D integer coordinates).
- Add a test for `sn_router` with a fake backend.

## Version 1.12.0 (2025-07-24)

### Bug fixes

- Address type checking flags.
- Change the methods `EstimatorBackend` and `SamplerBackend` and their subclasses to accept only `QUBOQAOA`, to avoid violating Liskov Substitution Principle.

## Version 1.11.0 (2025-07-23)

### Bug fixes

- Add optional transpilation step to `SamplerSimulation`, so that it can accept more simulators than just `AerSimulator`, e.g., our `IQMFakeApollo`.

## Version 1.10.0 (2025-07-21)

### Bug fixes

- Add a fallback routine to `_get_embedding` which allows it to embed larger problems on QPUs on which it would previously fail, by using a greedy algorithm for finding a Hamiltonian path in a graph.
- Add a small test to check that it works.

## Version 1.9.0 (2025-07-21)

### Features

- Add a Jupyter notebook showing how one can control / influence which qubits on the QPU get selected for execution of the circuit.
- Add input `**kwargs` to `transpiled_circuit` which get passed to the inner `transpile` call (from Qiskit).

## Version 1.8.0 (2025-07-09)

### Features

- Enable mypy type checking in CI and add temporary type ignores to the source code.

## Version 1.7.0 (2025-07-09)

### Features

- Normalize all line endings to LF. No functional changes.

## Version 1.6.0 (2025-06-25)

### Bug fixes

- Fix `seed` not working in `maxcut_generator` (it wasn't passed over to random graph generators inside of the function).

## Version 1.5.0 (2025-06-23)

### Bug fixes

- Fix `__init__.py` docstring in `star` transpilation submodule.

## Version 1.4.0 (2025-06-23)

### Features

- Add a citation of Elisabeth's QAOA paper to the documentation (docstring under `TreeQAOA` class).

## Version 1.3.0 (2025-06-20)

### Bug fixes

- Fix link to readme in `pyproject.toml` to make project description visible in PyPI.

## Version 1.2.0 (2025-06-19)

### Features

- Bump version for an updated repo organization. No functional changes.

## Version 1.1 (2025-06-06)

- Remove `exa-core` dependency.

## Version 1.0 (2025-06-06)

- Remove the usage of `mapomatic` in `transpiled_circuit`. The transpiled circuit is now just transpiled, not also placed on the best patch of the QPU.
- Remove `mapomatic` dependency.

## Version 0.30 (2025-05-21)

- Cosmetic changes to almost all docstrings, aimed at polishing the generated documentation.
  - Fixing links (to functions / classes / methods) within the library.
  - Adding a few more links to outside libraries.
  - Improving consistency about what is documented.

## Version 0.29 (2025-05-15)

- Add a new problem instance class: weighted maximum independent set `MaximumWeightISInstance`.
  - Create a new class `ISInstance` to serve as parent for `MISInstance` and `MaximumWeightISInstance`, carrying methods common for both subclasses.
- Add a new problem instance class: weighted maxcut `WeightedMaxCutInstance`.

## Version 0.28 (2025-05-09)

- Add a new jupyter notebook `Training the QAOA.ipynb` showcasing different ways to train the QAOA.
- Add the new notebook to the end-to-end testing.

## Version 0.27 (2025-05-09)

- Add an option to optimize the angles by minimizing CVaR.

## Version 0.26 (2025-04-29)

- Add links to the source code to API Reference in documentation.

## Version 0.25 (2025-04-29)

- Add the option to calculate Conditional Value at Risk (CVaR) for all problem classes, given a dictionary of counts.
  - Add a post-processing method that keeps only the best / worst quantile of measurement results, given a dictionary of counts (and a quantile).

## Version 0.24 (2025-05-09)

- Add two new jupyter notebook examples showing how the QAOA library is used.
  - A notebook showing how the library can be used to solve a sparse maxcut problem - `Sparse Maxcut.ipynb`.
  - A notebook showing how the library can be used to solve a constrained problem (portfolio optimization with a fixed budget) - `Portfolio Optimization.ipynb`.
  - Rename the SK model notebook from `small_sk_model_example.ipynb` to `SK Model and Transpilation.ipynb`.
- Add the three above-mentioned notebooks to the documentation using `myst-nb`.
- Minor fixes of constructing the `qiskit` circuit for star QPU.
  - Correct the usage of `MoveGate`.
  - Swap `move_in` and `move_out` when the layers are reversed during circuit construction.
- Add custom drawing method for `RoutingStar` (ovewriting the same method of `Routing`).

## Version 0.23 (2025-03-27)

- `twine` version bump.
- Expand testing for swap network helper functions.

## Version 0.22 (2025-03-26)

- Remake the subclasses of `QPU`.
  - Add a subclass that creates an instance of itself from `IQMBackend`.
  - Add an option to generate the QPU layout automatically using `planar_layout` from `networkx`.
- Add a check requiring the QPU layout to use integer coordinates when using the swap network transpiler.
- Allow the transpilers to work on any size QPU.
  - The swap network transpiler looks for rectangles within the provided QPU.
  - The greedy transpiler looks for almost circle / square / rectangle in the provided QPU.
  - The hardwired transpiler looks for matches of its specific subgraphs in the provided QPU.

## Version 0.21 (2025-02-20)

- Add Q-score and SK-model end-to-end examples as Jupyter notebooks. These examples can also be used for testing.
- Add comparisons of various transpilation methods as Jupyter notebooks.
- There has been a special `iqm-qaoa` account created for IQM Resonance to be used with end-to-end testing.

## Version 0.20 (2025-02-20)

- Rename `ConstrainedQUBOInstance` to `ConstrainedQuadraticInstance` and make it independent from `QUBOInstance`, so that now it inherits directly from `ProblemInstance`.
- Make most functionality of `ConstrainedQuadraticInstance` based on `ConstrainedQuadraticModel` from the `dimod` package.

## Version 0.19 (2025-02-18)

- Add package version information to package documentation

## Version 0.18 (2025-02-11)

- Add two post-processing methods to `ConstrainedQUBOInstance` and implement them in `MISInstance`.

## Version 0.17 (2025-02-04)

- Create a new module `backends.py` containing backend classes which now take the role of estimator (of expectation values) and sampler.
- Modify (and add) tests for the backends.
- Remove backend-related functionality from the `QUBOQAOA` class.
- Create a new module `circuits.py` containing functions that construct (quantum) circuits from a `QUBOQAOA` object. Formerly the functions were methods of the `QUBOQAOA` class.

## Version 0.16 (2025-01-31)

- Change the way that (optional) initial angles are inputted when `QUBOQAOA` is initialized. Previously one variable `initial_angles` was used. Now it's possible to use input variables `gammas` and `betas` instead.
- Add setters for `self.betas`, `self.gammas` and `self.angles` of `QUBOQAOA`.

## Version 0.15 (2025-01-24)

- Generate package documentation with `sphinx` and upload it to GitLab Pages for each released version of the package.

## Version 0.14 (2025-01-08)

- Replace local copy of `mapomatic` code with `iqm-mapomatic` package.

## Version 0.13 (2025-01-07)

- Fix estimator based on QUIMB, adding a warning.

## Version 0.12 (2024-12-16)

- Add a method `circuit` to the QUBOQAOA class, which builds the circuit and transpiles it to the HW.
- Implement the "hardwired" transpilation strategy.
- Implement the "sparse"/greedy/Ayse-Martin-Fedor transpilation strategy.
- Implement the swap network transpilation strategy.

## Version 0.11 (2024-11-22)

- Change the implementation of Goemans-Williamson algorithm to improve performance.
- Replace the structure of the problem instance classes to only store the BinaryQuadraticModel representation of the problem and calculate the other representations lazily.

## Version 0.10 (2024-11-11)

- Add TreeQAOA class with tree angle setting scheme.

## Version 0.9 (2024-11-05)

- Make classical solvers accept either a nx.Graph or a problem instance.
- Add tests for classical algorithms for maximum independent set and for constraints checker.

## Version 0.8 (2024-10-30)

- Refine problem classes, removing duplicate methods.

## Version 0.7 (2024-10-23)

- Add first batch of unit tests.

## Version 0.6 (2024-10-21)

- Update build tools to latest available versions.

## Version 0.5 (2024-10-16)

- Add license file.

## Version 0.4 (2024-10-16)

- Downgrade build tools to known working versions.

## Version 0.3 (2024-10-16)

- Update `setuptools_scm` configuration to fix package version string generation.

## Version 0.2 (2024-10-15)

- Fix release process

## Version 0.1 (2024-10-15)

- First public-ish release
