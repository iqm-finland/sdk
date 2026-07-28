.. _qrisp_iqm_transpilation:

Transpilation
=============

Plasma-SABRE (**P**\ermeabiLity **A**\ssisted **S**\ectionalized **M**\ovement **A**\cceleration) is a 
high-performance version of the established `SABRE`_ algorithm
(Li et al., ASPLOS 2019), purpose-built for Qrisp and IQM hardware.  It
outperforms existing approaches through several novel extensions:

* **PermeabilityGraph (PDAG)** — leverages Qrisp's commutation-aware DAG
  so the router can reorder commuting gates to avoid unnecessary SWAPs.
  To learn more about the Permeability Graph, please check out `this
  work`_.
* **Sectionalized SABRE** — splits the circuit at Qrisp's permeability
  terminator nodes and compiles each section independently with parallel
  random-seed trials.  Early mistakes are quarantined within their section
  rather than cascading through the whole circuit.
* **Parallel tempering** — distributes different greediness levels across
  parallel routing threads, analogous to simulated annealing at different
  temperatures.
* **Modified SABRE cost function** — relevance-decay-weighted extended set,
  descendant-count priority, and a congestion penalty that spreads work
  across idle qubits.
* **Numba-JIT core** — compiled to native code with ``int16`` index types
  throughout, packing more data into each cache line and reducing memory
  bandwidth pressure in the hot SABRE metric loop.

You control two key trade-offs through simple parameters on every pass:

* ``effort`` — higher values explore more layout candidates and routing
  trials per section, yielding better results at the cost of longer compilation time.
* ``depth_weight`` — ``-1`` minimizes gate count, ``0`` balances gate count
  and depth, ``+1`` minimizes circuit depth.

.. _SABRE: https://dl.acm.org/doi/10.1145/3297858.3304022
.. _this work: https://arxiv.org/abs/2606.31837

PassManager Infrastructure
--------------------------

All transpilation passes are designed for Qrisp's :class:`~qrisp.PassManager` class,
a composable pipeline where passes are added with the ``+=`` operator and
applied in sequence via ``pm.run(qc)``.  You can mix and match them
freely — combine plasma-sabre routing with your own custom decomposition
passes, or insert standard Qrisp optimizations before or after
IQM-specific conversion:

.. code-block:: python

    from qrisp import PassManager, convert_to_cz, combine_single_qubit_gates
    from iqm.qrisp_iqm import plasma_layout, plasma_route, transpile_to_iqm

    # One-liner: full plasma-sabre pipeline
    iqm_ready = transpile_to_iqm(qc, connectivity=topology)

    # Or build your own pipeline
    pm = PassManager()
    pm += combine_single_qubit_gates   # Qrisp built-in
    pm += plasma_layout(topology, effort=40, depth_weight=-0.5)
    pm += plasma_route(topology, effort=40, depth_weight=-0.5)
    pm += convert_to_cz()
    pm += convert_to_prx
    transpiled = pm.run(qc)

Default Pipeline
----------------

.. currentmodule:: iqm.qrisp_iqm.passes

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-short-name.rst
   :nosignatures:

   create_iqm_pass_manager
   transpile_to_iqm


Layout & Routing
----------------

.. currentmodule:: iqm.qrisp_iqm.passes.routing

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-short-name.rst
   :nosignatures:

   plasma_layout
   plasma_route
   vf2pp_layout


Optimization
------------

.. currentmodule:: iqm.qrisp_iqm.passes

.. autosummary::
   :toctree: api/qrisp_iqm
   :template: autosummary-short-name.rst
   :nosignatures:

   commute_phases
   measurement_parallelization
