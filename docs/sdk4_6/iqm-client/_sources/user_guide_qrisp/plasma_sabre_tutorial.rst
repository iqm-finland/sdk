.. _plasma_sabre_tutorial:

Plasma-Sabre Tutorial
=====================

This tutorial gives a practical overview of the **plasma-sabre** transpiler for Qrisp,
including user-facing passes and the IQM execution flow.

What you will learn
-------------------

- How to build and inspect a Qrisp circuit
- How to compose transpilation pipelines with ``PassManager``
- How to use layout/routing passes:

  - ``plasma_layout``
  - ``plasma_route``
  - ``vf2pp_layout``
  - ``manual_layout``

- How to use gate conversion passes:

  - ``convert_to_cz``
  - ``convert_to_prx``

- How to run full IQM transpilation via ``transpile_to_iqm``
- How to submit a transpiled circuit to an IQM backend


Setup
-----

.. code-block:: python

    # Core imports
    from qrisp import QuantumCircuit, Qubit, Clbit, PassManager, convert_to_cz, convert_to_prx, manual_layout

    # IQM imports
    from iqm.qrisp_iqm import plasma_layout, plasma_route, vf2pp_layout, transpile_to_iqm

    print("Imports successful.")

Output:

.. code-block:: text

    WARNING:jax._src.xla_bridge:864: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
    Imports successful.


Demo circuit
------------

We define a small demo circuit with various two-qubit interactions to exercise
the transpiler:

.. code-block:: python

    def build_demo_circuit() -> QuantumCircuit:
        """Build a small demo circuit with 2-qubit interactions."""
        qc = QuantumCircuit()

        # Give logical qubits distinctive names
        for i in range(4):
            qc.add_qubit(Qubit("original_qb_" + str(i)))

        for i in range(4):
            qc.add_clbit(Clbit("c" + str(i)))

        qc.h(0)
        qc.cx(0, 1)
        qc.ry(0.7, 2)
        qc.cz(1, 2)
        qc.cx(2, 3)
        qc.s(1)
        qc.cy(0, 2)

        # Add measurements for execution workflows
        qc.measure(qc.qubits, qc.clbits)
        return qc

    qc = build_demo_circuit()
    print(qc)

Output:

.. code-block:: text

                      ┌───┐                        ┌─┐
    original_qb_0: ───┤ H ├─────■────────────■─────┤M├───
                      └───┘   ┌─┴─┐   ┌───┐  │  ┌─┐└╥┘
    original_qb_1: ───────────┤ X ├─■─┤ S ├──┼──┤M├─╫────
                   ┌─────────┐└───┘ │ └───┘┌─┴─┐└╥┘ ║ ┌─┐
    original_qb_2: ┤ Ry(0.7) ├──────■───■──┤ Y ├─╫──╫─┤M├
                   └─────────┘        ┌─┴─┐└┬─┬┘ ║  ║ └╥┘
    original_qb_3: ───────────────────┤ X ├─┤M├──╫──╫──╫─
                                      └───┘ └╥┘  ║  ║  ║
               c0: ══════════════════════════╬═══╬══╩══╬═
                                             ║   ║     ║
               c1: ══════════════════════════╬═══╩═════╬═
                                             ║         ║
               c2: ══════════════════════════╬═════════╩═
                                             ║
               c3: ══════════════════════════╩═══════════


1. ``plasma_layout`` — Finding the best qubit permutation
---------------------------------------------------------

``plasma_layout`` solves the **initial placement problem**: given a quantum circuit with *logical*
qubits and a hardware topology with *physical* qubits connected by edges, find a mapping
(permutation) of logical → physical qubits that minimizes the routing cost downstream.

What it does concretely
~~~~~~~~~~~~~~~~~~~~~~~

The pass **permutes the circuit's qubits** — it re-labels which physical qubit each logical
qubit sits on.  No SWAP gates are inserted; the gate sequence stays the same, only the qubit
indices change.  For instance, if the original circuit applies ``cx(0, 3)`` and ``plasma_layout``
decides logical qubit 0 should live on physical qubit 2 and logical qubit 3 on physical qubit 1,
the output circuit will contain ``cx(2, 1)`` (and will have as many qubits as the hardware
topology, not just the circuit).

Example topology — a "cut ring"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Throughout this tutorial we use 6 physical qubits connected in a ring **with the edge (0, 1)
removed**::

    1 — 2 — 3 — 4 — 5 — 0

This is essentially a chain ``1-2-3-4-5-0``.  Because one edge of the ring is missing,
the layout pass actually has to work: naïvely mapping logical qubit 0 to physical qubit 0 and
logical qubit 1 to physical qubit 1 would place them on opposite ends of the chain, requiring
many SWAPs.  ``plasma_layout`` finds a better permutation.

Relevant parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``connectivity``
     - *(required)*
     - List of ``(u, v)`` edges describing hardware connectivity
   * - ``effort``
     - ``30``
     - More effort → more random candidates and refinement iterations → better layouts but slower compile
   * - ``depth_weight``
     - ``0.0``
     - ``-1`` = optimise for gate count only, ``0`` = balanced, ``+1`` = optimise for depth only

VF2++ fast path
~~~~~~~~~~~~~~~

Before starting the stochastic search, ``plasma_layout`` first tries a **VF2++ subgraph
isomorphism** check.  If the circuit's qubit-interaction graph is already a subgraph of the
hardware topology, the circuit can be mapped with zero routing cost and VF2++ returns
immediately.  The stochastic search is only triggered when VF2++ fails — which is the case for
our demo circuit, since the ``cy(0, 2)`` gate creates an interaction between qubits 0 and 2 that
doesn't correspond to any single edge in our cut-ring topology.

.. code-block:: python

    # Demonstrate plasma_layout alone: only qubit permutation, no SWAPs
    connectivity = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)
    ]

    pm_layout_only = PassManager()
    pm_layout_only += plasma_layout(connectivity, effort=40)

    original = build_demo_circuit()
    laid_out = pm_layout_only.run(original)

    print("Original circuit (4 qubits):")
    print(original)
    print(f"\nAfter plasma_layout (now {laid_out.num_qubits()} physical qubits, same gates, permuted indices):")
    print(laid_out)

Output:

.. code-block:: text

    Original circuit (4 qubits):
                      ┌───┐                        ┌─┐
    original_qb_0: ───┤ H ├─────■────────────■─────┤M├───
                      └───┘   ┌─┴─┐   ┌───┐  │  ┌─┐└╥┘
    original_qb_1: ───────────┤ X ├─■─┤ S ├──┼──┤M├─╫────
                   ┌─────────┐└───┘ │ └───┘┌─┴─┐└╥┘ ║ ┌─┐
    original_qb_2: ┤ Ry(0.7) ├──────■───■──┤ Y ├─╫──╫─┤M├
                   └─────────┘        ┌─┴─┐└┬─┬┘ ║  ║ └╥┘
    original_qb_3: ───────────────────┤ X ├─┤M├──╫──╫──╫─
                                      └───┘ └╥┘  ║  ║  ║
               c0: ══════════════════════════╬═══╬══╩══╬═
                                             ║   ║     ║
               c1: ══════════════════════════╬═══╩═════╬═
                                             ║         ║
               c2: ══════════════════════════╬═════════╩═
                                             ║
               c3: ══════════════════════════╩═══════════

    After plasma_layout (now 6 physical qubits, same gates, permuted indices):

     amended_qb_0: ──────────────────────────────────────

     amended_qb_1: ──────────────────────────────────────
                      ┌───┐                        ┌─┐
    original_qb_0: ───┤ H ├─────■────────────■─────┤M├───
                      └───┘   ┌─┴─┐   ┌───┐  │  ┌─┐└╥┘
    original_qb_1: ───────────┤ X ├─■─┤ S ├──┼──┤M├─╫────
                   ┌─────────┐└───┘ │ └───┘┌─┴─┐└╥┘ ║ ┌─┐
    original_qb_2: ┤ Ry(0.7) ├──────■───■──┤ Y ├─╫──╫─┤M├
                   └─────────┘        ┌─┴─┐└┬─┬┘ ║  ║ └╥┘
    original_qb_3: ───────────────────┤ X ├─┤M├──╫──╫──╫─
                                      └───┘ └╥┘  ║  ║  ║
               c0: ══════════════════════════╬═══╬══╩══╬═
                                             ║   ║     ║
               c1: ══════════════════════════╬═══╩═════╬═
                                             ║         ║
               c2: ══════════════════════════╬═════════╩═
                                             ║
               c3: ══════════════════════════╩═══════════


2. ``plasma_route`` — SWAP insertion
------------------------------------

``plasma_route`` takes a circuit that already has a **fixed layout** (e.g. from ``plasma_layout``)
and inserts SWAP gates so that every 2-qubit gate acts on physically adjacent qubits.

Relevant parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``connectivity``
     - *(required)*
     - Hardware topology edges
   * - ``effort``
     - ``30``
     - More effort → better results, slower compile
   * - ``depth_weight``
     - ``0.0``
     - ``-1`` = gate count, ``0`` = balanced, ``+1`` = depth

Combined pipeline: ``plasma_layout`` → ``plasma_route``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice you always chain the two passes.  Make sure ``depth_weight`` matches between them.

.. code-block:: python

    # Example hardware topology (6 physical qubits):
    connectivity = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)
    ]

    pm = PassManager()
    pm += plasma_layout(connectivity, effort=40)
    pm += plasma_route(connectivity, effort=40)

    routed_qc = pm.run(build_demo_circuit())
    print(routed_qc)

Output:

.. code-block:: text

     amended_qb_0: ─────────────────────────────────────────

     amended_qb_1: ─────────────────────────────────────────
                      ┌───┐                              ┌─┐
    original_qb_0: ───┤ H ├─────■──────────────────■─────┤M├
                      └───┘   ┌─┴─┐   ┌───┐┌─┐   ┌─┴─┐┌─┐└╥┘
    original_qb_1: ───────────┤ X ├─■─┤ S ├┤M├─X─┤ Y ├┤M├─╫─
                   ┌─────────┐└───┘ │ └───┘└╥┘ │ └───┘└╥┘ ║
    original_qb_2: ┤ Ry(0.7) ├──■───■───────╫──X───────╫──╫─
                   └─────────┘┌─┴─┐┌─┐      ║          ║  ║
    original_qb_3: ───────────┤ X ├┤M├──────╫──────────╫──╫─
                              └───┘└╥┘      ║          ║  ║
               c0: ═════════════════╬═══════╬══════════╬══╩═
                                    ║       ║          ║
               c1: ═════════════════╬═══════╩══════════╬════
                                    ║                  ║
               c2: ═════════════════╬══════════════════╩════
                                    ║
               c3: ═════════════════╩═══════════════════════


3. ``vf2pp_layout``
-------------------

``vf2pp_layout`` attempts to embed the circuit's interaction graph directly into the topology
via **VF2++ subgraph isomorphism**.
If successful, the circuit can run **without any routing SWAPs** at all.

Use this when you suspect your circuit connectivity already fits the device graph.
Note: ``plasma_layout`` already tries VF2++ internally as a fast path — this standalone pass
is useful when you want to *only* attempt VF2++. A crucial difference to calling VF2++ from
``plasma_layout`` is that ``vf2pp_layout`` will raise an Exception if there is no perfect layout.
``plasma_layout`` will simply proceed with heuristic layout selection.

When does it fail?
~~~~~~~~~~~~~~~~~~

Our demo circuit has the interaction ``cy(0, 2)``, i.e. logical qubits 0 and 2 talk to each
other although they are not neighbours on the cut-ring chain ``1-2-3-4-5-0``.  No relabelling
can fix this because the interaction graph contains a "triangle-like" structure that doesn't
fit into a path — so VF2++ raises an error.

When does it succeed?
~~~~~~~~~~~~~~~~~~~~~

A circuit whose interactions already form a **path** (or any subgraph of the topology)
will succeed.  Below we show both cases.

.. code-block:: python

    pm_vf2 = PassManager()
    pm_vf2 += vf2pp_layout(connectivity)

    # --- Case 1: demo circuit (fails — interaction graph doesn't fit the chain) ---
    try:
        vf2_qc = pm_vf2.run(build_demo_circuit())
        print("VF2++ layout succeeded on demo circuit:")
        print(vf2_qc)
    except ValueError as err:
        print("VF2++ layout failed on demo circuit (expected — cy(0,2) is non-adjacent):")
        print(err)

    # --- Case 2: a circuit with only path-like interactions (succeeds) ---
    print("\n--- Circuit with path interactions ---")
    path_qc = QuantumCircuit()
    # Give logical qubits distinctive names
    for i in range(4):
        path_qc.add_qubit(Qubit("original_qb_" + str(i)))
    for i in range(4):
        path_qc.add_clbit(Clbit("c_" + str(i)))
    path_qc.h(0)
    path_qc.cx(0, 1)       # 0-1
    path_qc.cx(1, 2)       # 1-2
    path_qc.cx(2, 3)       # 2-3
    path_qc.measure(path_qc.qubits, path_qc.clbits)

    print("Path circuit (interactions: 0-1, 1-2, 2-3):")
    print(path_qc)

    vf2_path_qc = pm_vf2.run(path_qc)
    print(f"VF2++ succeeded — mapped to {vf2_path_qc.num_qubits()} physical qubits, zero SWAPs needed:")
    print(vf2_path_qc)

Output:

.. code-block:: text

    VF2++ layout failed on demo circuit (expected — cy(0,2) is non-adjacent):
    VF2++ could not find a matching qubit set for the circuit. The circuit's connectivity graph is not a subgraph of the topology. Consider using 'plasma_layout' and 'plasma_route' for circuits requiring swap insertion.

    --- Circuit with path interactions ---
    Path circuit (interactions: 0-1, 1-2, 2-3):
                   ┌───┐          ┌─┐
    original_qb_0: ┤ H ├──■───────┤M├──────────────
                   └───┘┌─┴─┐     └╥┘     ┌─┐
    original_qb_1: ─────┤ X ├──■───╫──────┤M├──────
                        └───┘┌─┴─┐ ║      └╥┘┌─┐
    original_qb_2: ──────────┤ X ├─╫───■───╫─┤M├───
                             └───┘ ║ ┌─┴─┐ ║ └╥┘┌─┐
    original_qb_3: ────────────────╫─┤ X ├─╫──╫─┤M├
                                   ║ └───┘ ║  ║ └╥┘
              c_0: ════════════════╩═══════╬══╬══╬═
                                           ║  ║  ║
              c_1: ════════════════════════╩══╬══╬═
                                              ║  ║
              c_2: ═══════════════════════════╩══╬═
                                                 ║
              c_3: ══════════════════════════════╩═

    VF2++ succeeded — mapped to 6 physical qubits, zero SWAPs needed:

     amended_qb_0: ────────────────────────────────
                   ┌───┐          ┌─┐
    original_qb_0: ┤ H ├──■───────┤M├──────────────
                   └───┘┌─┴─┐     └╥┘     ┌─┐
    original_qb_1: ─────┤ X ├──■───╫──────┤M├──────
                        └───┘┌─┴─┐ ║      └╥┘┌─┐
    original_qb_2: ──────────┤ X ├─╫───■───╫─┤M├───
                             └───┘ ║ ┌─┴─┐ ║ └╥┘┌─┐
    original_qb_3: ────────────────╫─┤ X ├─╫──╫─┤M├
                                   ║ └───┘ ║  ║ └╥┘
     amended_qb_1: ────────────────╫───────╫──╫──╫─
                                   ║       ║  ║  ║
              c_0: ════════════════╩═══════╬══╬══╬═
                                           ║  ║  ║
              c_1: ════════════════════════╩══╬══╬═
                                              ║  ║
              c_2: ═══════════════════════════╩══╬═
                                                 ║
              c_3: ══════════════════════════════╩═


4. ``manual_layout``
--------------------

``manual_layout`` lets you choose physical qubits explicitly.

- Input: ``qubit_mapping``, where logical qubit ``i`` maps to physical ``qubit_mapping[i]``
- Mapping must be the same length as circuit qubits, with unique non-negative indices

This is useful when you want deterministic placement (e.g. due to calibration data).

.. code-block:: python

    # Map 4 logical qubits -> physical qubits [1, 2, 4, 5]
    manual_map = [1, 2, 4, 5]

    pm_manual = PassManager()
    pm_manual += manual_layout(manual_map)

    manual_qc = pm_manual.run(build_demo_circuit())
    print(manual_qc)

Output:

.. code-block:: text

     amended_qb_0: ──────────────────────────────────────
                      ┌───┐                        ┌─┐
    original_qb_0: ───┤ H ├─────■────────────■─────┤M├───
                      └───┘   ┌─┴─┐   ┌───┐  │  ┌─┐└╥┘
    original_qb_1: ───────────┤ X ├─■─┤ S ├──┼──┤M├─╫────
                              └───┘ │ └───┘  │  └╥┘ ║
     amended_qb_1: ─────────────────┼────────┼───╫──╫────
                   ┌─────────┐      │      ┌─┴─┐ ║  ║ ┌─┐
    original_qb_2: ┤ Ry(0.7) ├──────■───■──┤ Y ├─╫──╫─┤M├
                   └─────────┘        ┌─┴─┐└┬─┬┘ ║  ║ └╥┘
    original_qb_3: ───────────────────┤ X ├─┤M├──╫──╫──╫─
                                      └───┘ └╥┘  ║  ║  ║
               c0: ══════════════════════════╬═══╬══╩══╬═
                                             ║   ║     ║
               c1: ══════════════════════════╬═══╩═════╬═
                                             ║         ║
               c2: ══════════════════════════╬═════════╩═
                                             ║
               c3: ══════════════════════════╩═══════════


5. Gate conversion passes
-------------------------

``convert_to_cz``
~~~~~~~~~~~~~~~~~

Converts 2-qubit gates (such as ``cx``, ``cy``, ``swap``) into CZ-based forms.

``convert_to_prx``
~~~~~~~~~~~~~~~~~~

Converts single-qubit operations into PRX-style decomposition used in IQM-related flows.

These are typically used near the end of a transpilation pipeline.

.. code-block:: python

    pm_convert = PassManager()
    pm_convert += convert_to_cz()
    pm_convert += convert_to_prx

    converted_qc = pm_convert.run(build_demo_circuit())
    print(converted_qc)

Output:

.. code-block:: text

                   ┌──────────────┐┌────────┐                                »
    original_qb_0: ┤ R(3π/2,-π/2) ├┤ R(π,0) ├─■──────────────────────────────»
                   ├──────────────┤├────────┤ │ ┌──────────────┐┌────────┐   »
    original_qb_1: ┤ R(3π/2,-π/2) ├┤ R(π,0) ├─■─┤ R(3π/2,-π/2) ├┤ R(π,0) ├─■─»
                   └┬────────────┬┘└────────┘   └──────────────┘└────────┘ │ »
    original_qb_2: ─┤ R(0.7,π/2) ├─────────────────────────────────────────■─»
                   ┌┴────────────┴┐┌────────┐                                »
    original_qb_3: ┤ R(3π/2,-π/2) ├┤ R(π,0) ├────────────────────────────────»
                   └──────────────┘└────────┘                                »
               c0: ══════════════════════════════════════════════════════════»
                                                                             »
               c1: ══════════════════════════════════════════════════════════»
                                                                             »
               c2: ══════════════════════════════════════════════════════════»
                                                                             »
               c3: ══════════════════════════════════════════════════════════»
                                                                             »
    «                                                                        »
    «original_qb_0: ─────────────────────────────────────────────────────────»
    «               ┌────────┐  ┌──────────┐              ┌─┐                »
    «original_qb_1: ┤ R(π,0) ├──┤ R(π,π/4) ├──────────────┤M├────────────────»
    «               └────────┘  ├──────────┤  ┌──────────┐└╥┘┌──────────────┐»
    «original_qb_2: ────■───────┤ R(π,π/2) ├──┤ R(π,π/4) ├─╫─┤ R(3π/2,-π/2) ├»
    «                   │     ┌─┴──────────┴─┐└┬────────┬┘ ║ └─────┬─┬──────┘»
    «original_qb_3: ────■─────┤ R(3π/2,-π/2) ├─┤ R(π,0) ├──╫───────┤M├───────»
    «                         └──────────────┘ └────────┘  ║       └╥┘       »
    «           c0: ═══════════════════════════════════════╬════════╬════════»
    «                                                      ║        ║        »
    «           c1: ═══════════════════════════════════════╩════════╬════════»
    «                                                               ║        »
    «           c2: ════════════════════════════════════════════════╬════════»
    «                                                               ║        »
    «           c3: ════════════════════════════════════════════════╩════════»
    «                                                                        »
    «                                            ┌─┐┌────────┐          »
    «original_qb_0: ───────────■─────────────────┤M├┤ gphase ├──────────»
    «                          │                 └╥┘└────────┘          »
    «original_qb_1: ───────────┼──────────────────╫─────────────────────»
    «               ┌────────┐ │ ┌──────────────┐ ║ ┌────────┐┌────────┐»
    «original_qb_2: ┤ R(π,0) ├─■─┤ R(3π/2,-π/2) ├─╫─┤ R(π,0) ├┤ R(π,0) ├»
    «               └────────┘   └──────────────┘ ║ └────────┘└────────┘»
    «original_qb_3: ──────────────────────────────╫─────────────────────»
    «                                             ║                     »
    «           c0: ══════════════════════════════╩═════════════════════»
    «                                                                   »
    «           c1: ════════════════════════════════════════════════════»
    «                                                                   »
    «           c2: ════════════════════════════════════════════════════»
    «                                                                   »
    «           c3: ════════════════════════════════════════════════════»
    «                                                                   »
    «                              
    «original_qb_0: ───────────────
    «                              
    «original_qb_1: ───────────────
    «               ┌──────────┐┌─┐
    «original_qb_2: ┤ R(π,π/4) ├┤M├
    «               └──────────┘└╥┘
    «original_qb_3: ─────────────╫─
    «                            ║ 
    «           c0: ═════════════╬═
    «                            ║ 
    «           c1: ═════════════╬═
    «                            ║ 
    «           c2: ═════════════╩═
    «                              
    «           c3: ═══════════════
    «


6. Tuning ``effort`` and ``depth_weight``
-----------------------------------------

Both ``plasma_layout`` and ``plasma_route`` accept two knobs that let you trade off compilation
time against result quality, and gate count against circuit depth:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Effect
   * - ``effort``
     - Controls how many random seeds / candidates the router explores. Higher values find better solutions but take longer.
   * - ``depth_weight``
     - Steers the optimization target on a scale from **−1** (minimize gate/SWAP count) through **0** (balanced) to **+1** (minimize depth).

To see these in action, we compile a non-trivial circuit — a **5-bit quantum adder** — onto a
4 × 4 square-grid topology and compare three settings.

The ``decompose`` pass
~~~~~~~~~~~~~~~~~~~~~~

Before layout and routing, the compiled adder circuit contains multi-controlled gates (> 2
qubits).  These cannot be placed directly onto hardware.  The ``decompose`` pass recursively
breaks down any gate satisfying a predicate (here: more than 2 qubits) into 1- and 2-qubit
gates.  We include it in the pipeline so the router only sees hardware-compatible operations.

.. code-block:: python

    from qrisp import QuantumFloat, decompose

    # Build a 5-bit quantum adder circuit
    a = QuantumFloat(5)
    b = QuantumFloat(5)
    a += b
    qc = a.qs.compile()

    # 4x4 square grid coupling map
    N = 4
    connectivity = []
    for i in range(N**2):
        if i % N:
            connectivity.append((i, i - 1))
        if i > N:
            connectivity.append((i, i - N))

    def route_with_settings(depth_weight, effort):
        pm = PassManager()
        pm += decompose(decompose_predicate = lambda op: op.num_qubits > 2)
        pm += plasma_layout(connectivity=connectivity, depth_weight=depth_weight, effort=effort)
        pm += plasma_route(connectivity=connectivity, depth_weight=depth_weight, effort=effort)
        compiled = pm.run(qc)
        return compiled

    # --- Run 1: depth_weight = -1  (minimize gate count) ---
    qc_gateopt = route_with_settings(depth_weight=-1, effort=10)
    print("depth_weight = -1, effort = 10  (minimize gate count)")
    print(f"  CNOT depth : {qc_gateopt.cnot_depth()}")
    print(f"  Gate counts: {qc_gateopt.count_ops()}")

    # --- Run 2: depth_weight = +1  (minimize depth) ---
    qc_depthopt = route_with_settings(depth_weight=1, effort=10)
    print("\ndepth_weight = +1, effort = 10  (minimize depth)")
    print(f"  CNOT depth : {qc_depthopt.cnot_depth()}")
    print(f"  Gate counts: {qc_depthopt.count_ops()}")

    # --- Run 3: depth_weight = +1, higher effort ---
    qc_depthopt_hi = route_with_settings(depth_weight=1, effort=1000)
    print("\ndepth_weight = +1, effort = 1000 (minimize depth, try harder)")
    print(f"  CNOT depth : {qc_depthopt_hi.cnot_depth()}")
    print(f"  Gate counts: {qc_depthopt_hi.count_ops()}")

Output:

.. code-block:: text

    depth_weight = -1, effort = 10  (minimize gate count)
      CNOT depth : 55
      Gate counts: {'h': 10, 'cx': 50, 'p': 53, 'swap': 9}

    depth_weight = +1, effort = 10  (minimize depth)
      CNOT depth : 52
      Gate counts: {'h': 10, 'cx': 50, 'p': 53, 'swap': 10}

    depth_weight = +1, effort = 1000 (minimize depth, try harder)
      CNOT depth : 46
      Gate counts: {'h': 10, 'cx': 50, 'p': 53, 'swap': 11}


What to observe
~~~~~~~~~~~~~~~

- **``depth_weight = -1``** produces the fewest SWAP gates (lowest total gate count) but the
  deepest circuit — the router packs qubits tightly even if that serialises operations.
- **``depth_weight = +1``** trades extra SWAPs for a shallower circuit: the CNOT depth drops,
  while the SWAP / gate count increases.
- **Raising ``effort`` to 1000** at ``depth_weight = +1`` pushes the depth even lower — at the
  cost of yet more SWAPs and longer compilation time.  The router explores more candidates and
  finds increasingly aggressive parallelisation strategies.

In general, use ``depth_weight = -1`` when gate count (and thus error rate) matters most, and
``depth_weight = +1`` when circuit *duration* (i.e. thermal decay) on hardware is the bottleneck.
``effort`` controls how long you're willing to wait for a better solution.


7. End-to-end IQM transpilation: ``transpile_to_iqm``
-----------------------------------------------------

``transpile_to_iqm`` wraps the full plasma-sabre pipeline — layout, routing, gate conversion,
and several additional optimization passes — into a single call.  Because it bundles these
strong optimizations together, **it should be considered the default function for
production-level transpilation** when targeting IQM hardware.

.. note::

   The public function name is ``transpile_to_iqm`` (lowercase).

.. code-block:: python

    # Local transpilation example against a known coupling map
    iqm_ready_qc = transpile_to_iqm(build_demo_circuit(), connectivity=connectivity)
    print(iqm_ready_qc)

Output:

.. code-block:: text

    amended_qb_0: ─────────────────────────────────────────────────────────────»
                                                                                »
    amended_qb_10: ─────────────────────────────────────────────────────────────»
                   ┌─────────────┐   ┌────────────┐      ┌─┐                    »
    original_qb_1: ┤ R(π/2,-π/2) ├─■─┤ R(π/2,π/2) ├─■────┤M├────────────────────»
                   ├─────────────┤ │ └────────────┘ │    └╥┘   ┌────────────┐   »
    original_qb_0: ┤ R(π/2,-π/2) ├─■────────────────┼─────╫──■─┤ R(π/2,π/2) ├─■─»
                   └─────────────┘                  │     ║  │ └────────────┘ │ »
    amended_qb_1:  ─────────────────────────────────┼─────╫──┼────────────────┼─»
                                                    │     ║  │                │ »
    amended_qb_3:  ─────────────────────────────────┼─────╫──┼────────────────┼─»
                    ┌────────────┐                  │     ║  │  ┌──────────┐  │ »
    original_qb_2: ─┤ R(0.7,π/2) ├──────────────────■──■──╫──┼──┤ R(π/2,0) ├──┼─»
                   ┌┴────────────┤                     │  ║  │ ┌┴──────────┴┐ │ »
    original_qb_3: ┤ R(π/2,-π/2) ├─────────────────────■──╫──■─┤ R(π/2,π/2) ├─■─»
                   └─────────────┘                        ║    └────────────┘   »
    amended_qb_9:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_7:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_6:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_2:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_5:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_8:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_4:  ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
    amended_qb_11: ───────────────────────────────────────╫─────────────────────»
                                                          ║                     »
            c0:    ═══════════════════════════════════════╬═════════════════════»
                                                          ║                     »
            c1:    ═══════════════════════════════════════╩═════════════════════»
                                                                                »
            c2:    ═════════════════════════════════════════════════════════════»
                                                                                »
            c3:    ═════════════════════════════════════════════════════════════»
                                                                                »
    «                                                                     
    « amended_qb_0: ──────────────────────────────────────────────────────
    «                                                                     
    «amended_qb_10: ──────────────────────────────────────────────────────
    «                                                                     
    «original_qb_1: ──────────────────────────────────────────────────────
    «               ┌─────────────┐                 ┌─┐                   
    «original_qb_0: ┤ R(π/2,-π/2) ├─■───────────────┤M├───────────────────
    «               └─────────────┘ │               └╥┘                   
    « amended_qb_1: ────────────────┼────────────────╫────────────────────
    «                               │                ║                    
    « amended_qb_3: ────────────────┼────────────────╫────────────────────
    «                               │                ║    ┌───────────┐┌─┐
    «original_qb_2: ────────────────┼────────────────╫──■─┤ R(π/2,-π) ├┤M├
    «               ┌─────────────┐ │ ┌────────────┐ ║  │ └────┬─┬────┘└╥┘
    «original_qb_3: ┤ R(π/2,-π/2) ├─■─┤ R(π/2,π/2) ├─╫──■──────┤M├──────╫─
    «               └─────────────┘   └────────────┘ ║         └╥┘      ║ 
    « amended_qb_9: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    « amended_qb_7: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    « amended_qb_6: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    « amended_qb_2: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    « amended_qb_5: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    « amended_qb_8: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    « amended_qb_4: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    «amended_qb_11: ─────────────────────────────────╫──────────╫───────╫─
    «                                                ║          ║       ║ 
    «           c0: ═════════════════════════════════╬══════════╩═══════╬═
    «                                                ║                  ║ 
    «           c1: ═════════════════════════════════╬══════════════════╬═
    «                                                ║                  ║ 
    «           c2: ═════════════════════════════════╬══════════════════╩═
    «                                                ║                    
    «           c3: ═════════════════════════════════╩════════════════════
    «


8. Actual IQM backend call (with token placeholder)
---------------------------------------------------

This section demonstrates real hardware submission.

1. Install IQM client package (``iqm-client``) if needed
2. Set your token and server URL
3. Fetch architecture, transpile, submit

If ``token`` is left as placeholder, the code prints instructions and skips submission.

.. code-block:: python

    from iqm.qrisp_iqm import IQMBackend, create_iqm_pass_manager

    token = "YOUR_TOKEN_HERE"
    server_url = "https://resonance.iqm.tech"

    if token == "YOUR_TOKEN_HERE":
        print("Set your real IQM token in `token` to run this cell.")
    else:
        garnet = IQMBackend(device_instance="garnet", # Select garnet
                            server_url=server_url,
                            token=token, # Authenticate
                            pass_manager = PassManager()) # Create an empty pass manager to ensure circuits are passed to the backend as is

        iqm_connectivity = garnet.connectivity

        garnet.pm += create_iqm_pass_manager(connectivity=iqm_connectivity,
                                             effort=100,
                                             depth_weight=0.0,)

        result_counts = garnet.run(build_demo_circuit(), shots=100)

        print("IQM result counts:")
        print(result_counts)

Output (with a valid token):

.. code-block:: text

    IQM result counts:
    {'0000': 41, '0011': 5, '0100': 2, '0110': 2, '0111': 2, '1001': 2, '1010': 1, '1011': 1, '1100': 2, '1101': 7, '1110': 35}
