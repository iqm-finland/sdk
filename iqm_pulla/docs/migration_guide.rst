Migration guide
###############

This document describes the changes that need to be made to existing code to migrate between major versions of Pulla.

From 3.x to 4.0
---------------

The changes in 4.0 are not truly breaking, as your code should still run. However, there are some changes that
might affect the compilation results, and you might want to adjust your code accordingly:

- New compilation pass ``prepend_reset`` added to the (TimeBox-level) standard compiler stage. It adds a reset timebox
  to all circuits. It is the last pass of the TimeBox-level stage. If the calibration lacks `reset_wait` data, the
  ``prepend_reset`` will have no effect.
- Optional attribute :attr:``.CircuitExecutionOptions.active_reset_cycles`` added to `CircuitExecutionOptions`,
  that is used to control the reset functionality. By default, it is set to ``None``, which results in delay by wait.
- :meth`.Pulla.get_standard_compiler` now has an optional argument for overriding default circuit execution options.

From 2.x to 3.0
---------------

The function ``iqm.pulla.utils_qiskit.qiskit_to_cpc`` was replaced by
:func:`.iqm.pulla.utils_qiskit.qiskit_circuits_to_pulla` and :func:`.iqm.pulla.utils_qiskit.qiskit_to_pulla`.

:func:`.qiskit_circuits_to_pulla` is a more direct replacement, with the difference that it also requires a
mapping of qiskit qubit indices to physical qubit names which was not required before.
This can be e.g. obtained from an :class:`iqm.qiskit_iqm.iqm_provider.IQMBackend` instance.

:func:`.qiskit_to_pulla` is a convenience method that in addition to
converting the Qiskit circuit(s) also returns a :class:`.Compiler` instance that can be used to compile them
into a playlist. It takes as additional parameters a :class:`.Pulla` instance (for building the compiler),
and an :class:`~iqm.qiskit_iqm.iqm_provider.IQMBackend` instance (containing the calibration set ID and
qubit mapping to use). The IQMBackend instance should usually be the same one that was used to transpile
the Qiskit circuits.

.. code-block:: python

    # BEFORE
    pulla_circuits = qiskit_to_cpc(qiskit_circuits)

    # AFTER
    pulla_circuits = qiskit_circuits_to_pulla(qiskit_circuits, qubit_idx_to_name)
    # or
    pulla_circuits, compiler = qiskit_to_pulla(pulla, backend, qiskit_circuits)

From 1.x to 2.0
---------------

Compiler code is consolidated under the :mod:`iqm.cpc.compiler.compiler` module.
There is no more :mod:`iqm.cpc.compiler.compiler2`.
:class:`.Compiler`, :class:`.CompilationStage` are now under :mod:`iqm.cpc.compiler.compiler`.

:class:`.Pulla` no longer needs CoCoS URL:

.. code-block:: python

    # BEFORE
    Pulla(cocos_url=cocos_url,station_control_url=station_control_url)


    # AFTER
    p = Pulla(station_control_url)

:class:`.Compiler` initialization now requires all arguments to be keyword arguments.

.. code-block:: python

    # BEFORE
    return Compiler(
        calibration_set,
        chip_topology,
        channel_properties,
        component_channels,
        qubit_mapping,
        stages=get_standard_stages(),
    )


    # AFTER
    return Compiler(
        calibration_set=calibration_set,
        chip_topology=chip_topology,
        channel_properties=channel_properties,
        component_channels=component_channels,
        qubit_mapping=qubit_mapping,
        stages=get_standard_stages(),
    )

Pulla no longer retrieves qubit mapping from CoCoS. If your circuit uses same qubits names as physical qubits
(e.g. ``QB1``, ``QB2``, etc.), then you don't have to do anything. If your circuit uses other qubit names, then you have to
provide ``component_mapping`` to the Compiler like so:

.. code-block:: python

    compiler.component_mapping = {'0':'QB1', '1':'QB2', '2':'QB3'}

If you use Qiskit, the mapping can be generated with ``{str(idx): qb for idx, qb in backend._idx_to_qb.items()}``.

``.Compiler.set_default_implementation_for_locus`` changed to :meth:`.Compiler.set_default_implementation_for_loci`:

.. code-block:: python

    # BEFORE
    compiler.set_default_implementation_for_locus('cz', 'tgss', ('QB1', 'QB2'))

    # AFTER
    compiler.set_default_implementation_for_loci('cz', 'tgss', [('QB1', 'QB2')])

:meth:`.Compiler.amend_calibration_for_gate_implementation` now accepts a Locus (a tuple of strings)
instead of string qubit name, so you can apply the change to multiple loci in a single call:

.. code-block:: python

    # BEFORE
    compiler.amend_calibration_for_gate_implementation('prx', 'raised_cosine', qubit, CUSTOM_CAL_DATA)

    # AFTER
    compiler.amend_calibration_for_gate_implementation('prx', 'raised_cosine', (qubit,), CUSTOM_CAL_DATA)

:func:`iqm.pulla.utils.qiskit_to_cpc` and :func:`iqm.pulla.station_control_result_to_qiskit`, previously marked for
deprecation in v. 1.0, are now removed. Use :func:`iqm.pulla.utils_qiskit.qiskit_to_cpc` and
:func:`iqm.pulla.utils_qiskit.station_control_result_to_qiskit`, respectively.


From 0.x to 1.0
---------------

The main change in 1.0 is the splitting of ``Pulla`` and ``Compiler``, and the move of some compiler-related methods to
the more appropriate :class:`.Compiler` class. The :class:`.Pulla` class now only contains methods to retrieve calibration data,
construct a standard compiler instance, and submit pulse schedules to the server.

.. code-block:: python

    # BEFORE

    pulla = Pulla(cocos_url="<cocos_url>")
    pulla.compiler.stages = STANDARD_STAGES

    pulla.compiler.compile(circuits)
    pulla.compiler.build_settings(context, shots=100)

    pulla.execute(playlist, context, settings)


    # AFTER

    pulla = Pulla(cocos_url="<cocos_url>", station_control_url="<station_control_url>")
    compiler = p.get_standard_compiler()  # already contains standard stages by default

    compiler.compile(circuits)
    compiler.build_settings(context, shots=100)

    pulla.execute(playlist, context, settings)

Setting default implementation is now done in the compiler directly, and there is no need to manually refresh the
compiler anymore.

.. code-block:: python

    # BEFORE
    pulla.gates['cz'].set_default_implementation('slepian')
    pulla.gates['cz'].set_default_implementation_for_locus('tgss', ('QB1', 'QB2'))
    pulla.refresh_compiler()

    # AFTER
    compiler.gates['cz'].set_default_implementation('slepian')
    compiler.gates['cz'].set_default_implementation_for_locus('tgss', ('QB1', 'QB2'))
    # no refresh needed

Same goes for adding implementations, and amending the calibration set with custom data for custom implementations:

.. code-block:: python

    # BEFORE
    pulla.add_implementation(...)
    pulla.amend_calibration_for_gate_implementation(...)


    # AFTER
    compiler.add_implementation(...)
    compiler.amend_calibration_for_gate_implementation(...)

The calibration is now stored solely in the compiler instance, and can be retrieved using
:meth:`.Compiler.get_calibration`:

.. code-block:: python

    # BEFORE
    current_calibration_set = pulla.get_current_calibration()


    # AFTER
    current_calibration_set = compiler.get_calibration()

Fetching calibration sets from the server is still done via :meth:`.Pulla.fetch_latest_calibration_set`
and :meth:`.Pulla.fetch_calibration_set_by_id`.

Standard compilation stages are now available via :func:`.get_standard_stages`. This ensures the immutability of
built-in standard stages.

.. code-block:: python

    # BEFORE
    from iqm.cpc.compiler.standard_stages import STANDARD_STAGES
    stages = STANDARD_STAGES


    # AFTER
    from iqm.cpc.compiler.standard_stages import get_standard_stages
    stages = get_standard_stages()
