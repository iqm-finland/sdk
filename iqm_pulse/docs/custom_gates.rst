Custom gate implementations
###########################

QuantumOp
---------

Quantum gates are represented by :class:`~iqm.pulse.quantum_ops.QuantumOp` data classes, containing the required
metadata to define the gate. A QuantumOp is identified by its :attr:`~iqm.pulse.quantum_ops.QuantumOp.name`, and
:attr:`~iqm.pulse.quantum_ops.QuantumOp.arity` defines number of locus components the operation acts on. For example,
the PRX operation (Phased X Rotation) is a single-qubit operation, so its arity is 1, whereas the CZ (Controlled-Z) gate
acts on two qubits, having arity 2. Arity 0 has a special meaning that the operation in question can act on any number
of components (for example :class:`~iqm.pulse.gates.barrier.Barrier`).

The attribute :attr:`~iqm.pulse.quantum_ops.QuantumOp.symmetric` defines whether the effect of the quantum operation
is symmetric with respect to changing the order of its locus components. As an example, the CZ gate is a symmetric
two-qubit gate, whereas CNOT (Controlled-NOT) is not symmetric.

Some quantum operations are defined as "functions", taking one or more parameters to define the effect. These
arguments are stored in the attribute :attr:`~iqm.pulse.quantum_ops.QuantumOp.params`. As an example, the PRX gate
takes two arguments, ``angle`` (the rotation angle with respect to the z-axis of the Bloch sphere), and ``phase``
(the rotation phase in the rotating frame). On the other hand, many operations do not require any parameters, in
which case this field is an empty tuple (e.g. the CZ gate).

A QuantumOp has unambiguous definition in terms of its *intended* effect on the computational subspace of the
QPU component, but it can be *implemented* in various ways. Each implementation is represented as a
:class:`~iqm.pulse.gate_implementation.GateImplementation` subclass. A QuantumOp stores its known implementations in the
field :attr:`~iqm.pulse.quantum_ops.QuantumOp.implementations`. Note that even though
:class:`~iqm.pulse.quantum_ops.QuantumOp` is a frozen data class, the implementations dictionary can be modified, e.g.
to add new implementations or to change their order (usually programmatically by some client procedure, but nothing as
such prevents the user from manipulating the contents manually). The default implementation is how the user prefers
to implement the operation unless otherwise specified (in effect, this is what will get called in most cases the
operation is invoked). In the implementations dict, the default implementation is defined as the first entry.
QuantumOp contains helpful methods that allow setting and returning the default implementation for specific cases:
:meth:`~iqm.pulse.quantum_ops.QuantumOp.set_default_implementation`,
:meth:`~iqm.pulse.quantum_ops.QuantumOp.get_default_implementation_for_locus`, and
:meth:`~iqm.pulse.quantum_ops.QuantumOp.set_default_implementation_for_locus`.

The attribute :attr:`~iqm.pulse.quantum_ops.QuantumOp.unitary` stores a function that can be used to get the unitary
matrix representing the quantum operation in question. The unitary function must have the same arguments
as defined in :attr:`~iqm.pulse.quantum_ops.QuantumOp.params`, such that for each collection of these parameters it
gives the associated unitary matrix. Note that not all QuantumOps necessarily even represent a unitary gate (e.g.
the measure operation is not one), or the exact form of the unitary matrix might not be known. In these cases, the
field can be left ``None``. The unitary does not need to be defined for most of the basic usage of a QuantumOp, but certain
algorithmic methods (e.g. some implementations of Randomized Benchmarking) may require the unitary matrices to be known,
and such operations that do not define the getter function cannot then be used in these contexts.

For more information, see the API docs of :class:`~iqm.pulse.quantum_ops.QuantumOp` for the full list of fields needed
to define a quantum operation and the available class methods.

Custom gate implementations
---------------------------

GateImplementation class
^^^^^^^^^^^^^^^^^^^^^^^^

While :class:`~iqm.pulse.quantum_ops.QuantumOp` represents an abstract quantum operation, its *implementations*  contain
the concrete logic of how to make that operation happen using QC hardware. Gate implementations are subclasses of
:class:`~iqm.pulse.gate_implementation.GateImplementation`. In this section, the main features of that class are
introduced (for a full list of class methods see the API docs), with the emphasis being on how to create your own
gate implementations.

Starting with :meth:`~iqm.pulse.gate_implementation.GateImplementation.__init__`, it is important to note that the init
methods of all gate implementations must have the exact same signature:

.. code-block:: python

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: tuple[str,...],
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder
    ):

Here, ``parent`` is the ``QuantumOp`` this gate implementation implements, and ``name`` is the implementation's name in
the dictionary :attr:`~iqm.pulse.quantum_ops.QuantumOp.implementations`. ``locus`` is the set of (usually logical) components
the QuantumOp acts on (the size of the locus must be consistent with the ``parent``'s
:attr:`~iqm.pulse.quantum_ops.QuantumOp.arity`), while ``calibration_data`` gives the required calibration data values
for this implementation and ``locus`` (can be empty in case the implementation needs no calibration data). Finally,
The implementations store a reference to the :class:`~iqm.pulse.builder.ScheduleBuilder` that created it. This is
because GateImplementations are practically never created manually by calling the init method itself. Instead, one
needs a builder and uses :meth:`~iqm.pulse.builder.ScheduleBuilder.get_implementation`.

The responsibility of the init method is to (at least) store the ``calibration_data`` provided from the builder for
further use, but in many cases, one might want to create some intermediate objects like pulses or instructions **from**
that calibration data already at this point. Note that ScheduleBuilder caches its GateImplementations per each locus and
``calibration_data``, so as long as the calibration is not changed, the code in init will be called just once per locus.

GateImplementations are Callables, i.e. they implement the `__call__` method. It should take as its arguments at least
the QuantumOpt parameters defined for the ``parent`` in :attr:`~iqm.pulse.quantum_ops.QuantumOp.params`, but in
addition it may have optional extra arguments. The call method should return a :class:`~iqm.pulse.timebox.TimeBox` object
that contains the pulses, instructions and other logic required to implement the quantum operation in question. The
typical usage of gate implementations then looks like this (See :doc:`using_builder` and :doc:`pulse_timing` for more
info on scheduling and the ScheduleBuilder):

.. code-block:: python

    # this initializes the _default implementation_ class of PRX for QB1
    default_prx_QB1 = builder.get_implementation("prx", ("QB1",))
    # this initializes a specific PRX implementation for QB1, not necessarily the default
    special_prx_QB1 = builder.get_implementation("prx", ("QB1",), impl_name="my_special_PRX")
    # calling the implementation with the QuantumOp param values creates a TimeBox that can be then scheduled with
    # the normal scheduling logic
    default_box = default_prx_QB1(angle=np.pi, phase=np.pi/2)

    # the initialization of the impl class and the call can of course be also chained together like this:
    default_cz_box =  builder..get_implementation("cz", ("QB1", "QB2"))()  # CZ has no QuantumOp params!

The base class :meth:`~iqm.pulse.gate_implementation.GateImplementation.__call__` method does automatic TimeBox caching based
on the unique values of the call arguments, and in many cases, one does not want to reimplement this caching in their own
implementations. For this reason, there is the method ``_call`` which contains just the pure TimeBox creation logic.
Developers can choose to override that instead of ``__call__`` in cases where the call args are hashable python types,
and then they can utilize the default caching of TimeBoxes from the base class.

When writing a GateImplementation, a developer should consider what parts of the logic should go to the class init and
what to the ``__call__`` or ``_call`` method. A general rule of thumb would be that any parts that can be precomputed
and do not depend on the call arguments can go to init, and the rest to call.

As an example, let's go through a simple PRX ``_call`` method (note that the default PRX implementations do not
use this exact call method, as this is a simplified example for educational purposes):

.. code-block:: python

    def _call(self, angle: float, phase: float = 0.0) -> TimeBox:
        instruction = IQPulse(  # create the Instruction using the calibration data
            scale_i=angle,  # pulse amplitudes from the inputted angle
            scale_q=angle,
            wave_i=TruncatedGaussian(**self.calibration_data),  # pulse i waveform (normalized to one)
            wave_q=TruncatedGaussianDerivative(**self.calibration_data),  # pulse q waveform  (normalized to one)
            phase=phase,
        )
        # create the TimeBox
        return TimeBox.atomic(
            schedule=Schedule({self.channel: [instruction]}),  # atomic Schedule created from the pulse
            locus_components=self.locus,
            label=f"{self.__class__.__name__} on {self.locus}",  # (optional) label for identifying the TimeBox
        )

Here, we first create an :class:`.IQPulse` object which is a low-level Instruction. IQPulse
means a "complex pulse" which has two orthogonal components i and q -- this what drive pulses look like in general. In
this simplified example, we have hardcoded the pulse waveforms into :class:`.TruncatedGaussian` and
:class:`.TruncatedGaussianDerivative` for the i and q components, respectively (this is a DRAG implementation, so the
q component is the derivative of the i component). The waveforms are parametrized by the ``calibration_data`` for the
given ``locus`` (see the next subsection for more info on Waveforms and calibration data). The PRX QuantumOp param
``angle`` scales the pulse amplitude linearly (the waveforms are normalized to one), and the param ``phase`` defines relative
phase modulation. Then the returned TimeBox is created out of the ``instruction``. Note that
since we override ``_call`` here, instead of ``__call__``, so this implementation would utilize the default base class
caching such that the TimeBoxes are cached per unique values of ``(angle, phase)``.

Another important concept is a the so called locus mapping of a gate implementation. Locus mappings define on which
loci, i.e. groups of components, a given implementation can be defined. They are used to relay the information which
loci are supported to a client application (e.g. EXA). In addition, the gate implementation itself can programmatically
use this information ``self.builder.chip_topology``.

For example, a PRX can be defined on all single components that are connected to a drive line, and CZ can be defined on
connected pairs of qubits. Locus mappings live in ``ScheduleBuilder.chip_topology`` which is a
:class:`~exa.common.qcm_data.chip_topology.ChipTopology` object. Locus mapping is a dict whose keys are the loci
(``tuple[str, ...]`` keys denote asymmetric loci where the order of the components matter, and ``frozenset[str]`` type
loci denote symmetric ones), and the values are groups of components, typed ``tuple[str, ...]``, where each locus can be
mapped with some additional components that are needed for the operation of the implementation. For example, some CZ
implementation that tries to correct for crosstalk could map the non-locus components that see this crosstalk here.
The values of the dict can be left empty or just replicate the key components in case such extra information is not
needed.

GateImplementations can define their locus mappings via
:meth:`~iqm.pulse.gate_implementation.GateImplementation.get_custom_locus_mapping` or if a client application already
adds the mapping, we can just return its name via :meth:`~iqm.pulse.gate_implementation.GateImplementation.get_locus_mapping_name`.
If neither of these methods are overridden in a GateImplementation class, the default behaviour will be such that an
``arity==1`` loci will be assumed to use the mapping where all single qubits are the keys, and ``arity==2`` loci the
(symmetric) mapping where the keys are all pairs of connected qubits. For other arities there is no default behaviour,
so it is then mandatory to define the mapping explicitly using the aforementioned methods.

Instructions, Waveforms and calibration data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to implement most QuantumOps, one has to physically alter the state of the QPU. This is typically done by playing
specified and correctly calibrated pulses via the control electronics (this applies to all typical logical gates such as
e.g. PRX or CZ -- non-physcial metaoperations such as Barrier are an exception). In defining these pulses, there are two
levels of abstractions: :class:`.Waveform` and :class:`.Instruction`.

Waveform represents the physical form of the control pulse, typically normalized to the interval ``[-1.0, 1.0]``. The
Each :class:`~iqm.models.playlist.waveforms.Waveform` subclass can define any number of waveform parameters as class
attributes, which can be used to programmatically define the waveform. For example, a Gaussian could be defined in terms
of the average ``mu`` and spread ``sigma``. A Waveform class then essentially contains just the parameters
and a recipe for computing the samples as an ``np.ndarray``. As an example, here is how one writes the Waveform class
for ``Gaussian``:

.. code-block:: python

    class Gaussian(Waveform):

        # waveform parameters as class attributes
        sigma: float
        mu: float = 0.0

        def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
            offset_coords = sample_coords - self.center_offset
            return np.exp(-0.5 * (offset_coords / self.sigma) ** 2)

The Instructions :class:`.RealPulse` and
:class:`.IQPulse` allow handling the amplitudes (via the attribute ``scale``) without
having to resample the waveform for every different amplitude value. However, one can always choose to include
the amplitude into the sampling and then use ``scale=1``.

The waveform parameters (like ``sigma`` in the above Gaussian) typically require calibration when the Waveform is used
in a quantum gate. However, the GateImplementation usually has other calibrated parameters as well defined in the
implementation itself. As an example, here are the implementation-level parameters of the default PRX implementation,
defined as class attribute:

.. code-block:: python

    parameters: dict[str, Parameter | Setting] = {
        "duration": Parameter("", "pi pulse duration", "s"),
        "amplitude_i": Parameter("", "pi pulse I channel amplitude", ""),
        "amplitude_q": Parameter("", "pi pulse Q channel amplitude", ""),
    }

Note the amplitudes are defined here on this level, since the default PRX uses normalized Waveforms and factors in the
amplitudes via ``scale``. In these parameters, the unit is not just metadata. The control electronics understand time
in terms of samples and their sample rate, while human users typically want to input seconds instead of doing the sample
conversion manually. For this reason, there is logic that converts anything that has the unit ``"s"`` into samples.
Similarly, parameters with ``"Hz"`` units are converted to ``1/sample``. For the Waveform parameters, the same logic
applies, but by default it is assumed that all parameters are time-like and this converted from seconds to samples.
If some Waveform parameters needs to be made unitless or e.g. frequency-like (with ``"Hz"`` units), it can be achieved
with the method :meth:`~iqm.models.playlist.waveforms.Waveform.non_timelike_attributes`:

.. code-block:: python

    def non_timelike_attributes() -> dict[str, str]:
        return {
            "frequency": "Hz",
            "scalar_coeffiecient", ""
        }

In the above dict, the keys should be the attribute names and values their units.

More base classes
^^^^^^^^^^^^^^^^^

To make creating new GateImplementations more comfortable, there are additional base classes on top of
:class:`.GateImplementation` itself.

:class:`.CompositeGate` allows quick implementation of gates in terms of other gates,
using a similar syntax as with creating/scheduling several TimeBoxes together (see :doc:`using_builder`). At it
simplest, a ComposteGate is just the `_call` method:

.. code-block:: python

    class CompositeHadamard(CompositeGate):
        """Composite Hadamard that uses PRX"""
        registered_gates = ["prx"]
        # registering member gates is not mandatory, but allows calibrating them specifically inside _this_ composite

        def _call(self) -> TimeBox:
            member_prx = self.build("prx", self.locus)
            return member_prx(np.pi / 2, np.pi / 2 ) + member_prx(np.pi, 0.0)

Here, one could use also ``builder.get_implementation`` instead of
:meth:`~iqm.pulse.gate_implementation.CompositeGate.build`, but the latter allows calibrating the member gates
case specifically for this composite if they are first registered via
:attr:`~iqm.pulse.gate_implementation.CompositeGate.registered_gates` (in this case, there is
just one member, PRX).

Creating new implementations for the PRX, CZ and Measure gates often means just coming up with new waveforms for the
control pulses. If this is the case, there are helpful base classes that make those implementations into oneliners
(outside of defining the Waveforms themselves): :class:`~iqm.pulse.gates.prx.PRX_CustomWaveforms`,
:class:`~iqm.pulse.gates.cz.FluxPulseGate`, and :class:`~iqm.pulse.gates.measure.Measure_CustomWaveforms`. Using these
base classes at its simplest looks like this:

.. code-block:: python

    class PRX_MyCoolWaveforms(PRX_CustomWaveForms, wave_i=CoolWaveformI, wave_q=CoolWaveformQ):
        """PRX with my cool custom waveforms for the i and q drive pulse components"""

    class CZ_MyCoolWaveforms(FluxPulseGate, coupler_wave=CoolCouplerWaveform, qubit_wave=CoolQubitWaveform):
        """CZ with my cool qubit and coupler flux pulse waveforms"""

    class Measure_MyCoolWaveforms(Measure_CustomWaveforms, wave_i=CoolWaveformI, wave_q=CoolWaveformQ):
        """Measure with my cool custom waveforms for the i and q probe pulse components"""

All of these classes automatically include the associated Waveform parameters into the calibration parameters of
the implementation itself. There is also a general base class for any gate that implements a single ``IQPulse``
(both PRX_CustomWaveForms and Measure_MyCoolWaveforms actually inherit from it), regardless of the context:
:class:`~iqm.pulse.gate_implementation.CustomIQWaveforms`.


Registering gates and implementations
-------------------------------------

Gate definitions (i.e. QuantumOps) are stored in :class:`~iqm.pulse.builder.ScheduleBuilder`'s attribute
``op_table``. When the builder is created, the ``op_table`` comes preloaded with the all the basic QuantumOps needed for
typical circuit execution and their default implementations. These include e.g. the PRX gate, the CZ gate, the measure
operation, the conditional prx operation, the reset operation, and the barrier operation.

In order to add custom operations, there is a helpful function :func:`~iqm.pulse.gates.register_implementation` that
in addition to adding new implementations allows one to add altogether new quantum operations.

As an example here is a snippet that adds the CNOT gate, and its implementation, into an existing builder:

.. code-block:: python

    cnot_matrix = np.array([[1, 0, 0, 0],  # the unitary is not strictly necessary for basic use, but since
                            [0, 1, 0, 0],  # we do know its form for CNOT, why not add it
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=complex)
    cnot_op = QuantumOp(name="cnot", arity=2, symmetric=False, unitary=lambda: cnot_matrix)

    register_implementation(
        operations=my_builder.op_table,
        gate_name="cnot",
        impl_name="my_cnot_impl",
        impl_class=MyCNotClass,
        quantum_op_specs=cnot_op
    )

Here, the CNOT implementation ``MyCNotClass`` needs to be of course defined first (a QuantumOp always needs at least one
implementation).

**Note:** The end user cannot modify the canonical mapping (defined in iqm-pulse) between ``implementation_name`` and 
``implementation_class``.

Note that often :class:`.ScheduleBuilder` is created and operated by some client application, and the same application usually
has its own interface for adding/manipulating QuantumOps. However, if the user has access to the builder object, the
above method will always work.
