# Copyright 2024 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementing abstract quantum operations as instruction schedules.

.. note::

   Note the conceptual difference between :class:`quantum operations <.QuantumOp>` (ops) and
   :class:`instruction schedules <.Schedule>`. Ops represent abstract, ideal computational
   operations, whereas instruction schedules represent concrete control signal sequences for the
   quantum computer. One can (approximately) implement an op using a number of different
   instruction schedules.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable
import dataclasses
import re
from typing import TYPE_CHECKING, Any, TypeAlias, get_type_hints
from uuid import uuid4

import numpy as np

from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.playlist.instructions import Instruction
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.timebox import TimeBox
from iqm.pulse.utils import map_waveform_param_types

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.playlist.channel import ChannelProperties
    from iqm.pulse.playlist.waveforms import Waveform
    from iqm.pulse.quantum_ops import QuantumOp


Locus: TypeAlias = tuple[str, ...]
"""Sequence of QPU component physical names a quantum operation is acting on. The order may matter."""

OILCalibrationData: TypeAlias = dict[str, Any]
"""Calibration data for a particular implementation of a particular quantum operation at a particular locus."""

OICalibrationData: TypeAlias = dict[Locus | None, OILCalibrationData]
"""For a particular implementation of a particular quantum operation, maps operation loci to their calibration data."""

OCalibrationData: TypeAlias = dict[str, OICalibrationData]
"""For a particular quantum operation, maps implementation names to their calibration data."""

OpCalibrationDataTree: TypeAlias = dict[str, OCalibrationData]
"""Maps :class:`quantum operation <.QuantumOp>` names to their calibration data."""

NestedParams: TypeAlias = dict[str, Parameter | Setting | dict]
"""Nested dict defining the parameters required by GateImplementation classes."""
# TODO replace Parameter with a custom class, eliminating a part of the exa-common dependency

SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING = "single_components_with_drive"
"""Locus mapping name for mapping all components that have the drive operation defined."""
SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING = "single_components_with_readout"
"""Locus mapping name for mapping all components that have the readout operation defined."""
SINGLE_COMPONENTS_WITH_FLUX_AWG_LOCUS_MAPPING = "single_components_with_flux_awg"
"""Locus mapping name for mapping all components that have the flux operation defined and the flux controller
 has an AWG."""
PROBE_LINES_LOCUS_MAPPING = "probe_lines"
"""Locus mapping name for mapping all probe lines."""


class GateImplementation(abc.ABC):
    """ABC for implementing quantum gates and other quantum operations using instruction schedules.

    There is a separate GateImplementation subclass for every implementation of every operation type.
    Each GateImplementation instance represents a particular locus for that implementation, and encapsulates
    the calibration data it requires.

    All GateImplementation subclasses ``__init__`` must have exactly the below arguments in order to be
    usable via :meth:`.ScheduleBuilder.get_implementation`.

    GateImplementations also have the :meth:`~GateImplementation.__call__` method, which takes the operation parameters
    (e.g. rotation angles) as input, and returns a :class:`.TimeBox` instance which implements
    an instance of the operation at that locus.

    Args:
        parent: Quantum operation this instance implements.
        name: Name of the implementation provided by this instance.
        locus: Locus the operation acts on.
        calibration_data: (Raw) calibration data for the (operation, implementation, locus) represented by this instance
        builder: Schedule builder.

    """

    symmetric: bool = False
    """True iff the implementation is symmetric in its locus components.
    Only meaningful if ``arity != 1``, and the locus components are of the same type."""

    parameters: NestedParams = {}
    """Required calibration data, may be nested"""

    special_implementation: bool = False
    """Set to ``True`` if  the implementation is a special purpose implementation that should never get called in
    ``ScheduleBuilder.get_implementation`` unless explicitly requested via the ``impl_name`` argument."""

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ) -> None:
        self.parent = parent
        self.name = name
        self.locus = locus
        self.calibration_data = calibration_data
        self.builder = builder
        self.id = str(uuid4())  # unique str identifier, needed for certain caching properties
        self._timebox_cache: dict[tuple[Any, ...], TimeBox | list[TimeBox]] = {}

    @property
    def qualified_name(self) -> str:
        """Qualified name of the implementation."""
        return f"{self.parent.name}.{self.name}"

    @classmethod
    def needs_calibration(cls) -> bool:
        """Whether the implementation needs calibration data

        Returns True if the calibration dict must contain a node with keyed with
        <operation name>: <implementation name>: <appropriate locus> in order to use this implementation.
        """
        return bool(cls.parameters)

    def __call__(self, *args, **kwargs) -> TimeBox | list[TimeBox]:
        """TimeBox that implements a quantum operation.

        Implements TimeBox caching based on ``args`` and ``kwargs``. This method can be overridden if there's a need for
        different kind of caching (e.g. non-hashable args/kwargs) and/or specific additional logic.

        Returns:
            A TimeBox implementing the quantum operation represented by this instance,
            at the locus represented by this instance, with the given operation parameters.
            Alternatively, a list of TimeBoxes, in which case the exact relative timing
            of the TimeBoxes is not relevant for the implementation.

        """
        default_cache_key = tuple(args) + tuple(kwargs.items())
        try:
            hash(default_cache_key)
            key_is_hashable = True
        except TypeError:
            key_is_hashable = False
        if key_is_hashable and default_cache_key in self._timebox_cache:
            return self._timebox_cache[default_cache_key]
        timebox = self._call(*args, **kwargs)
        if key_is_hashable:
            self._timebox_cache[default_cache_key] = timebox
        return timebox

    def _call(self, *args, **kwargs) -> TimeBox | list[TimeBox]:
        """The GateImplementation-specific logic for implementing a quantum operation.

        Inheriting classes may override this method if the default :meth:`__call__` caching (based on the args & kwargs
        in the signature) is sufficient. Any additional caching may also be implemented inside this function if needed.
        """
        return NotImplementedError  # type: ignore[return-value]

    def build(
        self, op_name: str, locus: Locus, impl_name: str | None = None, strict_locus: bool = False
    ) -> GateImplementation:
        """Utility method for constructing a GateImplementation with ``self.builder``.

        Inheriting classes may override this in order to add additional logic.

        Args:
            op_name: operation name
            locus: locus the operation acts on
            impl_name: implementation name. Uses the assigned default implementation if not specified.
            strict_locus: iff False, for non-symmetric implementations of symmetric ops the locus order may
                be changed if no calibration data is available for the requested locus order

        Returns:
            Calibrated gate implementation.

        """
        return self.builder.get_implementation(op_name, locus, impl_name, strict_locus=strict_locus)

    def to_timebox(self, schedule: Schedule) -> TimeBox:
        """Wraps the given instruction schedule into an atomic/resolved timebox."""
        return TimeBox.atomic(
            schedule,
            locus_components=self.locus,
            label=f"{self.__class__.__name__} on {self.locus}",
        )

    def duration_in_seconds(self) -> float:
        """Duration of the Schedule of the gate implementation (in seconds).

        Can be left unimplemented if the duration e.g. depends on the gate arguments.
        Subclasses can reimplement this method in case it makes sense in their context.
        """
        raise NotImplementedError

    @classmethod
    def convert_calibration_data(
        cls,
        calibration_data: OILCalibrationData,
        params: NestedParams,
        channel_props: ChannelProperties,
        duration: float | None = None,
    ) -> OILCalibrationData:
        """Convert time-like items in the calibration data to fractions of the time duration of the gate.

        This is a convenience method for converting calibration data items involving time
        durations measured in seconds into fractions of the duration of the gate.

        * Values of items that are not measured in seconds or Hz are returned as is.
        * Additionally, converts ``duration`` to channel samples and adds it in the converted
          calibration data under the key ``"n_samples"``, while the original ``"duration"`` key is removed.

        Args:
            calibration_data: (subset of) calibration data for the gate/implementation/locus
            params: (subset of) ``cls.parameters`` specifying the ``calibration_data`` items
                to convert and return
            channel_props: used to convert ``"duration"`` from seconds into channel samples
            duration: Time duration of the gate, in seconds. If None, ``calibration_data`` must have
                an item named ``"duration"``, measured in seconds, which will be used instead.

        Returns:
            converted ``calibration_data`` items

        """

        def convert(name: str, unit: str, value: Any, dur: float) -> Any:
            """Convert time-like values to the units of multiples of ``duration`` and frequency-like values
            to the units of multiples of the inverse of duration.
            """
            if value is None:
                return None
            if (unit not in {"s", "Hz"}) or name == "duration":
                return value
            if unit == "s":

                def conversion(val: Any) -> Any:
                    return val / dur if dur > 0 else 0.0

            elif unit == "Hz":

                def conversion(val: Any) -> Any:
                    return val * dur

            if isinstance(value, Iterable):
                converted = list(map(conversion, value))
                return np.array(converted) if isinstance(value, np.ndarray) else converted
            return conversion(value)

        if duration is None:
            # duration should be found in the outermost dict
            duration = calibration_data["duration"]
            if duration is None:
                raise ValueError(f"Duration for {cls.__name__} has not been set.")
            converted = {"n_samples": channel_props.duration_to_int_samples(duration) if duration > 0 else 0}
        else:
            converted = {}

        for p_name, p in params.items():
            if p_name in calibration_data:
                data = calibration_data[p_name]
                if isinstance(p, Setting | Parameter):
                    value = convert(p_name, p.unit, data, duration)
                else:
                    # recursion for nested parameter dicts
                    value = cls.convert_calibration_data(data, p, channel_props, duration=duration)
                if p_name != "duration":  # duration is converted to n_samples so we don't want it in seconds here
                    converted[p_name] = value

        return converted

    @classmethod
    def get_parameters(cls, locus: Iterable[str], path: Iterable[str] = ()) -> SettingNode:
        """Calibration data tree the GateImplementation subclass expects for each locus.

        Helper method for EXA use.

        Args:
            locus: Locus component names to replace the wildcard character ``"*"`` in the calibration
                parameter names. One ``Setting`` will be generated for each component name in ``locus``.
                If there are no wildcard characters in ``cls.parameters``, this argument has no effect.
            path: parts of the dotted name for the root node, if any.

        Returns:
            EXA setting node describing the required calibration data for each locus.
            All the Setting values are ``None``.

        """

        def build_node(path: Iterable[str], dictionary: dict[str, Any]) -> SettingNode:
            node = SettingNode(".".join(path), path=".".join(path))
            for key, value in dictionary.items():
                wildcard_keys = [key.replace("*", q) for q in locus] if "*" in key else [key]

                for wkey in wildcard_keys:
                    new_path = (*tuple(path), wkey)
                    if isinstance(value, dict):
                        node.subtrees[wkey] = build_node(new_path, value)
                    elif isinstance(value, Setting | Parameter):
                        name = ".".join(new_path)
                        if isinstance(value, Parameter):
                            node.settings[wkey] = Setting(value.model_copy(update={"name": name}), None, path=name)
                        else:
                            node.settings[wkey] = Setting(
                                value.parameter.model_copy(update={"name": name}), value.value, path=name
                            )
                    else:
                        raise TypeError(f"{wkey}: value {value} is neither a Parameter, Setting nor a dict.")
            return node

        return build_node(path, cls.parameters)

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        """Get the name of the locus mapping stored in ``ScheduleBuilder.ChipTopology`` for this implementation.

        By default, it is ``"<operation_name>.<implementation_name>"``. Inheriting classes may
        override this for different behaviour.

        Args:
            operation_name: name of the quantum operation.
            implementation_name: name of the implementation

        Returns:
            name of the locus mapping

        """
        return f"{operation_name}.{implementation_name}"

    @classmethod
    def get_custom_locus_mapping(
        cls, chip_topology: ChipTopology, component_to_channels: dict[str, Iterable[str]]
    ) -> dict[tuple[str, ...] | frozenset[str], tuple[str, ...]] | None:
        """Get custom locus mapping for this GateImplementation.

        This method can be used to return the locus mapping (wrt. to the given ``ChipTopology``) for this
        ``GateImplementation``. Overriding this method allows a GateImplementation to be "self-sufficient" in the
        sense that it knows its own locus mapping.

        Args:
            chip_topology: ChipTopology instance in which context to create the custom locus mapping.
            component_to_channels: dict mapping QPU component names to an ``Iterable`` of channel operation names
                available for this component (i.e. "readout", "drive", "flux"). This info is often needed
                in building a locus mapping.

        Returns:
            Custom locus mapping for this GateImplementation or ``None`` if the gate implementation has no need for a
                custom locus mapping, otherwise the returned mapping should be like in
                :meth:`ChipTopology.set_locus_mapping`

        """
        return None


class CustomIQWaveforms(GateImplementation):
    """Base class for GateImplementations using custom waveform definition with IQPulses.

    The class contains logic for automatic gate calibration parameters handling for such gates (see the class
    attributes for more info). With given :class:`.Waveform` waveform definitions ``Something`` and ``SomethingElse``,
    an inheriting class may define the waveforms for the I and Q channels like this:
    ``class MyGate(CustomIQWaveforms, i_wave=Something, q_wave=SomethingElse)``.
    """

    wave_i: type[Waveform]
    """Waveform for the I channel."""
    wave_q: type[Waveform]
    """Waveform for the Q channel."""
    dependent_waves: bool
    """If set ``True``, the Q channel waveform is considered to depend on the I channel's waveform
    so that they share the waveform parameters, (e.g. a DRAG PRX implementation). If not provided,
    will be initialised as ``True``."""
    root_parameters: dict[str, Parameter | Setting] = {}
    """Parameters independent of the of Waveforms. Inheriting classes may override this to include parameters common
    to all such implementations."""
    excluded_parameters: list[str] = []
    """Parameters names to be excluded from ``self.parameters``. Inheriting classes may override this if certain
    parameters are not wanted in that class (also parameters defined by the waveforms can be excluded)."""

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ) -> None:
        super().__init__(parent, name, locus, calibration_data, builder)
        if getattr(self, "wave_i", None) is None or getattr(self, "wave_q", None) is None:
            raise ValueError(
                "You must provide valid Waveforms for both of the arguments `wave_i` and `wave_q` when inheriting"
                "from CustomIQWaveforms."
            )

    def __init_subclass__(
        cls,
        /,
        wave_i: type[Waveform] | None = None,
        wave_q: type[Waveform] | None = None,
        dependent_waves: bool | None = None,
    ) -> None:
        """Store the Waveform types used by this subclass, and their parameters.

        NOTE: if ``MyGate`` is a subclass of ``CustomIQWaveforms``, with some defined i and q waves, further
        inheriting from it like this ``MySubSubClass(MyGate, wave_i=Something, wave_q=SomethingElse)``
        changes the waves accordingly. If you do not provide any subclass arguments: ``class MySubSubClass(MyGate)``,
        the waves defined in ``MyGate`` and ``MyGate.dependent_waveforms`` will be retained. You must provide both waves
        at the same time if you want to change just one of them. Similarly, you can change the value of
        ``dependent_waveforms`` like this: ``class MySubSubClass(MyGate, dependent_waveforms=False)``.

        Args:
            wave_i: waveform for the I channel.
            wave_q: waveform for the Q channel.
            dependent_waves: if set ``True``, the Q channel waveform is considered to depend on the I channel's
                waveform such that they share the waveform parameters. If not provided, will be initialised as ``True``.

        """
        # fix __init_subclass__ behaviour for further inheritance from a subclass of CustomIQWaveforms
        # we can skip this function if the class attributes are already stored in the parent class
        # and the subsubclass definition does not change these
        # see more info in: https://stackoverflow.com/questions/55183288/inheriting-init-subclass-parameters
        # the unintuitive default ``None`` values and handling of these values is for overcoming this issue
        # so that the method itself behaves as expected in successive subclassing
        if (
            wave_i is None
            and wave_q is None
            and dependent_waves is None
            and hasattr(cls, "wave_i")
            and hasattr(cls, "wave_q")
        ):
            return
        if wave_i is not None or wave_q is not None:
            cls.wave_i = wave_i  # type: ignore[assignment]
            cls.wave_q = wave_q  # type: ignore[assignment]
        if not hasattr(cls, "dependent_waveforms") or dependent_waves is not None:
            # True by default but we don't want to change it in subsubclasses unless explicitly given
            cls.dependent_waves = True if dependent_waves is None else dependent_waves
        if getattr(cls, "wave_i", None) and getattr(cls, "wave_q", None):
            root_parameters = {k: v for k, v in cls.root_parameters.items() if k not in cls.excluded_parameters}
            parameters_i = {
                k: v
                for k, v in get_waveform_parameters(
                    cls.wave_i, label_prefix="I wave " if not cls.dependent_waves else ""
                ).items()
                if k not in cls.excluded_parameters
            }
            if cls.dependent_waves:
                cls.parameters = root_parameters | parameters_i  # type: ignore[assignment]
            else:
                parameters_q = {
                    k: v
                    for k, v in get_waveform_parameters(cls.wave_q, "Q wave ").items()
                    if k not in cls.excluded_parameters
                }
                cls.parameters = root_parameters | {"i": parameters_i, "q": parameters_q}


def get_waveform_parameters(wave: type[Waveform], label_prefix: str = "") -> dict[str, Setting | Parameter]:
    """Parameters that are required to initialize the given Waveform class.

    ``n_samples`` is handled separately since it is determined by the :class:`.Instruction` duration
    and channel sample rate, and thus is shared by all the waveforms of the Instruction.

    Args:
        wave: waveform class
        label_prefix: optional prefix for the parameter labels for providing additional information

    Returns:
        parameters of ``wave``, in the format expected by :attr:`GateImplementation.parameters`. Waveform parameters
        that have a defined default will be returned as ``Setting`` objects and those that do not have default
        as ``Parameter`` objects.

    """
    # This could also be a Waveform classmethod, but it's better if we keep the waveform module
    # lean and simple without any exa-common dependencies.
    nta = wave.non_timelike_attributes()
    parameters: dict[str, Setting | Parameter] = {}
    for field in dataclasses.fields(wave):
        if field.name != "n_samples":
            resolved_types = get_type_hints(wave)
            data_type, collection_type = map_waveform_param_types(resolved_types[field.name])
            abbr_waveform_name = re.sub("[^A-Z]", "", wave.__name__).lower()
            waveform_name = wave.__name__ if len(abbr_waveform_name) < 2 else abbr_waveform_name
            label = label_prefix + f"{field.name} of {waveform_name}"
            param = Parameter(
                "",
                label=label,
                unit=nta.get(field.name, "s"),
                data_type=data_type,
                collection_type=collection_type,
            )
            if not isinstance(field.default, dataclasses._MISSING_TYPE):
                parameters[field.name] = Setting(param, field.default)
            elif not isinstance(field.default_factory, dataclasses._MISSING_TYPE):
                parameters[field.name] = Setting(param, field.default_factory())
            else:
                parameters[field.name] = param

    return parameters


class SinglePulseGate(GateImplementation):
    """Base class for GateImplementations that play a single pulse on a single channel.

    The pulse is created in :meth:`_get_pulse` and the channel is specified in :meth:`_get_pulse_channel`.
    The base class also implements a basic :meth:`_call` method that just inserts the specified pulse into the specified
    channel, and a method for computing the pulse's duration. All of these methods can be overridden in subclasses.
    """

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        self.channel = self._get_pulse_channel()
        params = self.convert_calibration_data(calibration_data, self.parameters, builder.channels[self.channel])
        self.pulse = self._get_pulse(**params)

    def _call(self) -> TimeBox:
        return self.to_timebox(Schedule({self.channel: [self.pulse]}))

    def _get_pulse_channel(self) -> str:
        """Return the channel for the pulse.

        The default is the drive channel for a single qubit locus.
        """
        return self.builder.get_drive_channel(self.locus[0])

    def _get_pulse(self, **kwargs) -> Instruction:
        """Return pulse based on the provided calibration data."""
        raise NotImplementedError

    def duration_in_seconds(self) -> float:
        if self.pulse.duration == 0:
            return 0.0
        return self.builder.channels[self.channel].duration_to_seconds(self.pulse.duration)


class CompositeGate(GateImplementation):
    """Utility base class for creating gate implementations that are defined in terms of other gate implementations.

    Gates can be implemented using other pre-existing gate implementations by just utilizing the ScheduleBuilder in
    :attr:`~GateImplementation.builder` in the :meth:`__call__` method (e.g. by calling
    ``self.builder.get_implementation(<some gate>, <some locus>)``. In this way, any such "member gates" will use
    the common calibration that exists in :attr:`~GateImplementation.builder`. In order for a composite gate
    implementation to be able to calibrate its member gates with different calibration values from the common
    calibration, it needs to know what gates it considers as its "members". This is what the CompositeGate ABC is for.

    Inheriting from this class and defining e.g. ``registered_gates = ["prx", "cz"]`` allows one to calibrate the
    member operations (i.e. ``"prx"`` and ``"cz"`` in this example) inside this composite gate differently from the
    common calibration. However, if no specific calibration data is provided, the gate implementation will be calibrated
    with the common calibration.
    """

    registered_gates: list[str] = []
    """Gates that can be calibrated separately from their common calibration existing in ``self.builder``. The gate
    names should correspond to the keys in ``self.builder.op_table``. Other gates besides the ones given here can
    also be constructed via ``self.builder``, but these will always use the common calibration."""
    default_implementations: dict[str, str] = {}
    """Mapping from operation names to the designated default implementation of that operation. Filling this attribute
    allows one to define a different default implementation from the common default in ``self.builder.op_table`` to
    be used in he context of this composite gate. If an operation is not found in this dict as a key, this
    CompositeGate will use the common default as the default implementation for it.
    """

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ) -> None:
        super().__init__(parent, name, locus, calibration_data, builder)
        custom_defaults = {
            op: data.get("default_implementation") for op, data in calibration_data.items() if isinstance(data, dict)
        }
        self._default_implementations = self.default_implementations.copy()
        for op_name, impl in custom_defaults.items():
            if op_name in self.registered_gates and impl is not None:
                self._default_implementations[op_name] = impl

    def __call__(self, *args, **kwargs):
        default_cache_key = tuple(args) + tuple(kwargs.items())
        try:
            hash(default_cache_key)
            key_is_hashable = True
        except TypeError:
            key_is_hashable = False
        if key_is_hashable and (box := self.builder.composite_cache.get(self, default_cache_key)):
            return box
        box = self._call(*args, **kwargs)
        if key_is_hashable:
            self.builder.composite_cache.set(self, default_cache_key, box)
        return box

    def build(
        self, op_name: str, locus: Locus, impl_name: str | None = None, strict_locus: bool = False
    ) -> GateImplementation:
        """Construct a member gate implementation.

        If the gate ``op_name`` is registered, a specific calibration for it in the context of this CompositeGate
        will be sought for from ``self.builder.calibration``. If any (non-empty) calibration values are found in
        ``self.builder.calibration[self.name][op_name][<impl_name>]`` they will be merged to the common calibration
        (only non-empty values will be merged). If there are no values found, the
        common calibration will be used.

        Args:
            op_name: operation name
            locus: locus the operation acts on
            impl_name: Implementation name. If not given, uses the default implementation defined in the class instance
                if any, and otherwise the common default in ``self.builder.op_table``
            strict_locus: iff False, for non-symmetric implementations of symmetric ops the locus order may
                be changed if no calibration data is available for the requested locus order

        Returns:
            Calibrated gate implementation.

        """
        impl_name = impl_name or self._default_implementations.get(op_name, None)
        composite_calibration = None
        if op_name in self.registered_gates:
            # if op is registered and needs calibration, find the cal data under the CompositeGate
            impl_class = self.builder.get_implementation_class(op_name, impl_name)
            if impl_class.needs_calibration() or (
                isinstance(impl_class, CompositeGate) and impl_class.registered_gates
            ):
                try:
                    composite_calibration = self.builder.get_calibration(self.parent.name, self.name, self.locus)
                except ValueError:
                    pass
        return self.builder.get_implementation(
            op_name,
            locus,
            impl_name=impl_name,
            strict_locus=strict_locus,
            priority_calibration=composite_calibration,
            use_priority_order=True,
        )


class CompositeCache:
    """Cache for CompositeGate TimeBoxes.

    Result from :meth:`.CompositeGate.__call__`` (or other methods returning a TimeBox) cannot be stored in the normal
    cache ``GateImplementation._timebox_cache`` as composites can include any gates in their calls, and we cannot trust
    that the cache is flushed correctly just based on if the composite itself has its own calibration data changed
    (we would have to flush also when any of the composite's members get new calibration, and this cannot consistently
    be deduced). For this reason, CompositeCache will be flushed whenever ANY gate implementation gets new calibration
    data.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], TimeBox] = {}

    def set(
        self, gate_implementation: GateImplementation, cache_key: tuple[Any, ...], timebox: TimeBox, extra_id: str = ""
    ) -> None:
        """Set a TimeBox into the cache.

        Args:
            gate_implementation: gate implementation that created the TimeBox.
            cache_key: hashable key identifying the TimeBox (usually the :meth:`.CompositeGate.__call__` arguments cast
                into a tuple).
            timebox: TimeBox that will be added to the cache.
            extra_id: extra string id for further identifying the result if needed (for example if the TimeBox did not
                come from the call method, but some other method, this could be the method's name).

        """
        self._cache[(gate_implementation.id, gate_implementation.locus, extra_id, cache_key)] = timebox

    def get(
        self, gate_implementation: GateImplementation, cache_key: tuple[Any, ...], extra_id: str = ""
    ) -> TimeBox | None:
        """Get a TimeBox from the cache.

        Args:
            gate_implementation: gate implementation that created the TimeBox.
            cache_key: hashable key identifying the TimeBox (usually the :meth:`.CompositeGate.__call__` arguments cast
                into a tuple).
            extra_id: extra string id for further identifying the result (for example if the TimeBox did not come
                from the call method, but some other method, this could be the method's name).

        Returns:
            The cached TimeBox or None if not fund for this ``gate_implementation``, ``cache_key``, and ``extra_id``.

        """
        return self._cache.get((gate_implementation.id, gate_implementation.locus, extra_id, cache_key))

    def flush(self) -> None:
        """Flush the CompositeCache."""
        self._cache = {}
