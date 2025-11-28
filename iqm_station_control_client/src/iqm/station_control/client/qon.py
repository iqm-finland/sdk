# Copyright 2025 IQM
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
"""Qualified observation name parsing and creation."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
import logging
from typing import Annotated, Any, Final, TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass

from iqm.station_control.interface.models import ObservationBase

logger = logging.getLogger(__name__)


_FIELD_SEPARATOR: Final[str] = "."
"""Separates fields/settings tree path elements in an Observation dut_field."""

_SUFFIX_SEPARATOR: Final[str] = ":"
"""Separates suffixes in an Observation dut_field."""

LOCUS_SEPARATOR: Final[str] = "__"
"""Separates QPU components in a locus string."""

_PATTERN: Final[str] = r"^[A-Za-z_][A-Za-z0-9_]*$"
"""Regex that most observation name parts must follow."""

Locus: TypeAlias = tuple[str, ...]
"""Sequence of QPU component physical names a quantum operation acts on. The order may matter."""

Suffixes: TypeAlias = dict[str, str]
"""Suffixes in a dut_field, split into key/value pairs."""


def locus_to_locus_str(locus: Iterable[str]) -> str:
    """Convert a locus into a locus string."""
    return LOCUS_SEPARATOR.join(locus)


def locus_str_to_locus(locus_str: str) -> Locus:
    """Convert a locus string into a locus."""
    return tuple(locus_str.split(LOCUS_SEPARATOR))


class Domain(StrEnum):
    """Known observation domains/categories."""

    CONTROLLER_SETTING = "controllers"
    """Settings for the control instruments. Calibration data."""
    GATE_PARAMETER = "gates"
    """Parameters for quantum operations. Calibration data."""
    CHARACTERIZATION = "characterization"
    """Characterization data for the QPU."""
    METRIC = "metrics"
    """Quality metrics for quantum operations."""


class UnknownObservationError(RuntimeError):
    """Observation name was syntactically correct but contained unknown elements."""


def _parse_suffixes(suffixes: Iterable[str]) -> Suffixes:
    """Parse the given suffixes and return them in a sorted dictionary."""
    suffix_dict = {}
    for suffix in suffixes:
        if "=" not in suffix:
            raise ValueError(f"Invalid suffix: {suffix}")
        key, value = suffix.split("=")
        suffix_dict[key] = value
    # We're sorting the suffixes to ensure that the suffixes are always in the same order
    # (they are supposed to be in lexical order according to the key)
    return dict(sorted(suffix_dict.items()))


@dataclass(frozen=True, config={"extra": "forbid"})  # do not allow unknown keyword args
class QON:
    """Qualified observation name.

    Used for representing, creating and parsing observation names that conform to the current convention.
    When the convention changes, the first thing you should do is to update the classes
    in this module.

    .. note::

       This class provides a somewhat reliable way to encode more than one data item in
       :attr:`ObservationBase.dut_field`, aka "observation name". Eventually a more viable
       solution could be to give each of these items their own fields in the observation structure.

    """

    @property
    def domain(self) -> Domain:
        """Type/purpose of the observation."""
        raise NotImplementedError

    def __str__(self) -> str:
        """String representation of the qualified observation name."""
        raise NotImplementedError

    @classmethod
    def from_str(cls, name: str) -> QON:
        """Parse an observation name into a QON object.

        Args:
            name: Observation name (aka dut_field) to parse.

        Returns:
            Corresponding qualified observation name object.

        Raises:
            ValueError: Failed to parse ``name`` because it was syntactically incorrect.
            UnknownObservationError: Failed to parse ``name`` because it contains unknown elements.

        """
        parts = name.split(_FIELD_SEPARATOR, maxsplit=2)
        if len(parts) < 3:
            raise ValueError("Unparseable observation name.")

        # check the category/domain of the observation
        match parts[0]:
            case Domain.METRIC:
                return QONMetric._parse(parts[1], parts[2])
            case Domain.CHARACTERIZATION if parts[1] == "model":
                return QONCharacterization._parse(parts[2])
            case Domain.CONTROLLER_SETTING:
                return QONControllerSetting._parse(parts[1], parts[2])
            case Domain.GATE_PARAMETER:
                return QONGateParam._parse(parts[1], parts[2])

        raise UnknownObservationError("Unknown observation domain.")


@dataclass(frozen=True)
class QONCharacterization(QON):
    """QON representing a QPU property.

    Has the form ``characterization.model.{component}.{property}``

    Can parse e.g.

    characterization.model.QB5.t2_time

        component: QB5
        quantity: t2_time
    """

    component: str
    """Names of QPU component(s) that the observation describes."""
    quantity: str
    """Name of the quantity described by the observation."""

    @property
    def domain(self) -> Domain:
        return Domain.CHARACTERIZATION

    def __str__(self) -> str:
        return _FIELD_SEPARATOR.join([self.domain, "model", self.component, self.quantity])

    @classmethod
    def _parse(cls, rest: str) -> QONCharacterization:
        """Parse a characterization observation name."""
        parts = rest.split(_FIELD_SEPARATOR, maxsplit=1)
        if len(parts) < 2:
            raise ValueError("characterization.model observation name has less than 4 parts")

        component, quantity = parts
        return cls(
            component=component,
            quantity=quantity,
        )


class QONMetricRegistry:
    """Registry for QONMetric subclasses, mapping method names to parser classes.

    Allows extensible registration of new metric parsing strategies for new methods.
    """

    _registry: dict[str, type[QONMetric]] = {}
    """Mapping from QONMetric method name to the subclass that handles it."""
    _inv_registry: dict[type[QONMetric], frozenset[str]] = {}
    """Mapping from QONMetric subclass to the methods it handles."""

    @classmethod
    def register(cls, method_names: Iterable[str]):
        """Decorator for registering a QONMetric subclass as the parser for one or more method names.

        Args:
            method_names: One or more method names to associate with the subclass.

        Returns:
            The decorator function.

        """

        def decorator(subclass: type[QONMetric]):  # noqa: ANN202
            for name in method_names:
                if (owner := cls._registry.get(name)) is not None:
                    raise ValueError(f"Method {name} already registered to {owner.__name__}")
                cls._registry[name] = subclass
            cls._inv_registry[subclass] = frozenset(method_names)
            return subclass

        return decorator

    @classmethod
    def get_parser(cls, method_name: str) -> type[QONMetric]:
        """Retrieve the parser class for a given method name.

        Args:
            method_name: The method name to look up.

        Returns:
            The QONMetric subclass registered for the method.

        Raises:
            UnknownObservationError: ``method_name`` is not registered.

        """
        try:
            return cls._registry[method_name]
        except KeyError:
            raise UnknownObservationError("Unknown quality metric.")

    @classmethod
    def registered_methods(cls) -> list[str]:
        """Get a list of all registered method names.

        Returns:
            List of registered method names.

        """
        return list(cls._registry.keys())


@dataclass(frozen=True)
class QONMetric(QON):
    """Base class for QON representing a gate quality metric.

    Subclasses implement parsing and string representation for specific methods.
    """

    method: Annotated[str, Field(pattern=_PATTERN)]
    locus_str: str
    """Sequence of names of QPU components on which the gate is applied, or on which the experiment is run,
    encoded into a string."""
    metric: Annotated[str, Field(pattern=r"^[A-Za-z_0-9][A-Za-z0-9_]*$")]
    """Measured metric."""

    def __post_init__(self):
        if self.method not in QONMetricRegistry._inv_registry[self.__class__]:
            raise UnknownObservationError(
                f"{self.__class__.__name__} is not registered to handle the method {self.method}"
            )

    @property
    def domain(self) -> Domain:
        """Return the QON domain for metrics."""
        return Domain.METRIC

    @property
    def locus(self) -> Locus:
        """Locus of the metric."""
        return locus_str_to_locus(self.locus_str)

    @classmethod
    def _parse(cls, method: str, method_specific_part: str) -> QONMetric:
        """Parse a metric observation name using the appropriate registered subclass.

        Args:
            method: The method name.
            method_specific_part: The method-specific part of the metric observation name.

        Returns:
            Parsed metric name.

        Raises:
            UnknownObservationError: ``method`` is not registered.

        """
        parser_cls = QONMetricRegistry.get_parser(method)
        return parser_cls._parse(method, method_specific_part)


@QONMetricRegistry.register(
    [
        "rb",
        "irb",
        "ssro",
        "restless_ssro",
        "rrc",
        "qndness",
        "msmt_qndness",
        "npopex_echo",
        "npopex_long",
        "coherence_cz",
        "ncz_swap",
        "entanglement_coherence",
        "conditional_reset",
    ]
)
@dataclass(frozen=True)
class QONGateMetric(QONMetric):
    """QON representing a gate quality metric.

    Has the form ``metrics.{method}.{method_specific_part}``.

    Can parse/represent e.g. the following metrics:

    ``metrics.ssro.measure.constant.QB1.fidelity:par=d1:aaa=bbb``

        method: ssro
        gate: measure
        implementation: constant
        locus_str: QB1
        metric: fidelity
        suffixes: {"aaa": "bbb", "par": "d1"}

    ``metrics.ssro.measure.constant.QB1.fidelity``

        method: ssro
        gate: measure
        implementation: constant
        locus_str: QB1
        metric: fidelity
        suffixes: {}

    ``metrics.rb.prx.drag_crf.QB4.fidelity:par=d2``

        method: rb
        gate: prx
        implementation: drag_crf
        locus_str: QB4
        metric: fidelity
        suffixes: {"par": "d2"}
    """

    gate: Annotated[str, Field(pattern=_PATTERN)]
    """Name of the gate/quantum operation."""
    implementation: Annotated[str, Field(pattern=_PATTERN)]
    """Name of the gate implementation."""
    suffixes: Suffixes = Field(default_factory=dict)
    """Suffixes defining the metric further (if any)."""

    def __str__(self) -> str:
        parts = [self.domain, self.method, self.gate, self.implementation, self.locus_str, self.metric]
        name = _FIELD_SEPARATOR.join(parts)
        if self.suffixes:
            suffix_str = _SUFFIX_SEPARATOR.join(f"{k}={v}" for k, v in sorted(self.suffixes.items()))
            return f"{name}:{suffix_str}"
        return name

    @classmethod
    def _parse(cls, method: str, method_specific_part: str) -> QONGateMetric:
        """Parse a metric observation name that includes a gate and an implementation.

        Args:
            method: The method name.
            method_specific_part: The method-specific part of the metric observation name.

        Returns:
            Parsed metric name.

        Raises:
            ValueError: Observation name is malformed.

        """
        parts, suffixes = _split_obs_name(method_specific_part, maxsplit=3)
        if len(parts) < 4:
            raise ValueError(f"{method} gate quality metric name has less than 6 parts")
        gate, implementation, locus_str, metric = parts
        return cls(
            method=method,
            gate=gate,
            implementation=implementation,
            locus_str=locus_str,
            metric=metric,
            suffixes=suffixes,
        )


@QONMetricRegistry.register(["ghz_state", "coherence_gef", "relaxation_ef", "simultaneous_coherence_gef", "t1_coupled"])
@dataclass(frozen=True)
class QONSystemMetric(QONMetric):
    """QON representing a system quality metric.

    Has the form ``metrics.{method}.{locus_str}.{metric}``.

    Can parse/represent e.g. the following metrics:

    ``metrics.irb.circuit.random_circuit.QB4.fidelity:par=d2``

        method: irb
        gate: circuit
        implementation: random_circuit
        locus_str: QB4
        metric: fidelity
        suffixes: {"par": "d2"}

    ``metrics.ghz_state.QB1__QB2.coherence_lower_bound``

        method: ghz_state
        locus_str: QB1__QB2
        metric: coherence_lower_bound

    Subclasses implement parsing and string representation for specific methods.
    """

    def __str__(self) -> str:
        parts = [self.domain, self.method, self.locus_str, self.metric]
        return _FIELD_SEPARATOR.join(parts)

    @classmethod
    def _parse(cls, method: str, method_specific_part: str) -> QONSystemMetric:
        """Parse a metric observation name without a gate or an implementation.

        Args:
            method: The method name.
            method_specific_part: The method-specific part of the metric observation name.

        Returns:
            Parsed metric name.

        Raises:
            ValueError: Observation name is malformed.

        """
        parts = method_specific_part.split(_FIELD_SEPARATOR, maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"{method} system quality metric name has less than 4 parts")
        locus_str, metric = parts
        return cls(
            method=method,
            locus_str=locus_str,
            metric=metric,
        )


@dataclass(frozen=True)
class QONControllerSetting(QON):
    """QON representing a controller setting observation.

    Has the form ``controllers.{controller}[.{subcontroller}]*.{setting}``.
    """

    controller: Annotated[str, Field(pattern=_PATTERN)]
    """Name of the controller."""
    rest: str
    """Possible subcontroller names in a dotted structure, ending in the setting name."""

    @property
    def domain(self) -> Domain:
        return Domain.CONTROLLER_SETTING

    def __str__(self) -> str:
        return _FIELD_SEPARATOR.join([self.domain, self.controller, self.rest])

    @classmethod
    def _parse(cls, controller: str, controller_specific_part: str) -> QONControllerSetting:
        """Parse a controller setting observation name."""
        return cls(
            controller=controller,
            rest=controller_specific_part,
        )


@dataclass(frozen=True)
class QONGateParam(QON):
    """QON representing a gate parameter observation.

    Has the form ``gates.{gate}.{implementation}.{locus_str}.{parameter}``.
    """

    gate: Annotated[str, Field(pattern=_PATTERN)]
    """Name of the gate/quantum operation."""
    implementation: Annotated[str, Field(pattern=_PATTERN)]
    """Name of the gate implementation."""
    locus_str: str
    """Sequence of names of QPU components on which the gate is applied, encoded into a string."""
    parameter: str
    """Name of the gate parameter. May have further dotted structure."""

    @property
    def domain(self) -> Domain:
        return Domain.GATE_PARAMETER

    @property
    def locus(self) -> Locus:
        """Locus of the gate parameter."""
        return locus_str_to_locus(self.locus_str)

    def __str__(self) -> str:
        return _FIELD_SEPARATOR.join([self.domain, self.gate, self.implementation, self.locus_str, self.parameter])

    @classmethod
    def _parse(cls, gate: str, rest: str) -> QONGateParam:
        """Parse a gate parameter observation name."""
        parts = rest.split(_FIELD_SEPARATOR, maxsplit=2)
        if len(parts) < 3:
            raise ValueError("Gate parameter observation name has less than 5 parts")
        implementation, locus_str, param = parts
        return cls(
            gate=gate,
            implementation=implementation,
            locus_str=locus_str,
            parameter=param,
        )


def _split_obs_name(obs_name: str, *, maxsplit: int = -1) -> tuple[list[str], Suffixes]:
    """Split the given observation name into path elements and suffixes."""
    # some observation names may have suffixes, split them off
    fragments = obs_name.split(_SUFFIX_SEPARATOR)
    suffixes = _parse_suffixes(fragments[1:])

    # split the path elements
    path = fragments[0].split(_FIELD_SEPARATOR, maxsplit=maxsplit)
    return path, suffixes


def _convert_to_float(value: Any) -> float:
    """Attempt to convert the given value to float.

    Raises:
        ValueError: Conversion not possible.

    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert value to float: {type(value)}") from e


GATE_FIDELITY_METHODS = {
    "prx": "rb",
    "measure": "ssro",
}
"""Mapping from quantum operation name to the standard methods for obtaining its fidelity.
The default is "irb" for ops not mentioned here."""


class ObservationFinder(dict):
    """Query structure for a set of observations.

    This class enables reasonably efficient filtering of an observation set based on the observation
    name elements (e.g. find all T1 times / parameters of a particular gate/impl/locus etc. in the set).

    The class has utility methods for querying specific types observations. The idea is to keep
    all the logic related to the structure of the observation names encapsulated in this class/module.

    Currently implemented using a nested dictionary that follows the dotted structure of the observation names.
    The nested dictionary is not ideal for all searches/filterings, but it's just an implementation detail that
    can be improved later on without affecting the public API of this class.

    Args:
        observations: Observations to include in the query structure.
        skip_unparseable: If True, ignore any observation whose name cannot be parsed, otherwise
            raise an exception.

    """

    def __init__(self, observations: Iterable[ObservationBase], skip_unparseable: bool = False):
        def parse_observation_into_dict(name: str, dictionary: dict[str, Any]) -> tuple[dict[str, Any], str, Suffixes]:
            """Help insert the given observation name, split into path elements, into a nested dictionary.

            The returned values allow the caller to insert whatever they want under the last path element
            of ``name`` in the nested dict.

            Args:
                name: Observation name (aka dut_field) to be split into path elements.
                dictionary: Nested dictionary in which the path elements of ``name`` are inserted.

            Returns:
                The dict corresponding to the second-last path element of ``name``, last path element of ``name``,
                suffixes in ``name``.

            Raises:
                ValueError: Failed to parse ``name`` because it was syntactically incorrect.
                UnknownObservationError: Failed to parse ``name`` because it contains unknown elements.

            """
            path, suffixes = _split_obs_name(name)
            # check the category/domain of the observation
            match path[0]:
                case Domain.METRIC:
                    if len(path) < 4:
                        raise ValueError("Quality metric observation name has less than 4 parts.")
                case Domain.CHARACTERIZATION:
                    if len(path) < 4:
                        raise ValueError("Characterization observation name has less than 4 parts.")
                case Domain.CONTROLLER_SETTING:
                    if len(path) < 3:
                        raise ValueError("Controller setting observation name has less than 3 parts.")
                case Domain.GATE_PARAMETER:
                    if len(path) < 5:
                        raise ValueError("Gate parameter observation name has less than 5 parts.")
                case _:
                    raise UnknownObservationError("Unknown observation domain.")
            for path_element in path[:-1]:
                dictionary = dictionary.setdefault(path_element, {})
            return dictionary, path[-1], suffixes

        for obs in observations:
            try:
                last_dict, last_element, _ = parse_observation_into_dict(obs.dut_field, self)
                # TODO how to handle suffixes?
                if (existing_obs := last_dict.get(last_element)) is not None:
                    logger.warning(
                        "Repeated observations: using %s, ignoring %s",
                        existing_obs.dut_field,
                        obs.dut_field,
                    )
                else:
                    last_dict[last_element] = obs
            except (ValueError, UnknownObservationError) as err:
                message = f"{obs.dut_field}: {err}"
                if skip_unparseable:
                    logger.warning(message)
                else:
                    raise err.__class__(message)

    def _build_dict(self, pre_path: Iterable[str], keys: Iterable[str], post_path: Iterable[str]) -> dict[str, float]:
        """Get the same property for multiple path elements, if it exists.

        Follows ``pre_path`` to a base node, then for every item in ``keys`` follows ``[key] + post_path``
        and gets the corresponding value.

        Args:
            pre_path: Initial path in the tree to the base node.
            keys: Path elements under the base node to retrieve.
            post_path: Final path to follow for each of ``keys``.

        Returns:
            Mapping from ``keys`` to the corresponding values. If a key is missing or
            ``[key] + post_path`` could not be followed, that particular key does not appear in the mapping.

        Raises:
            KeyError: Could not follow ``pre_path``.

        """
        base_node: dict[str, Any] = self
        for step in pre_path:
            next_node = base_node.get(step)
            if not next_node or not isinstance(next_node, dict):
                raise KeyError(f"pre_path step {step} could not be found")
            base_node = next_node

        result = {}
        for key in keys:
            node: dict[str, Any] | ObservationBase | None = base_node.get(key)
            if node:
                for step in post_path:
                    if isinstance(node, ObservationBase):
                        break  # skip this key
                    if isinstance(node, dict):
                        node = node.get(step)
                        if not node:
                            break  # skip this key
                    else:
                        break  # skip this key
                else:
                    if isinstance(node, ObservationBase):
                        try:
                            result[key] = float(node.value)  # type: ignore[arg-type]
                        except (ValueError, TypeError):
                            pass  # skip this key if conversion fails
        return result

    def _follow_path(self, path: Iterable[str]) -> dict[str, Any] | ObservationBase:
        """Follow ``path``, return the final node/value."""
        node: dict[str, Any] | ObservationBase = self
        for step in path:
            if isinstance(node, ObservationBase):
                raise KeyError(f"path step '{step}' could not be found")
            next_node = node.get(step)
            if next_node is None:
                raise KeyError(f"path step '{step}' could not be found")
            node = next_node
        return node

    def _get_path_value(self, path: Iterable[str]) -> float:
        """Follow ``path``, return the final value."""
        node = self._follow_path(path)
        if not isinstance(node, ObservationBase):
            raise KeyError(f"path {path} does not end in an observation")
        return _convert_to_float(node.value)

    def _get_path_node(self, path: Iterable[str]) -> dict[str, Any]:
        """Follow ``path``, return the final node."""
        node = self._follow_path(path)
        if isinstance(node, ObservationBase):
            raise KeyError(f"path {path} does not end in a node")
        return node

    def get_coherence_times(self, components: Iterable[str]) -> tuple[dict[str, float], dict[str, float]]:
        """T1 and T2 coherence times for the given QPU components.

        If not found, the component will not appear in the corresponding dict.
        """
        try:
            t1 = self._build_dict(["characterization", "model"], components, ["t1_time"])
            t2 = self._build_dict(["characterization", "model"], components, ["t2_time"])
        except KeyError as exc:
            logger.warning("Missing characterization.model data: %s", exc)
            return {}, {}

        return t1, t2

    def get_gate_duration(self, gate_name: str, impl_name: str, locus: Locus) -> float | None:
        """Duration for the given gate/implementation/locus (in s), or None if not found."""
        locus_str = locus_to_locus_str(locus)
        try:
            return self._get_path_value(["gates", gate_name, impl_name, locus_str, "duration"])
        except KeyError:
            logger.warning("Missing duration for %s.%s.%s", gate_name, impl_name, locus_str)
            return None

    def get_gate_fidelity(self, gate_name: str, impl_name: str, locus: Locus) -> float | None:
        """Fidelity of the given gate/implementation/locus, or None if not found."""
        # irb is the default method
        method = GATE_FIDELITY_METHODS.get(gate_name, "irb")
        locus_str = locus_to_locus_str(locus)
        try:
            return self._get_path_value(["metrics", method, gate_name, impl_name, locus_str, "fidelity"])
        except KeyError:
            logger.warning("Missing fidelity for %s.%s.%s", gate_name, impl_name, locus_str)
            return None

    def get_measure_errors(self, gate_name: str, impl_name: str, locus: Locus) -> tuple[float, float] | None:
        """Measurement errors of the given gate/implementation/locus, or None if not found."""
        locus_str = locus_to_locus_str(locus)
        try:
            node = self._get_path_node(["metrics", "ssro", gate_name, impl_name, locus_str])
            error_0_to_1 = node["error_0_to_1"].value
            error_1_to_0 = node["error_1_to_0"].value
            return _convert_to_float(error_0_to_1), _convert_to_float(error_1_to_0)
        except KeyError:
            logger.warning("Missing errors for %s.%s.%s.", gate_name, impl_name, locus_str)
            return None

    def get_qubit_frequency(self, qubit: str) -> float | None:
        """Qubit drive frequency, or None if not found."""
        try:
            return self._get_path_value(["controllers", qubit, "drive", "frequency"])
        except KeyError:
            logger.warning(f"Missing drive frequency for {qubit}.")
            return None
