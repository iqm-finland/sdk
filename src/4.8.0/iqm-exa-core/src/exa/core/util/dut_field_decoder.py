#  *****************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  *****************************************************************************
"""Decode DUT fields into components and observation parameters."""

from collections.abc import Iterable
import re

from exa.common.errors import ValidationError
from exa.common.qcm_data import ChipTopology

_QBS = ["QB[0-9]*", "QB[0-9]+_QB[0-9]+"]
_TCS = ["TC[0-9]*", "TC-[0-9]+-[0-9]+"]
_PLS = ["PL-[0-9]*", "PL_RO-[0-9]*"]
_CRS = ["COMPR[0-9]*", "COMP_R[0-9]*"]


_QON_PREFIXES = ["characterization.model", "gates", "metrics"]
"""DUT field prefixes for qualified observation names."""


_STANDARD_PREFIXES = ["controllers", "characterization"]
"""DUT field prefixes to be removed for the decoding. The prefixes are in priority order, i.e. the first one in this
list that matches the dut_field's prefix, will be removed."""


def _remove_prefix(dut_field: str, prefixes: Iterable[str]) -> tuple[list[str], str]:
    """Remove the first found prefix from the fragments of the DUT field.

    Args:
        dut_field: The DUT field to be split into fragments by `.`.
        prefixes: The first prefix that matches the DUT field will be removed from the fragments.

    Returns:
        The fragments of the DUT field without the prefix if applicable and the prefix.

    """
    fragments = dut_field.split(".")
    for prefix in prefixes:
        if dut_field.startswith(prefix):
            return fragments[len(prefix.split(".")) :], prefix
    return fragments, ""


def _component_on_chip(locus_str: str, chip_topology: ChipTopology) -> str:
    """Get the component on the chip this locus refers to.

    Args:
        locus_str: The locus that should refer to a component on the chip.
        chip_topology: The chip topology to retrieve the component from.

    Returns:
        The chip component.

    Raises:
        ValidationError if locus cannot be resolved.

    """
    if locus_str in chip_topology.all_components:
        return locus_str
    if len(components := locus_str.split("__")) == 2:
        return chip_topology.get_coupler_for(*components)
    if len(components) == 3 and components[1].startswith("COMP"):
        # Special case for `QB1__COMPR2__QB3`
        return chip_topology.get_coupler_for(components[0], components[1])
    if len(components) > 2:
        raise ValidationError(
            f"Can't handle more than two components {components}. "
            "Suggest to add method to handle this case to the initializer."
        )
    raise ValidationError(f"Locus {locus_str} is neither a component on the chip nor a coupled pair of components.")


def _resolve_locus_string(locus_str: str) -> str:
    """Get the component on the chip this locus refers to using heuristics.

    Args:
        locus_str: The locus that should refer to a component on the chip.

    Returns:
        The chip component.

    Raises:
        ValueError if locus cannot be resolved.

    """
    if any(re.fullmatch(pattern, locus_str) for pattern in [*_QBS, *_TCS, *_PLS, *_CRS]):
        return locus_str
    if len(components := locus_str.split("__")) == 2:
        if components[0].startswith("QB") and components[1].startswith("COMP"):
            # On star QPUs the couplers are named with the qubit index only, e.g. TC3. Note, that the
            # qubit component is always the first element in the components list.
            return "".join(["TC", re.sub("[a-zA-Z]", "", components[0])])
        if components[0].startswith("QB") and components[1].startswith("QB"):
            # Map two components to their connecting coupler
            return "-".join(["TC", *sorted(re.sub("[a-zA-Z]", "", locus_str).split("__"), key=int)])
    if (
        len(components) == 3
        and components[0].startswith("QB")
        and components[1].startswith("COMP")
        and components[2].startswith("QB")
    ):
        return "".join(["TC", re.sub("[a-zA-Z]", "", components[0])])
    if len(components) > 2:
        raise ValueError(
            f"Can't handle more than two components {components}. "
            "Suggest to add method to handle this case to the initializer."
        )

    raise ValueError(f"Locus {locus_str} is neither a component on the chip nor a coupled pair of components.")


def _get_locus_from_qon(dut_field: str) -> tuple[str, int | None]:
    """Get locus and index of locus in fragments without prefix from QON DUT field."""
    # This will be replaced by parsing DUT fields through QON in the future
    fragments, prefix = _remove_prefix(dut_field, [*_QON_PREFIXES])
    if prefix == "characterization.model" and len(fragments) >= 2:
        return fragments[0], 0
    if prefix == "metrics" and len(fragments) >= 5:
        return fragments[3], 3
    if prefix == "gates" and len(fragments) >= 4:
        return fragments[2], 2
    return "", None


def decode_dut_field(dut_field: str, chip_topology: ChipTopology | None = None) -> tuple[str, str]:  # noqa: PLR0912
    """Decode a DUT field into component and parameter names for QPUPlots and GBC.

    DUT fields must follow the convention `{prefix}.{locus_str}.{suffix}`, where
    prefix is optional and can have more than one `.` separated entry and
    suffix can have more than one `.` separated entry
    Optionally the chip topology can be provided rather than using hardcoded string comparison.
    The `locus_str` must be resolvable into a component on the chip, that is, a qubit, a tunable coupler,
    a computational resonator, or a probe line. Couplers can be deducted from qubit-qubit pairs and
    qubit-computational resonator pairs.

    Args:
        dut_field: The DUT field that should contain a component and a quantity to plot.
        chip_topology: The chip topology is used to check compatibility of component names and to
            determine the coupler name from loci.

    Returns:
        The pair of component and quantity name.

    Raises:
        ValidationError: DUT field can't be interpreted.

    """
    msg = (
        f"DUT field '{dut_field}' can't be decoded into component and parameter. "
        f"If you think that this is a mistake please provide your own DUT field decoder."
    )
    locus_str, idx = _get_locus_from_qon(dut_field)
    fragments, prefix = _remove_prefix(dut_field, [*_QON_PREFIXES, *_STANDARD_PREFIXES])

    if len(fragments) < 2:
        raise ValidationError(msg)

    # Check for legacy names that start with a component
    if not locus_str and fragments[0].startswith(("QB", "TC", "COMP", "PL")):
        locus_str, idx = fragments[0], 0

    # Remove second level prefix
    if not locus_str and fragments[0] in ["metrics", "gates", "gate_properties"]:
        fragments, prefix = _remove_prefix(".".join(fragments), ["metrics", "gates", "gate_properties"])

    if len(fragments) < 2:
        raise ValidationError(msg)

    if not locus_str and prefix in ["gate_properties", *_QON_PREFIXES, "controllers"]:
        for i, fragment in enumerate(fragments[:-1]):
            if fragment.startswith(("QB", "TC", "COMP", "PL")):
                locus_str, idx = fragment, i
                break

    # Get component and build quantity from remaining fragments
    if locus_str and idx is not None:
        # Allow infix for special three component case `QB1__COMPR__QB2`
        infix = components[2] if len(components := locus_str.split("__")) == 3 else ""
        # Get component from chip topology
        if chip_topology:
            return _component_on_chip(locus_str, chip_topology), ".".join(
                filter(None, [*fragments[:idx], infix, *fragments[idx + 1 :]])
            )
        # Get component from hardcoded entries
        return (
            _resolve_locus_string(locus_str),
            ".".join(filter(None, [*fragments[:idx], infix, *fragments[idx + 1 :]])),
        )
    raise ValidationError(msg)
