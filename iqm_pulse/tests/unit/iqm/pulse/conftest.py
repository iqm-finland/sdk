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
from collections.abc import Iterable
import json
from pathlib import Path

import pytest

from exa.common.qcm_data.chad_model import CHAD
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.builder import ScheduleBuilder, load_config
from iqm.pulse.gate_implementation import OpCalibrationDataTree
from iqm.pulse.playlist.channel import FAST_FEEDBACK_CHANNEL_TEMPLATE, ChannelProperties, ProbeChannelProperties
from iqm.pulse.playlist.instructions import IQPulse, RealPulse, VirtualRZ, Wait
from iqm.pulse.quantum_ops import QuantumOpTable


@pytest.fixture
def config() -> tuple[QuantumOpTable, OpCalibrationDataTree]:
    """Op definitions and calibration data."""
    exa_yml_path = Path(__file__).parents[3] / "resources/experiment.yml"
    return load_config(str(exa_yml_path))


@pytest.fixture
def config_uhfqa() -> tuple[QuantumOpTable, OpCalibrationDataTree]:
    """Op definitions and calibration data for station with mixed instruments and
    multiple sample rates (UHFQA readout, SHFSG drive)."""
    exa_yml_path = Path(__file__).parents[3] / "resources/experiment_uhfqa.yml"
    return load_config(str(exa_yml_path))


@pytest.fixture
def config_star() -> tuple[QuantumOpTable, OpCalibrationDataTree]:
    """Op definitions and calibration data for a star station."""
    exa_yml_path = Path(__file__).parents[3] / "resources/experiment_star.yml"
    return load_config(str(exa_yml_path))


@pytest.fixture
def config_with_composite() -> tuple[QuantumOpTable, OpCalibrationDataTree]:
    """Op definitions and calibration data with a composite gate."""
    exa_yml_path = Path(__file__).parents[3] / "resources/composite_hadamard.yml"
    return load_config(str(exa_yml_path))


@pytest.fixture(scope="module")
def chad() -> CHAD:
    """Mocks the process of requesting a CHAD from Station Control."""
    chad_path = Path(__file__).parents[3] / "resources/crystal_20.json"
    with open(chad_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CHAD(**data)


@pytest.fixture(scope="module")
def chad_star() -> CHAD:
    """Mocks the process of requesting a CHAD from Station Control."""
    chad_path = Path(__file__).parents[3] / "resources/star_7.json"
    with open(chad_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CHAD(**data)


@pytest.fixture(scope="module")
def chip_topology_crystal_20() -> ChipTopology:
    """ChipTopology as if it was constructed from chip design record from Station Control."""
    chad_path = Path(__file__).parents[3] / "resources/crystal_20.json"
    with open(chad_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    return ChipTopology.from_chip_design_record(record)


@pytest.fixture(scope="module")
def chip_topology_star() -> ChipTopology:
    """ChipTopology as if it was constructed from chip design record from Station Control."""
    chad_path = Path(__file__).parents[3] / "resources/star_7.json"
    with open(chad_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    return ChipTopology.from_chip_design_record(record)


def _build_channel_props(
    qubits: Iterable[str],
    couplers: Iterable[str],
    probe_lines: Iterable[str],
    resonators: Iterable[str] = (),
    feedback_links: Iterable[tuple[str, str]] = (),
    uhfqa_sample_rate: float = 2.0e9,
    granularity: int = 16,
) -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
    """Mocks the process of requesting channel information from Station Control."""

    allowed_real = (Wait, RealPulse)
    allowed_iq = (Wait, IQPulse, VirtualRZ)
    HDAWG = ChannelProperties(2.0e9, granularity, 32, allowed_real)
    SHFSG = ChannelProperties(2.0e9, granularity, 32, allowed_iq, is_iq=True)
    UHFQA = ProbeChannelProperties(
        uhfqa_sample_rate,
        16,
        32,
        allowed_iq,
        is_iq=True,
        center_frequency=4.9e9,
        integration_start_dead_time=16,
        integration_stop_dead_time=16,
    )
    VIRTUAL_MOVE = ChannelProperties(2.0e9, 1, 0, allowed_iq, is_iq=True, is_virtual=True)
    VIRTUAL_FEEDBACK = ChannelProperties(2.0e9, 1, 0, allowed_iq, is_iq=True, is_virtual=True, blocks_component=False)
    VIRTUAL_UHFQA = ProbeChannelProperties(
        uhfqa_sample_rate,
        16,
        32,
        allowed_iq,
        is_iq=True,
        center_frequency=4.9e9,
        integration_start_dead_time=16,
        integration_stop_dead_time=16,
        is_virtual=True,
        blocks_component=False,
    )

    props = {}
    comp_channels = {}
    for q in qubits:
        props.update(
            {
                f"{q}__drive.awg": SHFSG,
                f"{q}__flux.awg": HDAWG,
            }
        )
        comp_channels[q] = {"drive": f"{q}__drive.awg", "flux": f"{q}__flux.awg"}
    for c in couplers:
        props.update(
            {
                f"{c}__flux.awg": HDAWG,
            }
        )
        comp_channels[c] = {"flux": f"{c}__flux.awg"}
    for r in resonators:
        props.update(
            {
                f"{r}__drive_virtual": VIRTUAL_MOVE,
            }
        )
        comp_channels[r] = {"drive": f"{r}__drive_virtual"}
    for p in probe_lines:
        props.update(
            {
                f"{p}__readout": UHFQA,
            }
        )
        props.update(
            {
                f"{p}__readout_virtual": VIRTUAL_UHFQA,
            }
        )
        comp_channels[p] = {"readout": f"{p}__readout", "readout_virtual": f"{p}__readout_virtual"}
        for q in qubits:
            if (p, q) in feedback_links:
                drive = f"{q}__drive.awg"
                ff_virtual_channel_name = FAST_FEEDBACK_CHANNEL_TEMPLATE.replace("{PROBE}", p).replace("{AWG}", drive)
                comp_channels[q].update({f"feedback_from_{p}": ff_virtual_channel_name})
                comp_channels[p].update({f"feedback_to_{drive}": ff_virtual_channel_name})
                props.update({ff_virtual_channel_name: VIRTUAL_FEEDBACK})
    return props, comp_channels


@pytest.fixture(scope="module")
def channel_props() -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_channel_props(
        qubits=["QB1", "QB2", "QB4", "QB5", "QB6", "QB10"],
        couplers=["TC-1-2", "TC-1-4", "TC-2-5", "TC-4-5", "TC-5-6", "TC-5-10"],
        probe_lines=["PL-A", "PL-B"],
        feedback_links=[("PL-B", "QB6"), ("PL-B", "QB5"), ("PL-A", "QB5"), ("PL-A", "QB1")],
    )


@pytest.fixture(scope="module")
def channel_props_star() -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_channel_props(
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5", "QB6"],
        couplers=["TC1", "TC2", "TC3", "TC4", "TC5", "TC6"],
        resonators=["COMP_R"],
        probe_lines=["PL"],
    )


@pytest.fixture
def schedule_builder(config, chip_topology_crystal_20, channel_props) -> ScheduleBuilder:
    """Fully initialized ScheduleBuilder for a partial Crystal 20 chip."""
    op_table, cal_data = config
    return ScheduleBuilder(op_table, cal_data, chip_topology_crystal_20, channel_props[0], channel_props[1])


@pytest.fixture
def schedule_builder_uhfqa(config_uhfqa, chip_topology_crystal_20) -> ScheduleBuilder:
    op_table, cal_data = config_uhfqa
    props, comp_channels = _build_channel_props(
        qubits=["QB1", "QB2", "QB4", "QB5", "QB6", "QB10"],
        couplers=["TC-1-2", "TC-1-4", "TC-2-5", "TC-4-5", "TC-5-6", "TC-5-10"],
        probe_lines=["PL-A", "PL-B"],
        uhfqa_sample_rate=1.8e9,
    )
    return ScheduleBuilder(op_table, cal_data, chip_topology_crystal_20, props, comp_channels)


@pytest.fixture
def schedule_builder_granularity(config_uhfqa, chip_topology_crystal_20) -> ScheduleBuilder:
    op_table, cal_data = config_uhfqa
    props, comp_channels = _build_channel_props(
        qubits=["QB1", "QB2", "QB4", "QB5", "QB6", "QB10"],
        couplers=["TC-1-2", "TC-1-4", "TC-2-5", "TC-4-5", "TC-5-6", "TC-5-10"],
        probe_lines=["PL-A", "PL-B"],
        granularity=8,
    )
    return ScheduleBuilder(op_table, cal_data, chip_topology_crystal_20, props, comp_channels)


@pytest.fixture
def schedule_builder_star(config_star, chip_topology_star, channel_props_star) -> ScheduleBuilder:
    """Fully initialized ScheduleBuilder for a Star 7 chip."""
    op_table, cal_data = config_star
    return ScheduleBuilder(op_table, cal_data, chip_topology_star, channel_props_star[0], channel_props_star[1])


@pytest.fixture
def schedule_builder_with_composite(config_with_composite, chip_topology_crystal_20, channel_props) -> ScheduleBuilder:
    """Fully initialized ScheduleBuilder for a partial Crystal 20 chip."""
    op_table, cal_data = config_with_composite
    return ScheduleBuilder(op_table, cal_data, chip_topology_crystal_20, channel_props[0], channel_props[1])
