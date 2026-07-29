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
"""Qualified observation name tests."""

import logging
import re

from pydantic import ValidationError
import pytest

from iqm.station_control.client.qon import (
    QON,
    ObservationFinder,
    QONCharacterization,
    QONControllerSetting,
    QONGateCharacterization,
    QONGateDefinition,
    QONGateMetric,
    QONGateParam,
    QONSystemMetric,
    UnknownObservationError,
    _split_obs_name,
)
from iqm.station_control.interface.models import ObservationBase


def observation(name: str, value: float | str | list[str], unit: str = "") -> ObservationBase:
    """Mock metric observation."""
    return ObservationBase(
        dut_field=name,
        value=value,
        unit=unit,
        uncertainty=None,
        invalid=False,
    )


class TestQON:
    """Test the QON class."""

    @pytest.mark.parametrize(
        "name,qon",
        [
            (
                "metrics.ssro.measure.constant.QB1.fidelity",  # basic SSRO metric
                QONGateMetric(
                    method="ssro", gate="measure", implementation="constant", locus_str="QB1", metric="fidelity"
                ),
            ),
            (
                "metrics.ssro.measure.constant.QB1.fidelity:aaa=bbb:par=d1",  # suffixes must be sorted by key
                QONGateMetric(
                    method="ssro",
                    gate="measure",
                    implementation="constant",
                    locus_str="QB1",
                    metric="fidelity",
                    suffixes={"aaa": "bbb", "par": "d1"},
                ),
            ),
            (
                "metrics.ghz_state.QB1__QB2.coherence_lower_bound",  # GHZ state metric
                QONSystemMetric(method="ghz_state", locus_str="QB1__QB2", metric="coherence_lower_bound"),
            ),
            (
                "characterization.model.QB5.t2_time",
                QONCharacterization(component="QB5", quantity="t2_time"),
            ),
            (
                "characterization.gate_properties.measure.constant.QB2.average_response_phase",
                QONGateCharacterization(
                    gate="measure", implementation="constant", locus_str="QB2", quantity="average_response_phase"
                ),
            ),
            (
                "gates.measure.constant.QB1.frequency",
                QONGateParam(
                    gate="measure",
                    implementation="constant",
                    locus_str="QB1",
                    parameter="frequency",
                ),
            ),
            (
                "gates.prx.drag_crf.QB1.duration",
                QONGateParam(
                    gate="prx",
                    implementation="drag_crf",
                    locus_str="QB1",
                    parameter="duration",
                ),
            ),
            (
                "gates.cz.tgss.QB4__QB1.qubit.full_width",
                QONGateParam(
                    gate="cz",
                    implementation="tgss",
                    locus_str="QB4__QB1",
                    parameter="qubit.full_width",
                ),
            ),
            (
                "gate_definitions.prx.default_implementation",
                QONGateDefinition(
                    gate="prx",
                    implementation=None,
                    quantity="default_implementation",
                ),
            ),
            (
                "gate_definitions.prx.drag_crf.override_default_for_loci",
                QONGateDefinition(
                    gate="prx",
                    implementation="drag_crf",
                    quantity="override_default_for_loci",
                ),
            ),
            (
                "controllers.QB1.drive.frequency",
                QONControllerSetting(
                    controller="QB1",
                    rest="drive.frequency",
                ),
            ),
        ],
    )
    def test_parse_observation_name(self, name, qon):
        """Test parsing valid observation names."""
        parsed = QON.from_str(name)
        # expected parsing result
        assert parsed == qon
        # can be converted back into the same string
        assert str(qon) == name

    @pytest.mark.parametrize(
        "name,match",
        [
            ("single_part", "Unparseable observation name."),
            ("QB1.t1_time", "Unparseable observation name."),  # no longer supported observation name
        ],
    )
    def test_unparseable_name(self, name: str, match: str):
        """Unparseable names must raise reasonable errors."""
        with pytest.raises(ValueError, match=re.escape(match)):
            QON.from_str(name)

    @pytest.mark.parametrize(
        "name,match",
        [
            ("three.part.name", "Unknown observation domain."),
            ("metrics.fake_method.rest", "Unknown quality metric."),
        ],
    )
    def test_unknown_observation_name(self, name: str, match: str):
        """Unparseable names must raise reasonable errors."""
        with pytest.raises(UnknownObservationError, match=re.escape(match)):
            QON.from_str(name)

    @pytest.mark.parametrize(
        "name,match",
        [
            (
                "metrics.ssro.constant.QB1.fidelity",  # Missing gate
                "ssro gate quality metric name has less than 6 parts",
            ),
            (
                "metrics.rb.prx.QB1.fidelity",  # Missing implementation
                "rb gate quality metric name has less than 6 parts",
            ),
            (
                "metrics.irb.cz.tgss.fidelity",  # Missing locus
                "irb gate quality metric name has less than 6 parts",
            ),
            (
                "metrics.ghz_state.QB1",  # Missing metric
                "ghz_state system quality metric name has less than 4 parts",
            ),
        ],
    )
    def test_invalid_quality_metric_name(self, name: str, match: str):
        """Test that parsing an improperly formed gate quality metric raises an error."""
        with pytest.raises(ValueError, match=re.escape(match)):
            QON.from_str(name)

    def test_invalid_suffix_format(self):
        """Test that parsing a metric with invalid suffix format raises an error."""
        with pytest.raises(ValueError, match=re.escape("Invalid suffix: invalid_suffix")):
            QON.from_str("metrics.ssro.measure.constant.QB1.fidelity:invalid_suffix")  # No equals sign

    @pytest.mark.parametrize(
        "name,match",
        [
            (
                "characterization.model.QB5",  # Missing property name
                "characterization.model observation name has less than 4 parts",
            ),
            (
                "characterization.QB5_t1_time",  # Missing the "model" element
                "Unparseable observation name.",
            ),
        ],
    )
    def test_invalid_characterization_observation_name(self, name: str, match: str):
        """Test that parsing an improperly formed characterization metric raises an error."""
        with pytest.raises(ValueError, match=re.escape(match)):
            QON.from_str(name)


class TestQONMetric:
    """Test the QONMetric class."""

    @pytest.mark.parametrize(
        "args,error_class,error",
        [
            (
                {"method": "aaa", "gate": "cz", "implementation": "crf", "locus_str": "QB1", "metric": "fidelity"},
                UnknownObservationError,
                "QONGateMetric is not registered to handle the method aaa",
            ),
            (
                {"method": "irb", "implementation": "crf", "locus_str": "QB1__QB2", "metric": "fidelity"},
                ValidationError,
                "1 validation error for QONGateMetric\ngate\n  Field required",
            ),
            (
                {"method": "irb", "gate": "cz", "locus_str": "QB1__QB2", "metric": "fidelity"},
                ValidationError,
                "1 validation error for QONGateMetric\nimplementation\n  Field required",
            ),
            (
                {
                    "method": "irb",
                    "gate": "cz",
                    "implementation": "crf",
                    "xxx": "yyy",
                    "locus_str": "QB1",
                    "metric": "fidelity",
                },
                ValidationError,
                "1 validation error for QONGateMetric\nxxx\n  Unexpected keyword argument",
            ),
        ],
    )
    def test_bad_init_gate_metric(self, args, error_class, error):
        """Disallowed initializations raise an error."""
        with pytest.raises(error_class, match=re.escape(error)):
            QONGateMetric(**args)

    @pytest.mark.parametrize(
        "args,error_class,error",
        [
            (
                {"method": "aaa", "locus_str": "QB1", "metric": "fidelity"},
                UnknownObservationError,
                "QONSystemMetric is not registered to handle the method aaa",
            ),
            (
                {"method": "ghz_state", "gate": "ggg", "locus_str": "QB1", "metric": "fidelity"},
                ValidationError,
                "1 validation error for QONSystemMetric\ngate\n  Unexpected keyword argument",
            ),
            (
                {"method": "ghz_state", "implementation": "iii", "locus_str": "QB1", "metric": "fidelity"},
                ValidationError,
                "1 validation error for QONSystemMetric\nimplementation\n  Unexpected keyword argument",
            ),
        ],
    )
    def test_bad_init_system_metric(self, args, error_class, error):
        """Disallowed initializations raise an error."""
        with pytest.raises(error_class, match=re.escape(error)):
            QONSystemMetric(**args)

    @pytest.mark.parametrize(
        "qon,name",
        [
            (
                QONGateMetric(
                    method="rb",
                    gate="prx",
                    implementation="drag_crf",
                    locus_str="QB1",
                    metric="fidelity",
                    suffixes={"b": "1", "a": "2"},  # not in lexical order
                ),
                "metrics.rb.prx.drag_crf.QB1.fidelity:a=2:b=1",  # suffixes in lexical order
            ),
        ],
    )
    def test_str(self, qon, name):
        """Test some quirks of the metric name convention."""
        assert str(qon) == name


class TestObservationFinder:
    """Test the ObservationFinder class."""

    @pytest.mark.parametrize(
        "obs_name,error,match",
        [
            ("a.b.c", UnknownObservationError, "a.b.c: Unknown observation domain"),
            ("metrics.b.c", ValueError, "Quality metric observation name has less than 4 parts"),
            ("characterization.model.QB1", ValueError, "Characterization observation name has less than 4 parts"),
            ("controllers.QB1", ValueError, "Controller setting observation name has less than 3 parts"),
            ("gates.prx.drag_gaussian.QB1", ValueError, "Gate parameter observation name has less than 5 parts"),
            ("gate_definitions.prx", ValueError, "Gate definition observation name has less than 3 parts"),
        ],
    )
    def test_init_bad_observation_name(self, caplog, obs_name, error, match):
        caplog.set_level(logging.WARNING)

        with pytest.raises(error, match=match):
            ObservationFinder([observation(obs_name, value=1.0)])
        assert not caplog.records  # no warnings so far

        # skipping unparseable observations just yields warnings
        ObservationFinder([observation(obs_name, value=1.0)], skip_unparseable=True)
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.levelname == "WARNING"
        assert match in rec.message

    @pytest.mark.parametrize(
        "first,second,value",
        [
            (
                observation("metrics.rb.prx.drag_crf.QB1.fidelity:par=d1", value=0.95),
                observation("metrics.rb.prx.drag_crf.QB1.fidelity:par=d2", value=0.99),
                0.95,
            ),
        ],
    )
    def test_repeated_observations(self, caplog, first, second, value):
        """With repeated observation names, only the first one is kept in the finder."""
        caplog.set_level(logging.WARNING)

        of = ObservationFinder(
            [
                observation("metrics.rb.prx.drag_crf.QB10.fidelity:par=d2", value=0.99),
                first,
                observation("metrics.rb.prx.drag_crf.QB100.fidelity:par=d2", value=0.99),
                second,
            ]
        )
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.levelname == "WARNING"
        assert f"Repeated observations: using {first.dut_field}, ignoring {second.dut_field}" in rec.message

        # check that we get the correct value
        path, _ = _split_obs_name(first.dut_field)
        assert of._get_path_value(path) == value

    def test_get_path_value(self):
        obs = [
            observation(name, value=value)
            for name, value in [
                ("gates.prx.drag_crf.QB1.duration", 1.0),
                ("characterization.model.QB1.t1_time", 2.0),
                ("metrics.ssro.measure.constant.QB1.fidelity", 3.0),
                ("controllers.QB1.flux.voltage", 4.0),
                ("gate_definitions.prx.default_implementation", "drag_crf"),
                ("gate_definitions.prx.drag_gaussian.override_default_for_loci", ["QB2"]),
            ]
        ]
        finder = ObservationFinder(obs)

        with pytest.raises(KeyError, match="does not end in an observation"):
            finder._get_path_value([])

        with pytest.raises(KeyError, match="path step 'a' could not be found"):
            finder._get_path_value(["a", "z"])

        with pytest.raises(KeyError, match="path step 'cz' could not be found"):
            finder._get_path_value(["gates", "cz"])

        # successful query
        for o in obs:
            if not o.dut_field.startswith("gate_definitions."):
                # only floats can be queried for now
                res = finder._get_path_value(o.dut_field.split("."))
                assert res == o.value

    def test_get_path_node(self):
        obs = [
            observation(name, value=value)
            for name, value in [
                ("gates.prx.drag_crf.QB1.duration", 1.0),
                ("characterization.model.QB1.t1_time", 2.0),
                ("metrics.ssro.measure.constant.QB1.fidelity", 3.0),
                ("controllers.QB1.flux.voltage", 4.0),
            ]
        ]
        finder = ObservationFinder(obs)

        with pytest.raises(KeyError, match="does not end in a node"):
            finder._get_path_node(["controllers", "QB1", "flux", "voltage"])

        with pytest.raises(KeyError, match="path step 'a' could not be found"):
            finder._get_path_node(["a", "z"])

        with pytest.raises(KeyError, match="path step 'cz' could not be found"):
            finder._get_path_node(["gates", "cz"])

        res = finder._get_path_node([])
        assert res.keys() == {"gates", "characterization", "metrics", "controllers"}

        res = finder._get_path_node(["controllers"])
        assert res.keys() == {"QB1"}

    @pytest.mark.parametrize(
        "observations,result",
        [
            (
                [
                    observation("gates.prx.drag_crf.QB1.duration", value=100e-9, unit="s"),
                    observation("characterization.model.QB1.t1_time", value=40e-6, unit="s"),
                    observation("characterization.model.QB7.t1_time", value=75e-6, unit="s"),
                    observation("characterization.model.QB1.t2_time", value=10e-6, unit="s"),
                    observation("characterization.model.QB5.t2_time", value=20e-6, unit="s"),
                    observation("characterization.model.QB1.t2_echo_time", value=30e-6, unit="s"),
                ],
                ({"QB1": 40e-6, "QB7": 75e-6}, {"QB1": 10e-6, "QB5": 20e-6}),
            ),
            (
                [
                    observation("metrics.ssro.measure.constant.QB1.fidelity", value=0.78),
                    observation("metrics.ghz_state.QB1__QB2.coherence_lower_bound", value=0.67),
                ],
                ({}, {}),
            ),
        ],
    )
    def test_get_coherence_times(self, observations, result):
        finder = ObservationFinder(observations)
        res = finder.get_coherence_times(["QB1", "QB5", "QB7", "QB10"])
        assert res == result

    @pytest.mark.parametrize(
        "observations",
        [
            (
                [
                    observation("characterization.model.QB1.t2_time", value=20e-9, unit="s"),
                    observation("controllers.QB1.drive.frequency", value=20e-9, unit="Hz"),
                    observation("metrics.ssro.measure.constant.QB1.fidelity:par=d1", value=0.95),
                    observation("gates.prx.drag_crf.QB1.duration", value=20e-9, unit="s"),
                    observation("gates.prx.drag_crf.QB1.full_width", value=15e-9, unit="s"),
                    observation("gates.cz.tgss.QB4__QB1.qubit.full_width", value=55e-9, unit="s"),
                    observation("gates.cz.tgss.QB4__QB1.duration", value=70e-9, unit="s"),
                ]
            ),
        ],
    )
    def test_get_gate_duration(self, observations):
        finder = ObservationFinder(observations)
        assert finder.get_gate_duration("prx", "drag_crf", ("QB1",)) == 20e-9
        assert finder.get_gate_duration("prx", "drag_crf", ("QB2",)) is None
        assert finder.get_gate_duration("prx", "drag_gaussian", ("QB1",)) is None
        assert finder.get_gate_duration("prx", "drag_gaussian", ("QB2",)) is None
        assert finder.get_gate_duration("cz", "tgss", ("QB4", "QB1")) == 70e-9
        assert finder.get_gate_duration("cz", "tgss", ("QB1", "QB4")) is None

    @pytest.mark.parametrize(
        "observations",
        [
            (
                [
                    observation("characterization.model.QB1.t1_time", value=20e-9, unit="s"),
                    observation("controllers.QB1.drive.frequency", value=20e-9, unit="Hz"),
                    observation("gates.prx.drag_crf.QB1.duration", value=20e-9, unit="s"),
                    observation("metrics.ssro.measure.constant.QB1.fidelity:par=d1", value=0.99),
                    observation("metrics.rb.prx.drag_crf.QB1.fidelity:par=d2", value=0.994),
                    observation("metrics.rb.prx.drag_crf.QB5.fidelity:par=d2", value=0.995),
                    observation("metrics.irb.cz.crf_crf.QB2__QB1.fidelity:par=d2", value=0.97),
                ]
            ),
        ],
    )
    def test_get_gate_fidelity(self, observations):
        finder = ObservationFinder(observations)
        assert finder.get_gate_fidelity("prx", "drag_crf", ("QB1",)) == 0.994
        assert finder.get_gate_fidelity("prx", "drag_crf", ("QB5",)) == 0.995
        assert finder.get_gate_fidelity("prx", "drag_gaussian", ("QB1",)) is None
        assert finder.get_gate_fidelity("prx", "drag_crf", ("QB2",)) is None
        assert finder.get_gate_fidelity("cz", "crf_crf", ("QB2", "QB1")) == 0.97
        assert finder.get_gate_fidelity("cz", "crf_crf", ("QB4", "QB1")) is None
        assert finder.get_gate_fidelity("cz", "crf_crf", ("QB1", "QB4")) is None
        assert finder.get_gate_fidelity("cz", "tgss", ("QB4", "QB1")) is None
        assert finder.get_gate_fidelity("cz", "tgss", ("QB4", "QB2")) is None

    @pytest.mark.parametrize(
        "observations,results",
        [
            (
                [
                    observation("characterization.model.QB1.t1_time", value=40e-6, unit="s"),
                    observation("characterization.model.QB5.t2_time", value=20e-6, unit="s"),
                ],
                {"QB1": None, "QB2": None, "QB3": None},
            ),
            (
                [
                    observation("metrics.ssro.measure.constant.QB1.error_0_to_1", value=0.78),
                    observation("metrics.ssro.measure.constant.QB1.error_1_to_0:aaa=bbb:par=d1", value=0.96),
                    observation("metrics.ssro.measure.constant.QB2.error_0_to_1", value=0.93),
                    observation("metrics.ssro.measure.constant.QB2.error_1_to_0", value=0.91),
                    observation("metrics.ssro.measure.constant.QB3.error_0_to_1", value=0.95),  # both must be present
                ],
                {"QB1": (0.78, 0.96), "QB2": (0.93, 0.91), "QB3": None},
            ),
        ],
    )
    def test_get_measure_errors(self, observations, results):
        finder = ObservationFinder(observations)
        res = {q: finder.get_measure_errors("measure", "constant", (q,)) for q in ["QB1", "QB2", "QB3"]}
        assert res == results

    @pytest.mark.parametrize(
        "observations",
        [
            [
                observation("controllers.QB1.drive.frequency", value=4.5e9, unit="Hz"),
                observation("controllers.QB5.drive.frequency", value=5e9, unit="Hz"),
            ],
        ],
    )
    def test_get_qubit_frequencies(self, observations):
        finder = ObservationFinder(observations)
        assert finder.get_qubit_frequency("QB1") == 4.5e9
        assert finder.get_qubit_frequency("QB2") is None
        assert finder.get_qubit_frequency("QB5") == 5e9
