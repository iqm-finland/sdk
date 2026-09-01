# Copyright 2022-2026 IQM
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

"""Unit tests for ReadoutErrorCharacterization and RECConfiguration."""

from __future__ import annotations

import dataclasses
from datetime import datetime
import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from iqm.error_reduction_tools.readout_characterization.readout_error_characterization import (
    ReadoutErrorCharacterization,
    RECConfiguration,
)
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_QUBITS = ["QB1", "QB2"]
_COUNTS_BY_PREP: dict[str, dict[str, int]] = {
    "II": {"00": 1000, "01": 50},
    "XX": {"11": 1000, "10": 30},
}
_RAW_RESULTS: dict[str, Any] = {
    "counts_by_prep": _COUNTS_BY_PREP,
    "measured_qubits": _QUBITS,
}
_CHARACT_DATA: dict[str, np.ndarray] = {
    "QB1": np.array([[0.95, 0.03], [0.05, 0.97]]),
    "QB2": np.array([[0.97, 0.04], [0.03, 0.96]]),
}
_CHARACT_DATA_STD: dict[str, np.ndarray] = {
    "QB1": np.array([[0.005, 0.006], [0.005, 0.006]]),
    "QB2": np.array([[0.004, 0.007], [0.004, 0.007]]),
}
_ERROR_PROBS: dict[str, dict[str, np.ndarray]] = {
    "charact_data": _CHARACT_DATA,
    "charact_data_std": _CHARACT_DATA_STD,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "iqm.error_reduction_tools.readout_characterization.readout_error_characterization"


def _make_rec_with_results(
    *,
    with_probs: bool = False,
    config: RECConfiguration | None = None,
) -> ReadoutErrorCharacterization:
    """Return a REC instance that already has raw results (and optionally cached probs)."""
    rec = ReadoutErrorCharacterization(client=None)
    rec._raw_results = _RAW_RESULTS.copy()
    if config is not None:
        rec._config = config
    if with_probs:
        rec._error_probabilities = _ERROR_PROBS
    return rec


# ===========================================================================
# RECConfiguration
# ===========================================================================


class TestRECConfiguration:
    def test_default_values(self):
        cfg = RECConfiguration()

        assert cfg.num_circuits == 50
        assert cfg.shots == 10_000
        assert cfg.qubits is None
        assert cfg.seed is None
        assert cfg.symmetrize is True
        assert cfg.equatorial_randomization is True

    def test_custom_values(self):
        cfg = RECConfiguration(
            num_circuits=20,
            shots=10_000,
            qubits=["QB1", "QB3"],
            seed=42,
            symmetrize=False,
            equatorial_randomization=False,
        )

        assert cfg.num_circuits == 20
        assert cfg.shots == 10_000
        assert cfg.qubits == ["QB1", "QB3"]
        assert cfg.seed == 42
        assert cfg.symmetrize is False
        assert cfg.equatorial_randomization is False

    def test_asdict_has_expected_keys(self):
        cfg = RECConfiguration(seed=7)
        d = dataclasses.asdict(cfg)

        assert set(d.keys()) == {
            "num_circuits",
            "shots",
            "qubits",
            "seed",
            "symmetrize",
            "equatorial_randomization",
        }
        assert d["seed"] == 7

    def test_equality(self):
        assert RECConfiguration() == RECConfiguration()
        assert RECConfiguration(seed=1) != RECConfiguration(seed=2)

    def test_two_instances_are_independent(self):
        """Mutable default field 'qubits' must not be shared between instances."""
        cfg1 = RECConfiguration(qubits=["QB1"])
        cfg2 = RECConfiguration(qubits=["QB2"])

        assert cfg1.qubits != cfg2.qubits


# ===========================================================================
# ReadoutErrorCharacterization — construction
# ===========================================================================


class TestInit:
    def test_with_none_client(self):
        rec = ReadoutErrorCharacterization(client=None)

        assert rec._client is None
        assert rec._config is None
        assert rec._job is None
        assert rec._job_info is None
        assert rec._raw_results is None
        assert rec._error_probabilities is None

    def test_with_client(self):
        client = MagicMock()
        rec = ReadoutErrorCharacterization(client=client)

        assert rec._client is client

    def test_two_instances_are_independent(self):
        rec1 = ReadoutErrorCharacterization(client=None)
        rec2 = ReadoutErrorCharacterization(client=None)

        rec1._config = RECConfiguration(shots=1)

        assert rec2._config is None


# ===========================================================================
# __repr__
# ===========================================================================


class TestRepr:
    def test_empty_instance(self):
        rec = ReadoutErrorCharacterization(client=None)
        r = repr(rec)

        assert r.startswith("ReadoutErrorCharacterization(")
        assert "client" not in r

    def test_with_client(self):
        rec = ReadoutErrorCharacterization(client=MagicMock())
        assert "client=<connected>" in repr(rec)

    def test_with_config(self):
        rec = ReadoutErrorCharacterization(client=None)
        rec._config = RECConfiguration(shots=999)
        assert "config=" in repr(rec)

    def test_with_submitted_job(self):
        rec = ReadoutErrorCharacterization(client=None)
        rec._job = MagicMock()
        assert "job=<submitted>" in repr(rec)

    def test_with_results(self):
        rec = _make_rec_with_results()
        assert "results_available(n_qubits=2)" in repr(rec)

    def test_with_probabilities(self):
        rec = _make_rec_with_results(with_probs=True)
        assert "probabilities_computed" in repr(rec)


# ===========================================================================
# submit_job
# ===========================================================================


class TestSubmitJob:
    def test_raises_without_client(self):
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(RuntimeError, match="No client available"):
            rec.submit_job()

    def test_returns_self(self):
        client = MagicMock()
        mock_job = MagicMock()
        mock_info = {"qubits": _QUBITS}

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(mock_job, mock_info)):
            rec = ReadoutErrorCharacterization(client=client)
            result = rec.submit_job()

        assert result is rec

    def test_stores_job_and_job_info(self):
        client = MagicMock()
        mock_job = MagicMock()
        mock_info = {"qubits": _QUBITS, "num_circuits": 10}

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(mock_job, mock_info)):
            rec = ReadoutErrorCharacterization(client=client)
            rec.submit_job()

        assert rec._job is mock_job
        assert rec._job_info is mock_info

    def test_stores_config(self):
        client = MagicMock()
        cfg = RECConfiguration(shots=1_000, qubits=["QB1"])

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(MagicMock(), {})):
            rec = ReadoutErrorCharacterization(client=client)
            rec.submit_job(cfg)

        assert rec._config is cfg

    def test_defaults_to_rec_configuration(self):
        client = MagicMock()

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(MagicMock(), {})) as mock_run:
            rec = ReadoutErrorCharacterization(client=client)
            rec.submit_job()

        _, kwargs = mock_run.call_args
        assert kwargs["shots"] == RECConfiguration().shots
        assert kwargs["number_of_circuits"] == RECConfiguration().num_circuits

    def test_passes_config_fields_to_run(self):
        client = MagicMock()
        cfg = RECConfiguration(
            num_circuits=20,
            shots=5_000,
            qubits=["QB1", "QB2"],
            seed=99,
            symmetrize=False,
            equatorial_randomization=False,
        )

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(MagicMock(), {})) as mock_run:
            ReadoutErrorCharacterization(client=client).submit_job(cfg)

        mock_run.assert_called_once_with(
            client=client,
            qubits=["QB1", "QB2"],
            number_of_circuits=20,
            shots=5_000,
            symmetrize=False,
            seed=99,
            equatorial_randomization=False,
        )


# ===========================================================================
# retrieve_results
# ===========================================================================


class TestRetrieveResults:
    def test_raises_with_no_job_at_all(self):
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(RuntimeError, match="No job available"):
            rec.retrieve_results()

    def test_raises_when_only_job_provided(self):
        """Passing only job without job_info is ambiguous and must raise."""
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(ValueError, match="must be provided together"):
            rec.retrieve_results(job=MagicMock())

    def test_raises_when_only_job_info_provided(self):
        """Passing only job_info without job is ambiguous and must raise."""
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(ValueError, match="must be provided together"):
            rec.retrieve_results(job_info={"qubits": _QUBITS})

    def test_explicit_pair_bypasses_submit_job(self):
        """Passing both job and job_info directly works without calling submit_job."""
        external_job = MagicMock()
        external_info = {
            "qubits": _QUBITS,
            "num_circuits": 10,
            "prep_strings": [],
            "shots_per_circuit": 500,
        }
        rec = ReadoutErrorCharacterization(client=None)
        # _job and _job_info are both None — submit_job was never called

        with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS):
            rec.retrieve_results(job=external_job, job_info=external_info)

        assert rec._raw_results is _RAW_RESULTS

    def test_explicit_pair_used_verbatim(self):
        """The explicit job/job_info pair must be forwarded as-is, ignoring internal state."""
        external_job = MagicMock()
        external_info = {"qubits": _QUBITS}
        rec = ReadoutErrorCharacterization(client=None)
        rec._job = MagicMock()  # internal job should NOT be used
        rec._job_info = {"qubits": ["QB9"]}  # internal info should NOT be used

        with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS) as mock_ret:
            rec.retrieve_results(job=external_job, job_info=external_info)

        called_job, called_info = mock_ret.call_args[0]
        assert called_job is external_job
        assert called_info is external_info

    def test_returns_self(self):
        rec = ReadoutErrorCharacterization(client=None)
        rec._job = MagicMock()
        rec._job_info = {}

        with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS):
            result = rec.retrieve_results()

        assert result is rec

    def test_stores_raw_results(self):
        rec = ReadoutErrorCharacterization(client=None)
        rec._job = MagicMock()
        rec._job_info = {}

        with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS):
            rec.retrieve_results()

        assert rec._raw_results is _RAW_RESULTS

    def test_uses_internal_job_and_info_by_default(self):
        internal_job = MagicMock()
        internal_info = {"qubits": _QUBITS}
        rec = ReadoutErrorCharacterization(client=None)
        rec._job = internal_job
        rec._job_info = internal_info

        with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS) as mock_ret:
            rec.retrieve_results()

        mock_ret.assert_called_once_with(internal_job, internal_info)


# ===========================================================================
# get_readout_error_probabilities
# ===========================================================================


class TestGetReadoutErrorProbabilities:
    def test_raises_without_results(self):
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(RuntimeError, match="No results available"):
            rec.get_readout_error_probabilities()

    def test_computes_and_returns_probabilities(self):
        rec = _make_rec_with_results()

        with patch(f"{MODULE}.compute_error_probabilities", return_value=_ERROR_PROBS) as mock_comp:
            result = rec.get_readout_error_probabilities()

        mock_comp.assert_called_once_with(_RAW_RESULTS)
        assert result is _ERROR_PROBS

    def test_caches_result_on_second_call(self):
        rec = _make_rec_with_results()

        with patch(f"{MODULE}.compute_error_probabilities", return_value=_ERROR_PROBS) as mock_comp:
            first = rec.get_readout_error_probabilities()
            second = rec.get_readout_error_probabilities()

        assert mock_comp.call_count == 1
        assert first is second

    def test_returns_pre_cached_value_without_recomputing(self):
        """When _error_probabilities is already set, compute_error_probabilities must not be called."""
        rec = _make_rec_with_results(with_probs=True)

        with patch(f"{MODULE}.compute_error_probabilities") as mock_comp:
            result = rec.get_readout_error_probabilities()

        mock_comp.assert_not_called()
        assert result is _ERROR_PROBS


# ===========================================================================
# to_dict
# ===========================================================================


class TestToDict:
    def test_raises_without_results(self):
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(RuntimeError, match="No results to serialize"):
            rec.to_dict()

    def test_mandatory_keys_present(self):
        rec = _make_rec_with_results(config=RECConfiguration())
        d = rec.to_dict()

        assert "counts_by_prep" in d
        assert "measured_qubits" in d
        assert "config" in d
        assert "timestamp" in d

    def test_counts_and_qubits_match_raw_results(self):
        rec = _make_rec_with_results()
        d = rec.to_dict()

        assert d["counts_by_prep"] == _COUNTS_BY_PREP
        assert d["measured_qubits"] == _QUBITS

    def test_config_serialized_when_set(self):
        cfg = RECConfiguration(shots=999, seed=7)
        rec = _make_rec_with_results(config=cfg)
        d = rec.to_dict()

        assert d["config"] == dataclasses.asdict(cfg)

    def test_config_is_none_when_not_set(self):
        rec = _make_rec_with_results()  # no config
        d = rec.to_dict()

        assert d["config"] is None

    def test_charact_data_absent_when_probs_not_computed(self):
        rec = _make_rec_with_results(with_probs=False)
        d = rec.to_dict()

        assert "charact_data" not in d
        assert "charact_data_std" not in d

    def test_charact_data_present_when_probs_computed(self):
        rec = _make_rec_with_results(with_probs=True)
        d = rec.to_dict()

        assert "charact_data" in d
        assert "charact_data_std" in d

    def test_charact_data_is_list_of_lists(self):
        """np.ndarray must be converted to nested lists for JSON serialization."""
        rec = _make_rec_with_results(with_probs=True)
        d = rec.to_dict()

        for v in d["charact_data"].values():
            assert isinstance(v, list)
        for v in d["charact_data_std"].values():
            assert isinstance(v, list)

    def test_timestamp_is_iso_format(self):
        rec = _make_rec_with_results()
        d = rec.to_dict()

        # Should not raise
        datetime.fromisoformat(d["timestamp"])

    def test_to_dict_stores_timestamp_on_instance(self):
        rec = _make_rec_with_results()
        assert rec._timestamp is None

        rec.to_dict()

        assert rec._timestamp is not None

    def test_output_is_json_serializable(self):
        rec = _make_rec_with_results(config=RECConfiguration(), with_probs=True)
        d = rec.to_dict()

        # Must not raise
        json.dumps(d)


# ===========================================================================
# from_dict
# ===========================================================================


class TestFromDict:
    def _base_dict(self, *, with_probs: bool = False, with_config: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "counts_by_prep": _COUNTS_BY_PREP,
            "measured_qubits": _QUBITS,
            "config": dataclasses.asdict(RECConfiguration()) if with_config else None,
            "timestamp": "2026-03-17T12:00:00",
        }
        if with_probs:
            d["charact_data"] = {k: v.tolist() for k, v in _CHARACT_DATA.items()}
            d["charact_data_std"] = {k: v.tolist() for k, v in _CHARACT_DATA_STD.items()}
        return d

    def test_client_is_none(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict())
        assert rec._client is None

    def test_job_and_job_info_are_none(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict())
        assert rec._job is None
        assert rec._job_info is None

    def test_raw_results_restored(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict())

        assert rec._raw_results["counts_by_prep"] == _COUNTS_BY_PREP
        assert rec._raw_results["measured_qubits"] == _QUBITS

    def test_config_restored(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict(with_config=True))
        assert isinstance(rec._config, RECConfiguration)
        assert rec._config == RECConfiguration()

    def test_config_is_none_when_absent(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict(with_config=False))
        assert rec._config is None

    def test_error_probabilities_absent_when_not_in_dict(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict(with_probs=False))
        assert rec._error_probabilities is None

    def test_error_probabilities_restored_from_dict(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict(with_probs=True))

        assert rec._error_probabilities is not None
        for qubit in _QUBITS:
            np.testing.assert_array_almost_equal(
                rec._error_probabilities["charact_data"][qubit],
                _CHARACT_DATA[qubit],
            )
            np.testing.assert_array_almost_equal(
                rec._error_probabilities["charact_data_std"][qubit],
                _CHARACT_DATA_STD[qubit],
            )

    def test_backward_compat_missing_charact_data_std(self):
        """Older files without 'charact_data_std' must load without error."""
        d = self._base_dict(with_probs=True)
        del d["charact_data_std"]

        rec = ReadoutErrorCharacterization.from_dict(d)

        assert rec._error_probabilities is not None
        assert rec._error_probabilities["charact_data_std"] == {}

    def test_charact_data_values_are_numpy_arrays(self):
        rec = ReadoutErrorCharacterization.from_dict(self._base_dict(with_probs=True))

        for v in rec._error_probabilities["charact_data"].values():
            assert isinstance(v, np.ndarray)
        for v in rec._error_probabilities["charact_data_std"].values():
            assert isinstance(v, np.ndarray)

    def test_timestamp_restored(self):
        ts = "2026-03-17T12:34:56.789"
        d = self._base_dict()
        d["timestamp"] = ts

        rec = ReadoutErrorCharacterization.from_dict(d)

        assert rec._timestamp == ts

    def test_timestamp_is_none_when_absent(self):
        d = self._base_dict()
        d.pop("timestamp", None)

        rec = ReadoutErrorCharacterization.from_dict(d)

        assert rec._timestamp is None


# ===========================================================================
# save / load round-trip
# ===========================================================================


class TestSaveLoad:
    def test_round_trip_without_probs(self, tmp_path):
        path = str(tmp_path / "rec.json")
        original = _make_rec_with_results(config=RECConfiguration(seed=42))

        original.save(path)
        loaded = ReadoutErrorCharacterization.load(path)

        assert loaded._raw_results["measured_qubits"] == _QUBITS
        assert loaded._raw_results["counts_by_prep"] == _COUNTS_BY_PREP
        assert loaded._config == RECConfiguration(seed=42)
        assert loaded._error_probabilities is None

    def test_round_trip_with_probs(self, tmp_path):
        path = str(tmp_path / "rec_probs.json")
        original = _make_rec_with_results(config=RECConfiguration(), with_probs=True)

        original.save(path)
        loaded = ReadoutErrorCharacterization.load(path)

        for qubit in _QUBITS:
            np.testing.assert_array_almost_equal(
                loaded._error_probabilities["charact_data"][qubit],
                _CHARACT_DATA[qubit],
            )
            np.testing.assert_array_almost_equal(
                loaded._error_probabilities["charact_data_std"][qubit],
                _CHARACT_DATA_STD[qubit],
            )

    def test_saved_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "rec.json")
        _make_rec_with_results().save(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_save_raises_without_results(self, tmp_path):
        path = str(tmp_path / "rec.json")
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(RuntimeError, match="No results to serialize"):
            rec.save(path)

    def test_load_restores_config(self, tmp_path):
        path = str(tmp_path / "rec.json")
        cfg = RECConfiguration(shots=1_234, seed=3)
        _make_rec_with_results(config=cfg).save(path)

        loaded = ReadoutErrorCharacterization.load(path)

        assert loaded._config == cfg

    def test_load_restores_timestamp(self, tmp_path):
        path = str(tmp_path / "rec_ts.json")
        original = _make_rec_with_results()
        original.save(path)

        loaded = ReadoutErrorCharacterization.load(path)

        assert loaded._timestamp is not None
        assert loaded._timestamp == original._timestamp


# ===========================================================================
# plot_error_probabilities
# ===========================================================================


class TestPlotErrorProbabilities:
    def test_raises_when_no_results(self):
        rec = ReadoutErrorCharacterization(client=None)

        with pytest.raises(RuntimeError, match="No results available"):
            rec.plot_error_probabilities()

    def test_delegates_to_visualization_module(self):
        rec = _make_rec_with_results(with_probs=True)
        mock_viz = MagicMock()

        with patch.dict(sys.modules, {"iqm.error_reduction_tools.readout_characterization.visualization": mock_viz}):
            rec.plot_error_probabilities(title="Test", show_plot=False)

        mock_viz.plot_error_probabilities.assert_called_once_with(_ERROR_PROBS, title="Test", show_plot=False)

    def test_kwargs_forwarded(self):
        rec = _make_rec_with_results(with_probs=True)
        mock_viz = MagicMock()

        with patch.dict(sys.modules, {"iqm.error_reduction_tools.readout_characterization.visualization": mock_viz}):
            rec.plot_error_probabilities(show_plot=False)

        _, kwargs = mock_viz.plot_error_probabilities.call_args
        assert kwargs.get("show_plot") is False


# ===========================================================================
# Method chaining
# ===========================================================================


class TestChaining:
    def test_submit_retrieve_chain_returns_same_instance(self):
        client = MagicMock()
        mock_job = MagicMock()
        mock_info = {}

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(mock_job, mock_info)):
            with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS):
                rec = ReadoutErrorCharacterization(client=client)
                result = rec.submit_job().retrieve_results()

        assert result is rec
        assert rec._raw_results is _RAW_RESULTS

    def test_full_chain(self):
        client = MagicMock()
        mock_job = MagicMock()
        mock_info = {}

        with patch(f"{MODULE}.run_calibration_circuits", return_value=(mock_job, mock_info)):
            with patch(f"{MODULE}.retrieve_calibration_results", return_value=_RAW_RESULTS):
                with patch(f"{MODULE}.compute_error_probabilities", return_value=_ERROR_PROBS):
                    probs = (
                        ReadoutErrorCharacterization(client=client)
                        .submit_job()
                        .retrieve_results()
                        .get_readout_error_probabilities()
                    )

        assert probs is _ERROR_PROBS
