# Copyright 2024-2025 IQM
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

from datetime import datetime, timezone
import uuid

import numpy as np
import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter, Sweep
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import Sweeps
from iqm.station_control.interface.models import JobExecutorStatus, RunData, SweepData, SweepResults

DUMMY_TIMESTAMP = datetime.now(timezone.utc)
QUBIT_1 = "QB1"
PROBE_LINE_1 = "PL_RO-1"
DUT_LABEL = "M138_W36_A22_N05"


@pytest.fixture
def return_parameter() -> Parameter:
    return Parameter(
        name="QB1__readout.result",
        label="Readout signal",
        unit="V",
        data_type=DataType.COMPLEX,
        collection_type=CollectionType.NDARRAY,
    )


@pytest.fixture
def settings(return_parameter) -> SettingNode:
    _settings = SettingNode(name="root", controllers=SettingNode("controllers"))
    _settings.controllers[QUBIT_1] = SettingNode(
        name=QUBIT_1,
        QB1__readout=SettingNode(name="QB1__readout"),
    )
    _settings.controllers[QUBIT_1].QB1__readout.result = return_parameter
    _settings.controllers[PROBE_LINE_1] = SettingNode(name=PROBE_LINE_1)
    return _settings


@pytest.fixture
def sweeps() -> Sweeps:
    return [
        (
            Sweep(
                parameter=Parameter(
                    name="QB1__readout.frequency",
                    label="RF frequency",
                    unit="Hz",
                    data_type=DataType.FLOAT,
                    collection_type=CollectionType.SCALAR,
                ),
                data=[4900000000.0 + i * (5000000000.0 - 4900000000.0) / 100.0 for i in range(40)],
            ),
        )
    ]


@pytest.fixture
def hard_sweeps() -> list:
    return []


@pytest.fixture
def request_metadata():
    return {
        "components": ["dummy_component"],
        "default_data_parameters": ["dummy_data_parameter"],
        "default_sweep_parameters": ["dummy_sweep_parameter", "dummy_sweep_parameter2"],
    }


@pytest.fixture
def complete_sweep_results() -> SweepResults:
    return {
        "QB1__readout.result": [
            np.array([0.3966668 + 0.60797483j]),
            np.array([0.56322683 + 0.15308522j]),
            np.array([0.009805 + 0.93189581j]),
            np.array([0.51292882 + 0.84966336j]),
            np.array([0.63108479 + 0.3506445j]),
            np.array([0.35513196 + 0.67341471j]),
            np.array([0.7975558 + 0.252086j]),
            np.array([0.02890332 + 0.83518719j]),
            np.array([0.11209141 + 0.23684272j]),
            np.array([0.77098268 + 0.11165443j]),
            np.array([0.79010949 + 0.42518574j]),
            np.array([0.22364582 + 0.93659597j]),
            np.array([0.10330843 + 0.19367069j]),
            np.array([0.26633723 + 0.27096521j]),
            np.array([0.56561033 + 0.34537909j]),
            np.array([0.84277925 + 0.46434322j]),
            np.array([0.82815716 + 0.32708441j]),
            np.array([0.6585713 + 0.36820628j]),
            np.array([0.5767659 + 0.87169661j]),
            np.array([0.23686943 + 0.17379782j]),
            np.array([0.33560868 + 0.03792931j]),
            np.array([0.07875081 + 0.64046843j]),
            np.array([0.48091518 + 0.93719554j]),
            np.array([0.80980932 + 0.66098185j]),
            np.array([0.70603815 + 0.60578963j]),
            np.array([0.27698725 + 0.5533618j]),
            np.array([0.7441153 + 0.25168008j]),
            np.array([0.60828426 + 0.54104947j]),
            np.array([0.7802419 + 0.89821964j]),
            np.array([0.99519951 + 0.61848065j]),
            np.array([0.97002231 + 0.12920906j]),
            np.array([0.87390379 + 0.77185258j]),
            np.array([0.51620239 + 0.37564894j]),
            np.array([0.80809744 + 0.472233j]),
            np.array([0.18033479 + 0.38836614j]),
            np.array([0.12346678 + 0.94137644j]),
            np.array([0.23501886 + 0.79577116j]),
            np.array([0.51777319 + 0.25210899j]),
            np.array([0.46004982 + 0.7714734j]),
            np.array([0.52574089 + 0.08023739j]),
        ]
    }


@pytest.fixture
def interrupted_sweep_results(complete_sweep_results, return_parameter) -> SweepResults:
    complete_sweep_results[return_parameter.name] = complete_sweep_results[return_parameter.name][:30]  # Interrupted
    return complete_sweep_results


@pytest.fixture
def complete_run_data(return_parameter, hard_sweeps, settings, sweeps) -> RunData:
    """A fixture of a :class:`.RunData` instance, initialized with a complete run data."""
    run_data = RunData(
        run_id=uuid.uuid4(),
        username="username",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        software_version_set_id=1,
        hard_sweeps={return_parameter.name: hard_sweeps},
        components=["dummy_component"],
        default_data_parameters=["dummy_data_parameter"],
        default_sweep_parameters=["dummy_sweep_parameter", "dummy_sweep_parameter2"],
        sweep_data=SweepData(
            sweep_id=uuid.uuid4(),
            dut_label=DUT_LABEL,
            settings=settings,
            sweeps=sweeps,
            return_parameters=[return_parameter.name],
            created_timestamp=DUMMY_TIMESTAMP,
            modified_timestamp=DUMMY_TIMESTAMP,
            begin_timestamp=DUMMY_TIMESTAMP,
            end_timestamp=DUMMY_TIMESTAMP,  # Not interrupted
            job_status=JobExecutorStatus.READY,
        ),
        created_timestamp=DUMMY_TIMESTAMP,
        modified_timestamp=DUMMY_TIMESTAMP,
        begin_timestamp=DUMMY_TIMESTAMP,
        end_timestamp=None,
    )
    return run_data


@pytest.fixture
def interrupted_run_data(complete_run_data) -> RunData:
    """A fixture of a :class:`.RunData` instance, initialized with an interrupted run data."""
    complete_run_data.sweep_data.end_timestamp = None
    return complete_run_data
