# Copyright 2022-2023 Qiskit on IQM developers
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
"""Fake (i.e. simulated) backend for IQM's 20-qubit Apollo architecture"""

from iqm.iqm_client import StaticQuantumArchitecture
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile, IQMFakeBackend


def IQMFakeApollo() -> IQMFakeBackend:
    """Return IQMFakeBackend instance representing IQM's Apollo architecture."""
    architecture = StaticQuantumArchitecture(
        qubits=[f"QB{i}" for i in range(1, 21)],
        computational_resonators=[],
        connectivity=[
            ("QB1", "QB2"),
            ("QB1", "QB4"),
            ("QB2", "QB5"),
            ("QB3", "QB4"),
            ("QB3", "QB8"),
            ("QB4", "QB5"),
            ("QB4", "QB9"),
            ("QB5", "QB6"),
            ("QB5", "QB10"),
            ("QB6", "QB7"),
            ("QB6", "QB11"),
            ("QB7", "QB12"),
            ("QB8", "QB9"),
            ("QB8", "QB13"),
            ("QB9", "QB10"),
            ("QB9", "QB14"),
            ("QB10", "QB11"),
            ("QB10", "QB15"),
            ("QB11", "QB12"),
            ("QB11", "QB16"),
            ("QB12", "QB17"),
            ("QB13", "QB14"),
            ("QB14", "QB15"),
            ("QB14", "QB18"),
            ("QB15", "QB16"),
            ("QB15", "QB19"),
            ("QB16", "QB17"),
            ("QB16", "QB20"),
            ("QB18", "QB19"),
            ("QB19", "QB20"),
        ],
    )
    # Note that these specs are ballpark numbers and don't correspond directly to a specific device
    error_profile = IQMErrorProfile(
        t1s={
            "QB1": 49500.0,
            "QB2": 49400.0,
            "QB3": 41500.0,
            "QB4": 50300.0,
            "QB5": 49100.0,
            "QB6": 48700.0,
            "QB7": 48800.0,
            "QB8": 49900.0,
            "QB9": 48200.0,
            "QB10": 49600.0,
            "QB11": 42700.0,
            "QB12": 50100.0,
            "QB13": 47500.0,
            "QB14": 48300.0,
            "QB15": 39200.0,
            "QB16": 49000.0,
            "QB17": 49100.0,
            "QB18": 49200.0,
            "QB19": 41600.0,
            "QB20": 41800.0,
        },
        t2s={
            "QB1": 09100.0,
            "QB2": 10100.0,
            "QB3": 10900.0,
            "QB4": 09600.0,
            "QB5": 08900.0,
            "QB6": 10200.0,
            "QB7": 08500.0,
            "QB8": 09000.0,
            "QB9": 09300.0,
            "QB10": 09800.0,
            "QB11": 09400.0,
            "QB12": 09900.0,
            "QB13": 10000.0,
            "QB14": 09000.0,
            "QB15": 09500.0,
            "QB16": 09700.0,
            "QB17": 10100.0,
            "QB18": 09200.0,
            "QB19": 08200.0,
            "QB20": 12000.0,
        },
        single_qubit_gate_depolarizing_error_parameters={
            "prx": {
                "QB1": 0.00116,
                "QB2": 0.00102,
                "QB3": 0.00182,
                "QB4": 0.00100,
                "QB5": 0.00119,
                "QB6": 0.00110,
                "QB7": 0.00124,
                "QB8": 0.00109,
                "QB9": 0.00111,
                "QB10": 0.00132,
                "QB11": 0.00114,
                "QB12": 0.00116,
                "QB13": 0.00112,
                "QB14": 0.00107,
                "QB15": 0.00211,
                "QB16": 0.00119,
                "QB17": 0.00126,
                "QB18": 0.00133,
                "QB19": 0.00160,
                "QB20": 0.00115,
            }
        },
        two_qubit_gate_depolarizing_error_parameters={
            "cz": {
                ("QB1", "QB2"): 0.0120,
                ("QB1", "QB4"): 0.0121,
                ("QB2", "QB5"): 0.0102,
                ("QB3", "QB4"): 0.0150,
                ("QB8", "QB3"): 0.0103,
                ("QB4", "QB5"): 0.0160,
                ("QB9", "QB4"): 0.0117,
                ("QB5", "QB6"): 0.0098,
                ("QB10", "QB5"): 0.0210,
                ("QB6", "QB7"): 0.0187,
                ("QB11", "QB6"): 0.0135,
                ("QB12", "QB7"): 0.0141,
                ("QB8", "QB9"): 0.0192,
                ("QB8", "QB13"): 0.0183,
                ("QB9", "QB10"): 0.0133,
                ("QB9", "QB14"): 0.0109,
                ("QB10", "QB11"): 0.0134,
                ("QB10", "QB15"): 0.0122,
                ("QB11", "QB12"): 0.0100,
                ("QB16", "QB11"): 0.0201,
                ("QB17", "QB12"): 0.0180,
                ("QB13", "QB14"): 0.0127,
                ("QB14", "QB15"): 0.0142,
                ("QB18", "QB14"): 0.0191,
                ("QB16", "QB15"): 0.0146,
                ("QB19", "QB15"): 0.0162,
                ("QB16", "QB17"): 0.0133,
                ("QB16", "QB20"): 0.0220,
                ("QB18", "QB19"): 0.0190,
                ("QB19", "QB20"): 0.0114,
            }
        },
        single_qubit_gate_durations={"prx": 42.0},
        two_qubit_gate_durations={"cz": 130.0},
        readout_errors={
            "QB1": {"0": 0.051, "1": 0.052},
            "QB2": {"0": 0.049, "1": 0.048},
            "QB3": {"0": 0.057, "1": 0.049},
            "QB4": {"0": 0.049, "1": 0.048},
            "QB5": {"0": 0.050, "1": 0.051},
            "QB6": {"0": 0.053, "1": 0.055},
            "QB7": {"0": 0.051, "1": 0.052},
            "QB8": {"0": 0.058, "1": 0.050},
            "QB9": {"0": 0.052, "1": 0.049},
            "QB10": {"0": 0.044, "1": 0.049},
            "QB11": {"0": 0.056, "1": 0.049},
            "QB12": {"0": 0.051, "1": 0.050},
            "QB13": {"0": 0.053, "1": 0.051},
            "QB14": {"0": 0.048, "1": 0.049},
            "QB15": {"0": 0.046, "1": 0.049},
            "QB16": {"0": 0.044, "1": 0.051},
            "QB17": {"0": 0.048, "1": 0.048},
            "QB18": {"0": 0.049, "1": 0.052},
            "QB19": {"0": 0.050, "1": 0.051},
            "QB20": {"0": 0.056, "1": 0.057},
        },
        name="sample-chip",
    )

    return IQMFakeBackend(architecture, error_profile, name="IQMFakeApolloBackend")
