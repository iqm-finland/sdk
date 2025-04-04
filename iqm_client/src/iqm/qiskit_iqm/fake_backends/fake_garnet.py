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
"""Fake (i.e. simulated) backend for IQM's 20-qubit Garnet architecture"""

from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile, IQMFakeBackend


def IQMFakeGarnet() -> IQMFakeBackend:
    """Return IQMFakeBackend instance representing IQM's Garnet architecture."""
    error_profile = IQMErrorProfile(
        t1s={
            "QB1": 37741.0,
            "QB2": 32584.0,
            "QB3": 24468.0,
            "QB4": 11555.0,
            "QB5": 34257.0,
            "QB6": 40051.0,
            "QB7": 35708.0,
            "QB8": 25686.0,
            "QB9": 34113.0,
            "QB10": 37391.0,
            "QB11": 25809.0,
            "QB12": 60725.0,
            "QB13": 44802.0,
            "QB14": 48137.0,
            "QB15": 39052.0,
            "QB16": 43968.0,
            "QB17": 36670.0,
            "QB18": 38151.0,
            "QB19": 50012.0,
            "QB20": 54911.0,
        },
        t2s={
            "QB1": 09180.0,
            "QB2": 10040.0,
            "QB3": 10950.0,
            "QB4": 09670.0,
            "QB5": 08960.0,
            "QB6": 10130.0,
            "QB7": 08590.0,
            "QB8": 09050.0,
            "QB9": 09380.0,
            "QB10": 09820.0,
            "QB11": 09460.0,
            "QB12": 09940.0,
            "QB13": 10070.0,
            "QB14": 09030.0,
            "QB15": 09590.0,
            "QB16": 09710.0,
            "QB17": 10180.0,
            "QB18": 09250.0,
            "QB19": 08270.0,
            "QB20": 11930.0,
        },
        single_qubit_gate_depolarizing_error_parameters={
            "prx": {
                "QB1": 0.00085,
                "QB2": 0.00183,
                "QB3": 0.00165,
                "QB4": 0.00111,
                "QB5": 0.00114,
                "QB6": 0.00384,
                "QB7": 0.00265,
                "QB8": 0.00084,
                "QB9": 0.00122,
                "QB10": 0.00113,
                "QB11": 0.00274,
                "QB12": 0.00076,
                "QB13": 0.00089,
                "QB14": 0.00074,
                "QB15": 0.00278,
                "QB16": 0.00067,
                "QB17": 0.00085,
                "QB18": 0.00061,
                "QB19": 0.00069,
                "QB20": 0.00088,
            }
        },
        two_qubit_gate_depolarizing_error_parameters={
            "cz": {
                ("QB1", "QB2"): 0.00578,
                ("QB1", "QB4"): 0.00804,
                ("QB2", "QB5"): 0.00749,
                ("QB3", "QB4"): 0.00809,
                ("QB8", "QB3"): 0.00599,
                ("QB4", "QB5"): 0.00431,
                ("QB9", "QB4"): 0.00650,
                ("QB5", "QB6"): 0.00474,
                ("QB10", "QB5"): 0.00339,
                ("QB6", "QB7"): 0.00527,
                ("QB11", "QB6"): 0.01401,
                ("QB12", "QB7"): 0.00294,
                ("QB8", "QB9"): 0.00399,
                ("QB8", "QB13"): 0.00485,
                ("QB9", "QB10"): 0.00638,
                ("QB9", "QB14"): 0.00548,
                ("QB10", "QB11"): 0.00682,
                ("QB10", "QB15"): 0.00961,
                ("QB11", "QB12"): 0.00899,
                ("QB16", "QB11"): 0.00712,
                ("QB17", "QB12"): 0.00407,
                ("QB13", "QB14"): 0.00251,
                ("QB14", "QB15"): 0.00506,
                ("QB18", "QB14"): 0.00420,
                ("QB16", "QB15"): 0.00771,
                ("QB19", "QB15"): 0.00711,
                ("QB16", "QB17"): 0.00643,
                ("QB16", "QB20"): 0.00562,
                ("QB18", "QB19"): 0.00507,
                ("QB19", "QB20"): 0.00578,
            }
        },
        single_qubit_gate_durations={"prx": 20.0},
        two_qubit_gate_durations={"cz": 40.0},
        readout_errors={
            "QB1": {"0": 0.0255, "1": 0.0260},
            "QB2": {"0": 0.0245, "1": 0.0240},
            "QB3": {"0": 0.0285, "1": 0.0245},
            "QB4": {"0": 0.0245, "1": 0.0240},
            "QB5": {"0": 0.0250, "1": 0.0255},
            "QB6": {"0": 0.0265, "1": 0.0275},
            "QB7": {"0": 0.0255, "1": 0.0260},
            "QB8": {"0": 0.0290, "1": 0.0250},
            "QB9": {"0": 0.0260, "1": 0.0245},
            "QB10": {"0": 0.0220, "1": 0.0245},
            "QB11": {"0": 0.0280, "1": 0.0245},
            "QB12": {"0": 0.0255, "1": 0.0250},
            "QB13": {"0": 0.0265, "1": 0.0255},
            "QB14": {"0": 0.0240, "1": 0.0245},
            "QB15": {"0": 0.0230, "1": 0.0245},
            "QB16": {"0": 0.0220, "1": 0.0255},
            "QB17": {"0": 0.0240, "1": 0.0240},
            "QB18": {"0": 0.0245, "1": 0.0260},
            "QB19": {"0": 0.0250, "1": 0.0255},
            "QB20": {"0": 0.0280, "1": 0.0285},
        },
        name="Garnet",
    )
    fake_backend = IQMFakeApollo()
    fake_backend.name = "IQMFakeGarnetBackend"
    return fake_backend.copy_with_error_profile(error_profile)
