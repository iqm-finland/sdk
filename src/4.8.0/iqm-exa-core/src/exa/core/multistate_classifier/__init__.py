#  ********************************************************************************
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
#  ********************************************************************************
"""Classifiers to be used for multistate readout."""

__all__ = [
    "DistanceClassifier",
    "LDAClassifier",
    "MultiStateClassifierOptions",
    "SVMClassifier",
    "add_classifier_to_settings",
    "compute_three_state_probability_from_complex_data",
]

from .distance_classifier import DistanceClassifier
from .linear_discriminant_analysis import LDAClassifier
from .support_vector_machine import SVMClassifier
from .utils import (
    MultiStateClassifierOptions,
    add_classifier_to_settings,
    compute_three_state_probability_from_complex_data,
)

CLASSIFIERS = {"DistanceClassifier": DistanceClassifier, "LDAClassifier": LDAClassifier, "SVMClassifier": SVMClassifier}
