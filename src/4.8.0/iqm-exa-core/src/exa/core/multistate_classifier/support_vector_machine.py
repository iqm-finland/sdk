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
"""Classify states based on distance on a linear SVM."""

from typing import Any, ClassVar

import numpy as np
from sklearn import svm
import xarray as xr

from .classifier import MultiStateClassifier, SingleComponentClassifier
from .linear_discriminant_analysis import apply_lda_discriminator
from .utils import apply_assignment_inverse, get_assigment_inverse


def apply_svm_discriminator(
    data: xr.Dataset, component: str, data_param: str, calibration_parameters: dict[str, Any]
) -> None:
    """Apply linear SVM.

    The apply function takes same shape of calibrated parameters in both lda and svm discriminator, so here we can reuse
    the function.

    Updates the provided data with the uncorrected state probabilities and classified states.

    Args:
        data: The dataset containing the raw readout to classify.
        component: The component whose readout data to classify.
        data_param: The data variable containing the raw readout.
        calibration_parameters: The model parameters.

    """
    return apply_lda_discriminator(data, component, data_param, calibration_parameters)


class SVMComponentClassifier(SingleComponentClassifier):
    """Classifier for a single QPU component - a qubit - based on a linear support vector machine (SVM)."""

    LABEL = "svm_component_classifier"
    """The label used as part of the observation name for this classifier."""
    MODEL_PARAMETERS: ClassVar[list[str]] = ["coef", "inter", "inverse_assignment_error"]
    """The names of the model parameters used by this classifier."""

    def apply(
        self,
        input_data: xr.Dataset,
        component: str,
        data_variable_label: str,
        component_override: dict[str, Any] | None,
    ) -> None:
        """Apply the SVM classifier to data.

        Updates the provided data with the classification results.

        Args:
            input_data: The dataset to classify.
            component: The component whose readout data to classify.
            data_variable_label: The data variable containing the raw readout.
            component_override: The model parameters to use.

        """
        component_parameters = component_override or self.get_parameter_values(component)

        parameters = {
            "coef": component_parameters[self.get_observation_name(component, "coef")],
            "inter": component_parameters[self.get_observation_name(component, "inter")],
        }

        apply_svm_discriminator(input_data, component, data_variable_label, parameters)

        inv_obs_name = self.get_observation_name(component, "inverse_assignment_error")
        if (inv_matrix := component_parameters.get(inv_obs_name)) is not None:
            apply_assignment_inverse(input_data, component, inv_matrix)

    def calibrate(self, input_data: xr.Dataset, component: str, data_variable_label: str) -> dict[str, Any]:
        """Calibrate the SVM classifier with data.

        Args:
            input_data: The labeled calibration dataset.
            component: The component) to calibrate the classifier for.
            data_variable_label: The data variable containing the raw readout.

        Returns:
            The mapping of observation names to calibrated model parameters.

        """
        _svm = svm.LinearSVC(
            multi_class="crammer_singer",
            dual=False,
        )

        train_data = np.concatenate(
            [input_data[data_variable_label].sel(pi_pulse_index=idx).values for idx in range(3)]
        )

        norm = np.max(np.abs(train_data))
        train_data = train_data / norm

        train_data = np.array([[cval.real, cval.imag] for cval in train_data])

        label_data = np.repeat([0, 1, 2], int(train_data.shape[0] / 3))

        indexes = np.arange(len(label_data))
        rng = np.random.default_rng(seed=1)
        rng.shuffle(indexes)

        label_data = label_data[indexes]
        train_data = train_data[indexes]

        _svm.fit(train_data, label_data)

        parameters = dict(zip(["coef", "inter"], [_svm.coef_, _svm.intercept_ * norm]))

        apply_svm_discriminator(input_data, component, data_variable_label, parameters)

        inv_matrix = get_assigment_inverse(input_data, component)
        parameters[self.get_observation_name(component, "inverse_assignment_error")] = inv_matrix

        for calib_label in ["coef", "inter"]:
            parameters[self.get_observation_name(component, calib_label)] = parameters.pop(calib_label)

        return parameters


class SVMClassifier(MultiStateClassifier):
    """Support vector machine classifier.

    Classifier model parameters are the linear SVM decision function coefficients and intercepts per component.
    """

    component_classifier = SVMComponentClassifier
    """The single component classifier class used to classify the readout of each individual component."""
