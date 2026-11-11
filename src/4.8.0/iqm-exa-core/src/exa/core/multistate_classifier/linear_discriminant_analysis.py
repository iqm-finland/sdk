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
"""Classify states based on LDA."""

from typing import Any, ClassVar

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xarray as xr

from .classifier import MultiStateClassifier, SingleComponentClassifier
from .parameters import get_uncorrected_state_label
from .utils import apply_assignment_inverse, get_assigment_inverse


def apply_lda_discriminator(
    data: xr.Dataset, component: str, data_param: str, calibration_parameters: dict[str, Any]
) -> None:
    """Apply lda discriminator to data.

    Updates the provided data with the uncorrected state probabilities and classified states.

    Args:
        data: The dataset containing the raw readout to classify.
        component: The component whose readout data to classify.
        data_param: The data variable containing the raw readout.
        calibration_parameters: The model parameters.

    """
    coef = calibration_parameters["coef"]
    inter = calibration_parameters["inter"]

    mean_g_state, mean_e_state, mean_f_state, classified_states = xr.apply_ufunc(
        _apply_lda_classifier_to_dataset,
        data[data_param],
        kwargs={"coef": coef, "inter": inter},
        input_core_dims=[["repetitions"]],
        output_core_dims=[[], [], [], ["repetitions"]],
        exclude_dims={"repetitions"},
        vectorize=True,
    )

    uncorrected_state_label_0 = get_uncorrected_state_label(component, 0)
    uncorrected_state_label_1 = get_uncorrected_state_label(component, 1)
    uncorrected_state_label_2 = get_uncorrected_state_label(component, 2)

    data.update(
        {
            uncorrected_state_label_0: mean_g_state,
            uncorrected_state_label_1: mean_e_state,
            uncorrected_state_label_2: mean_f_state,
            f"{component}_state": classified_states,
        }
    )


class LDAComponentClassifier(SingleComponentClassifier):
    """Classifier for a single qpu component - a qubit."""

    LABEL = "lda_component_classifier"
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
        """Apply the LDA classifier to data.

        Updates the provided data with the classification results.

        Args:
            input_data: The dataset to classify.
            component: The component whose readout data to classify.
            data_variable_label: The data variable containing the raw readout.
            component_override: The Model parameters to use.

        """
        component_parameters = component_override or self.get_parameter_values(component)

        parameters = {
            "coef": component_parameters[self.get_observation_name(component, "coef")],
            "inter": component_parameters[self.get_observation_name(component, "inter")],
        }

        apply_lda_discriminator(input_data, component, data_variable_label, parameters)

        inv_obs_name = self.get_observation_name(component, "inverse_assignment_error")
        if (inv_matrix := component_parameters.get(inv_obs_name)) is not None:
            apply_assignment_inverse(input_data, component, inv_matrix)

    def calibrate(self, input_data: xr.Dataset, component: str, data_variable_label: str) -> dict[str, Any]:
        """Calibrate the LDA classifier with data.

        Args:
            input_data: The labeled calibration dataset`.
            component: The component to calibrate the classifier for.
            data_variable_label: The data variable containing the raw readout.

        Returns:
            Tha mapping of observation names to calibrated model parameters.

        """
        lda = LinearDiscriminantAnalysis()

        parameters = {}

        train_data = np.concatenate(
            [input_data[data_variable_label].sel(pi_pulse_index=idx).values for idx in range(3)]
        )

        train_data = np.array([[cval.real, cval.imag] for cval in train_data])

        label_length = int(train_data.shape[0] / 3)
        label_data = np.repeat([0, 1, 2], label_length)

        lda.fit(train_data, label_data)

        parameters.update(dict(zip(["coef", "inter"], [lda.coef_, lda.intercept_])))

        apply_lda_discriminator(input_data, component, data_variable_label, parameters)

        inv_matrix = get_assigment_inverse(input_data, component)
        parameters[self.get_observation_name(component, "inverse_assignment_error")] = inv_matrix

        for calib_label in ["coef", "inter"]:
            parameters[self.get_observation_name(component, calib_label)] = parameters.pop(calib_label)

        return parameters


class LDAClassifier(MultiStateClassifier):
    """LDA classifier.

    Classifier model parameters are the linear discriminant analysis decision function coefficients and intercepts
    per component.
    """

    component_classifier = LDAComponentClassifier
    """The single component classifier class used to classify the readout of each individual component."""


def _apply_lda_classifier_to_dataset(data: np.ndarray, **kwargs) -> tuple[np.ndarray, ...]:
    """Classify readout data into ground-excited-f states using the LDA decision function."""
    coef = kwargs["coef"]
    inter = kwargs["inter"]
    eval_of_g, eval_of_e, eval_of_f = (
        data.real * coef[:, 0, np.newaxis] + data.imag * coef[:, 1, np.newaxis] + inter[:, np.newaxis]
    )
    g_state = (eval_of_g >= eval_of_e) & (eval_of_g >= eval_of_f)
    e_state = (eval_of_e > eval_of_g) & (eval_of_e >= eval_of_f)
    f_state = (eval_of_f > eval_of_g) & (eval_of_f > eval_of_e)

    classified_states = e_state.astype(int) * 1 + f_state.astype(int) * 2  # Convert state to int

    return np.mean(g_state), np.mean(e_state), np.mean(f_state), classified_states
