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
"""Classify states based on distance to average."""

from typing import Any, ClassVar

import numpy as np
import xarray as xr

from iqm.cpc.compiler._utils.stages import REPETITIONS

from .classifier import MultiStateClassifier, SingleComponentClassifier
from .parameters import get_scaled_average_state_label, get_uncorrected_state_label
from .utils import apply_assignment_inverse, get_assigment_inverse


def apply_dist_discriminator(
    data: xr.Dataset, component: str, data_param: str, component_parameters: dict[str, Any]
) -> None:
    """Apply discrimantor to data based on average voltage values for the first three states.

    Updates the provided data with the uncorrected state probabilities and classified states.

    Args:
        data: The dataset containing the raw readout to classify.
        component: The component whose readout data to classify.
        data_param: The data variable containing the raw readout.
        component_parameters: The model parameters.

    """
    mean_g_para = component_parameters["state_0_mean"]
    mean_e_para = component_parameters["state_1_mean"]
    mean_f_para = component_parameters["state_2_mean"]

    mean_g_state, mean_e_state, mean_f_state, classified_states = xr.apply_ufunc(
        _apply_dist_classifier_to_dataset,
        data[data_param],
        mean_g_para,
        mean_e_para,
        mean_f_para,
        input_core_dims=[["repetitions"], [], [], []],
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


def apply_scaled_average_readout(
    input_data: xr.Dataset, component: str, data_variable_label: str, component_parameters: dict[str, Any]
) -> None:
    """Apply scaled average readout to data - same logic as exa complex readout -> state probability.

     Updates the provided data with the scaled average state labels.

    Args:
        input_data: The dataset containing the raw readout data.
        component: Component (e.g. qubit) whose readout data to process.
        data_variable_label: The data variable containing the raw readout data.
        component_parameters: The model parameters.

    """
    avg_data = input_data.mean(dim=REPETITIONS.name)[data_variable_label]
    max_i = np.max(np.abs(avg_data.real))
    max_q = np.max(np.abs(avg_data.imag))

    matrix_s_avg: np.ndarray[tuple[Any, ...], Any] = np.ones((3, 3))
    matrix_s_avg[1, 0] = component_parameters["state_0_mean"].real / max_i
    matrix_s_avg[2, 0] = component_parameters["state_0_mean"].imag / max_q

    matrix_s_avg[1, 1] = component_parameters["state_1_mean"].real / max_i
    matrix_s_avg[2, 1] = component_parameters["state_1_mean"].imag / max_q

    matrix_s_avg[1, 2] = component_parameters["state_2_mean"].real / max_i
    matrix_s_avg[2, 2] = component_parameters["state_2_mean"].imag / max_q

    matrix_s_avg = np.linalg.pinv(matrix_s_avg)

    fixed = np.tensordot(
        matrix_s_avg,
        np.array([np.ones(avg_data.shape), avg_data.real / max_i, avg_data.imag / max_q]),
        axes=([-1], [0]),
    )

    input_data.update(
        {get_scaled_average_state_label(component, label): (avg_data.dims, fixed[label]) for label in range(3)}
    )


class SingleComponentDistanceClassifier(SingleComponentClassifier):
    """Component classifier that assigns states based on distance to the average readout value of each state."""

    LABEL = "distance_component_classifier"
    """The label used as part of the observation name for this classifier."""
    MODEL_PARAMETERS: ClassVar[list[str]] = ["state_0_mean", "state_1_mean", "state_2_mean", "inverse_assignment_error"]
    """The names of the model parameters used by this classifier."""

    def apply(
        self,
        input_data: xr.Dataset,
        component: str,
        data_variable_label: str,
        component_override: dict[str, Any] | None,
    ) -> None:
        """Apply the distance classifier to data.

        Updates the provided data with the classification results.

        Args:
            input_data: The dataset to classify.
            component: The component whose readout data to classify.
            data_variable_label: The data variable containing the raw readout.
            component_override: The model parameters to use.

        """
        component_parameters = component_override or self.get_parameter_values(component)

        parameters = {
            "state_0_mean": component_parameters[self.get_observation_name(component, "state_0_mean")],
            "state_1_mean": component_parameters[self.get_observation_name(component, "state_1_mean")],
            "state_2_mean": component_parameters[self.get_observation_name(component, "state_2_mean")],
        }
        apply_dist_discriminator(input_data, component, data_variable_label, parameters)

        inv_obs_name = self.get_observation_name(component, "inverse_assignment_error")
        if (inv_matrix := component_parameters.get(inv_obs_name)) is not None:
            apply_assignment_inverse(input_data, component, inv_matrix)

        # Apply discriminator to average readout data
        apply_scaled_average_readout(input_data, component, data_variable_label, parameters)

    def calibrate(self, input_data: xr.Dataset, component: str, data_variable_label: str) -> dict[str, Any]:
        """Calibrate the distance classifier with data.

        Args:
            input_data: The labeled calibration dataset.
            component: The component to calibrate the classifier for.
            data_variable_label: The data variablecontaining the raw readout.

        Returns:
            The mapping of observation names to calibrated model parameters.

        """
        parameters: dict[str, Any] = {
            f"state_{idx}_mean": input_data[data_variable_label].sel(pi_pulse_index=idx).mean("repetitions").values
            for idx in range(3)
        }

        apply_dist_discriminator(input_data, component, data_variable_label, parameters)

        inv_matrix = get_assigment_inverse(input_data, component)
        parameters[self.get_observation_name(component, "inverse_assignment_error")] = inv_matrix

        # Update labels to observations
        for calib_label in ["state_0_mean", "state_1_mean", "state_2_mean"]:
            parameters[self.get_observation_name(component, calib_label)] = parameters.pop(calib_label)

        return parameters


class DistanceClassifier(MultiStateClassifier):
    """Simple classifier to distinguish g, e and f states.

    Classifier model parameters are three average readout values per component.
    """

    component_classifier = SingleComponentDistanceClassifier
    """Single component classifier class used to classify the readout of each individual component."""


def _apply_dist_classifier_to_dataset(
    data: np.ndarray, mean_g_para: complex, mean_e_para: complex, mean_f_para: complex
) -> tuple[np.ndarray, ...]:
    """Classify readout data into ground-excited-f states by nearest-neighbor distance to the state averages."""
    dist_of_data = np.absolute(data - np.array([mean_g_para, mean_e_para, mean_f_para])[:, None])

    g_state = (dist_of_data[0] <= dist_of_data[1]) & (dist_of_data[0] <= dist_of_data[2])
    e_state = (dist_of_data[1] < dist_of_data[0]) & (dist_of_data[1] <= dist_of_data[2])
    f_state = (dist_of_data[2] < dist_of_data[0]) & (dist_of_data[2] < dist_of_data[1])

    classified_states = e_state.astype(int) * 1 + f_state.astype(int) * 2  # Convert state to int

    return np.mean(g_state), np.mean(e_state), np.mean(f_state), classified_states
