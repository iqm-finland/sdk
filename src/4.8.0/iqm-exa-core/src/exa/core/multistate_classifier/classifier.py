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
"""Generic interface to implement classification of complex/voltage readout data to state."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import itertools
import logging
from typing import Any, ClassVar

import numpy as np
import xarray as xr

from exa.common.data import DataType, Parameter, Setting, SettingNode
from exa.core.experiment_utils import load_observations

from .utils import get_state_vector, multi_kron, package_mitigated_result

logger = logging.getLogger(__name__)


class SingleComponentClassifier(ABC):
    """ABC for single component classifier.

    Implementation of classifier to single component readouts without any correlations to other components. Also
    implements assignment correction.
    """

    MODEL_PARAMETERS: ClassVar[list[str]] = []
    """The names of the model parameters used by this classifier."""
    LABEL = "multi_state_classifier"
    """The label used as part of the observation name for this classifier."""

    @abstractmethod
    def apply(
        self,
        input_data: xr.Dataset,
        component: str,
        data_variable_label: str,
        component_override: dict[str, Any] | None,
    ) -> None:
        """Apply classifier to data.

        Updates the input data with the classification results.

        Args:
            input_data: The dataset to classify.
            component: The component whose readout data to classify.
            data_variable_label: The data variable containing the raw readout to classify.
            component_override: The mapping of observation names to model parameters.

        """

    @abstractmethod
    def calibrate(self, input_data: xr.Dataset, component: str, data_variable_label: str) -> dict:
        """Calibrate classifier with data.

        Args:
            input_data: The labeled calibration dataset.
            component: The component to calibrate the classifier for.
            data_variable_label: The data variable containing the raw readout to calibrate.

        Returns:
            The mapping of observation names to calibrated model parameters.

        """

    def get_observation_name(self, component: str, parameter: str) -> str:
        """Map component and model parameter to label that is used for retrieving and saving observations.

        Args:
            component: The component the parameter belongs to.
            parameter: The name of the model parameter.

        Returns:
            The observation name for the given component and parameter.

        """
        return f"{component}.{self.LABEL}.{parameter}"

    def get_observation_names(self, component: str) -> list:
        """Get observation names corresponding to the component.

        Args:
            component: The component to get the observation names for.

        Returns:
            The observation names for all model parameters of the component.

        """
        return [self.get_observation_name(component, mp) for mp in self.MODEL_PARAMETERS]

    def get_parameter_values(self, component: str) -> dict:
        """Get observation names and values from observations.

        Args:
            component: The component to get the parameter values for.

        Returns:
            The mapping of observation names to model parameter values.

        """
        return dict(load_observations(self.get_observation_names(component)).items())

    def get_component_settings(self, component: str) -> SettingNode:
        """Get settings corresponding to component classifier parameters.

        Parameters missing from observations are left out.

        Args:
            component: The component to get the classifier settings for.

        Returns:
            The setting node containing the classifier model parameters loaded from observations.

        """
        classifier_node = SettingNode(f"{component}_parameters")
        observations = load_observations(self.get_observation_names(component))

        try:
            for mp_label in self.MODEL_PARAMETERS:
                setattr(
                    classifier_node,
                    mp_label,
                    Setting(
                        Parameter(mp_label, data_type=DataType.ANYTHING),
                        observations[self.get_observation_name(component, mp_label)],  # KeyError if not a obs
                    ),
                )
        except KeyError as error:
            logger.debug("Three states classifier not found: %s", error)

        return classifier_node

    def setting_node_to_parameters(self, setting_node: SettingNode, component: str) -> dict:
        """Get parameters from setting node.

        Inverse of :meth:`get_component_settings`.

        Args:
            setting_node: The setting node containing the classifier model parameters for the component.
            component: The component the setting node belongs to.

        Returns:
            The mapping of observation names to model parameter values.

        """
        parameters = {}
        for mp_label in self.MODEL_PARAMETERS:
            parameters[self.get_observation_name(component, mp_label)] = getattr(setting_node, mp_label).value
        return parameters


class MultiStateClassifier:
    """Implementation of the multi-component three-state classifier."""

    LABEL = "multi_state_classifier"
    """The label used as part of the observation name for this classifier."""
    MODEL_PARAMETERS: ClassVar[list[str]] = []
    """The names of the model parameters used by this classifier."""

    component_classifier = SingleComponentClassifier
    """The single component classifier to classify the readout of each individual component."""

    def apply_to_data(
        self,
        input_data: xr.Dataset,
        settings: SettingNode,
        component_mapping: dict[str, str],
        override_parameters: dict[str, Any] | None = None,
        measure_gate: str = "measure",
    ) -> None:
        """Apply classifier to input_data along data_parameters.

        The readout phase is undone only for applying the classifier.

        Args:
            input_data: Xarray dataset for which to apply the classifier
            settings: Setting trees for the experiment.
            component_mapping: Dictionary of components and data_variables for which to apply the classifier.
            override_parameters: Override any parameters that by default are taken from observations
            measure_gate: The gate used for readout. Defaults to `'measure'`.

        """
        if hasattr(settings, "classifier") and not override_parameters:
            override_parameters = self.setting_node_to_parameters(settings.classifier, component_mapping.keys())

        single_component_classifier = self.component_classifier()  # type: ignore[abstract]
        for component, data_label in component_mapping.items():
            # Undo the readout phase while applying the classifier
            phase = settings.get_gate_node_for_locus(measure_gate, component).phase.value
            input_data[data_label] = input_data[data_label] * np.exp(-1j * phase)
            single_component_classifier.apply(input_data, component, data_label, override_parameters)
            input_data[data_label] = input_data[data_label] * np.exp(1j * phase)

    def mitigate_tensor_groups(
        self,
        input_data: xr.Dataset,
        settings: SettingNode,
        component_mapping: dict[str, str],
        component_groups: list[tuple[str, ...]],
        override_parameters: dict[str, Any] | None = None,
        measure_gate: str = "measure",
    ) -> None:
        """Apply classifier to input_data along data_parameters.

        Args:
            input_data: Xarray dataset for which to apply the classifier
            settings: Setting trees for the experiment.
            component_mapping: Dictionary of components and data_variables for which to apply the classifier.
            component_groups: List of component groups for which to perform the mitigation. The order of components
                within the group matters!
            override_parameters: Override any parameters that by default are taken from observations
            measure_gate: The gate used for readout. Defaults to `'measure'`.

        """
        if override_parameters:
            parameters = override_parameters
        else:
            parameters = self.setting_node_to_parameters(settings.classifier, component_mapping.keys())

        single_component_classifier = self.component_classifier()  # type: ignore[abstract]
        for group in component_groups:
            if not all(f"{qubit}_state" in input_data for qubit in group):
                self.apply_to_data(input_data, settings, component_mapping, parameters, measure_gate=measure_gate)
            state_vector = get_state_vector(input_data, group)
            matrices = [
                parameters[single_component_classifier.get_observation_name(qubit, "inverse_assignment_error")]
                for qubit in group
            ]
            full_matrix = multi_kron(*matrices)
            corrected = np.tensordot(full_matrix, state_vector, axes=([-1], [0]))
            result_da = package_mitigated_result(corrected, input_data, group)
            input_data.update({result_da.name: result_da})

    def calibrate_with_data(
        self,
        labeled_input_data: xr.Dataset,
        component_mapping: dict[str, str],
        settings: SettingNode,
        measure_gate: str = "measure",
    ) -> dict[str, Any]:
        """Calibrate the classifier for a set of components using labeled data.

        Assumes the data be of the form "0...0": data, "1...1": data, etc.

        Args:
            labeled_input_data: The dataset with labeled calibration data for each component.
            component_mapping: The map of components to data variables to calibrate the classifier for.
            settings: The setting trees for the experiment.
            measure_gate: The gate used for readout.

        Returns:
            The mapping of observation names to calibrated model parameters for all components.

        """
        single_component_classifier = self.component_classifier()  # type:ignore[abstract]

        component_model_parameters = {}
        for component, data_label in component_mapping.items():
            # Undo the readout phase while applying the classifier
            phase = settings.get_gate_node_for_locus(measure_gate, component).phase.value
            labeled_input_data[data_label] = labeled_input_data[data_label] * np.exp(-1j * phase)
            component_parameters = single_component_classifier.calibrate(labeled_input_data, component, data_label)
            component_model_parameters.update(component_parameters)
            labeled_input_data[data_label] = labeled_input_data[data_label] * np.exp(1j * phase)

        return component_model_parameters  # type: ignore[return-value]

    def get_observation_name(self, parameter: str) -> str:
        """Map component and model parameter to label that is used for retrieving and saving observations.

        Args:
            parameter: The name of the model parameter.

        Returns:
            The observation name for the given parameter.

        """
        return f"{self.LABEL}.{parameter}"

    def get_observation_names(self) -> list[str]:
        """Get a list of observation names based on MODEL_PARAMETERS.

        Returns:
            The observation names for all model parameters.

        """
        return [self.get_observation_name(mp) for mp in self.MODEL_PARAMETERS]

    def get_settings(self, components: Iterable) -> SettingNode:
        """Get the classifier settings for a set of components.

        Components whose classifier parameters are missing from observations are left out.

        Args:
            components: The components to get the classifier settings for.

        Returns:
            The setting node containing the classifier model parameters for each component.

        """
        classifier_node = SettingNode(self.LABEL)
        single_component_classifier = self.component_classifier()  # type:ignore[abstract]

        missing_classifier = []
        sub_parameters = [single_component_classifier.get_observation_names(component) for component in components]
        observations = load_observations(list(itertools.chain.from_iterable(sub_parameters)))
        for component in components:
            single_classifier_node = SettingNode(f"{component}_parameters")
            try:
                for mp_label in single_component_classifier.MODEL_PARAMETERS:
                    single_classifier_node[mp_label] = Setting(
                        Parameter(mp_label, data_type=DataType.ANYTHING),
                        observations[single_component_classifier.get_observation_name(component, mp_label)],
                    )
                classifier_node[f"{component}_parameters"] = single_classifier_node
            except KeyError as error:
                missing_classifier.append(component)
                logger.debug("Three states classifier not found: %s", error)
        if missing_classifier:
            logger.warning(
                f"{missing_classifier} classifiers were not found. Three states readout might not work properly for "
                f"these components."
            )
        return classifier_node

    def setting_node_to_parameters(self, setting_node: SettingNode, components: Iterable[str]) -> dict[str, Any]:
        """Apply inverse of get_settings.

        Args:
            setting_node: Setting node containing the classifier model parameters for the components.
            components: The components the setting node contains parameters for.

        Returns:
            The mapping of observation names to model parameter values.

        """
        parameters = {}
        for mp_label in self.MODEL_PARAMETERS:
            parameters[self.get_observation_name(mp_label)] = getattr(setting_node, mp_label).value

        single_component_classifier = self.component_classifier()  # type: ignore[abstract]
        for component in components:
            comp_parameters = single_component_classifier.setting_node_to_parameters(
                getattr(setting_node, f"{component}_parameters"), component
            )
            parameters.update(comp_parameters)

        return parameters
