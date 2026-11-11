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
"""Load environment configuration from process environment and .env files."""

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os

from dotenv import dotenv_values, find_dotenv

logger = logging.getLogger(__name__)


_SOURCE_FROM_SC = "__FROM STATION CONTROL__"


@dataclass(frozen=True)
class EnvironmentConfig:
    """Resolved EXA environment configuration."""

    station_control_service_url: str = ""
    iqm_server_url: str = ""
    iqm_token: str | None = None
    configuration_path: str = ""
    loggers_conf_path: str = ""
    rdbms_url: str = ""
    qcm_data_api_url: str = ""


def _load_discovered_env_file_values() -> dict[str, str]:
    """Return non-null values from an auto-discovered .env file."""
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        return {}

    return {key: value for key, value in dotenv_values(env_path).items() if value is not None}


def _create_environment_config(values: Mapping[str, str | None]) -> EnvironmentConfig:
    return EnvironmentConfig(
        station_control_service_url=values.get("STATION_CONTROL_SERVICE_URL") or "",
        iqm_server_url=values.get("IQM_SERVER_URL") or "",
        iqm_token=values.get("IQM_TOKEN"),
        configuration_path=values.get("EXA_CONFIGURATION_PATH") or _SOURCE_FROM_SC,
        loggers_conf_path=values.get("EXA_LOGGERS_CONF_PATH") or "",
        rdbms_url=values.get("EXA_RDBMS_URL") or "",
        qcm_data_api_url=values.get("QCM_DATA_API_URL") or "",
    )


def load_environment(env_path: str | None = None) -> EnvironmentConfig:
    """Load environment configuration from process environment or custom env file.

    Args:
        env_path: Path to the env file to load explicitly. If not provided, values from an auto-discovered ``.env``
            file are loaded without overriding existing process environment variables.

    Raises:
        FileNotFoundError: If env_path is explicitly provided but the file does not exist.

    """
    merged_environment = _load_discovered_env_file_values()
    merged_environment.update(os.environ)

    if env_path is None:
        return _create_environment_config(merged_environment)

    if not os.path.isfile(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    env_file_values = {key: value for key, value in dotenv_values(env_path).items() if value is not None}

    merged_environment.update(env_file_values)

    config = _create_environment_config(merged_environment)

    sensitive_variables = {"IQM_TOKEN", "EXA_RDBMS_URL"}
    resolved_values = {
        "STATION_CONTROL_SERVICE_URL": config.station_control_service_url,
        "IQM_SERVER_URL": config.iqm_server_url,
        "IQM_TOKEN": config.iqm_token,
        "EXA_CONFIGURATION_PATH": config.configuration_path,
        "EXA_LOGGERS_CONF_PATH": config.loggers_conf_path,
        "EXA_RDBMS_URL": config.rdbms_url,
        "QCM_DATA_API_URL": config.qcm_data_api_url,
    }
    for variable_name, value in resolved_values.items():
        if variable_name in sensitive_variables:
            logger.info(
                "%s resolved from env file %s (%s)",
                variable_name,
                env_path,
                "set" if value else "not set",
            )
            continue
        logger.info(
            "%s resolved from env file %s: %s",
            variable_name,
            env_path,
            value or "<empty>",
        )
    return config
