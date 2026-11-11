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
"""Experiment tracker that retries to rerun immediately if a run fails."""

import logging

from exa.core.sequence import ExperimentTracker

logger = logging.getLogger(__name__)


class RetryOnRunFailure(ExperimentTracker):
    """Experiment tracker that retries to run if run failed, max 5 times."""

    def run(self, *args, **kwargs):  # noqa: ANN201
        retries = 0
        while retries < 5:
            try:
                return super().run(*args, **kwargs)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as err:
                retries += 1
                logger.warning("Failure, %s. Retry #%d.", err, retries)
        raise RuntimeError(f"Tried to run {retries} times but got an error each time.")
