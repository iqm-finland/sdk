# Copyright 2024 IQM
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
"""IQM Circuit to Pulse Compiler.

IQM Circuit to Pulse Compiler is a Python-based library for converting quantum circuits
into :class:`instruction schedules <exa.pulse.pulse_schedule.PulseSchedule>`
(which map ``Station Control`` controller names to their pulse playlists) and Station Control settings
required for circuit execution, using the calibration data it is given.
The generated schedules and settings can be sent to Station Control
for execution on real or simulated quantum hardware.

CPC is normally only accessed indirectly through Cocos or its reference client
`IQM client <https://docs.meetiqm.com/iqm-client/index.html>`_,
or a frontend such as
`Cirq on IQM <https://docs.meetiqm.com/iqm-client/user_guide_cirq.html>`_ or
`Qiskit on IQM <https://docs.meetiqm.com/iqm-client/user_guide_qiskit.html>`_.
"""

# from importlib.metadata import PackageNotFoundError, version

# try:
#     # Change here if project is renamed and does not equal the package name
#     DIST_NAME = "iqm-cocos"
#     __version__ = version(DIST_NAME)
# except PackageNotFoundError:  # pragma: no cover
#     __version__ = "unknown"
# finally:
#     del version, PackageNotFoundError
