#  ********************************************************************************
#   _____  ____ _
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
"""Implementation of a X_pi gate for the e-f subspace."""

from .modulated_pulse_gate import ModulatedTGSS


class X_EF_GateImplementation(ModulatedTGSS):  # noqa: N801
    r"""A gate acting on the e-f subspace, which is implemented using a single X_pi IQ-pulse.

    This class implements pi-rotation X-gates on the e-f subspace of a specific qubit using an :class:`.IQPulse`
    instance, derived from the pulse calibration data provided at construction by
    the class method :meth:`_x_ef_pi_pulse`.
    The pulse is used to resonantly drive the e-f transition of the qubit.

    The pulse calibration data consists of the parameters of an :math:`X^{ef}_\pi` :class:`.IQPulse` only.
    """

    # This is just a rename of the generic ModulatedTGSS, which is enough for now.
