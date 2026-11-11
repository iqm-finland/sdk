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
"""Helper functions for experiments that use FFT for their analysis."""

import numpy as np
import xarray as xr


def _add_padding_to_data_zeta_coupling(
    raw_data: np.ndarray, delay_array: np.ndarray, zero_padding: int
) -> tuple[np.ndarray, np.ndarray]:
    """Function to normalize and add padding to data.

    First, we normalize the data along the second dimension (usually `delay`). In order to do it elegantly, we transpose
    the data, such that the first dimension is now `delay`, making it easy to subtract a 3D array with a 2D array.
    The data is then padded with the number specified in `zero_padding`, which is done by adding zeros along the
    second axis at the end. Finally, also `delay_array` is padded with zeros at the end.

    Args:
        raw_data: Data containing the dimensions corresponding to the conditional sweep, delay sweep and some third
            sweep, usually flux sweep.
        delay_array: Array corresponding to the delay sweep.
        zero_padding: Amount of zeros to be added along the delay axis at the end to smoothen the FFT spectrum

    Returns:
        Tuple containing padded data and the padded delay array

    """
    normalized_data = np.transpose(np.transpose(raw_data, (1, 0, 2)) - np.mean(raw_data, axis=1), (1, 0, 2))
    padded_data = np.pad(normalized_data, ((0, 0), (0, zero_padding), (0, 0)), "constant")

    padded_delay_array = np.pad(delay_array, (0, zero_padding), "constant")

    return padded_data, padded_delay_array


def _add_padding_to_data_2d(
    raw_data: np.ndarray, delay_array: np.ndarray, zero_padding: int
) -> tuple[np.ndarray, np.ndarray]:
    """Function to normalize and add padding to data for two-dimensional input.

    First, we subtract the DC component (mean of the raw data along the wait time axis).
    The data is then padded with the number specified in `zero_padding`, which is done by adding zeros along the
    first axis at the end. Finally, also `delay_array` is padded with zeros at the end.

    Args:
        raw_data: Data containing the dimensions corresponding to the wait time sweep and some additional sweep, e.g.
                  demodulation frequency sweep in case of resonator T2 ramsey.
        delay_array: Array corresponding to the wait time sweep.
        zero_padding: Amount of zeros to be added along the wait time axis at the end to smoothen the FFT spectrum

    Returns:
        Tuple containing padded data and the padded delay array

    """
    data_after_subtracting_mean = raw_data - np.mean(raw_data, axis=0)
    padded_data = np.pad(data_after_subtracting_mean, ((0, zero_padding), (0, 0)), "constant")

    padded_delay_array = np.pad(delay_array, (0, zero_padding), "constant")

    return padded_data, padded_delay_array


def _add_padding_to_data(
    raw_data: xr.DataArray, delay_array_name: str, zero_padding: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Function to subtract the mean and add zero padding to the data along the ``delay_array_name`` dimension.

    First, we subtract the DC component from the raw data, i.e., the mean of the raw data along the delay axis,
    typically called ``wait_time``. The data is then padded with zeros at the end, specified by the number
    ``zero_padding``. Finally, also the data array containing the delay times is padded with zeros at the end.

    Args:
        raw_data: Data containing the dimension corresponding to the delay sweep and possibly extra sweeps.
        delay_array_name: The name of the array corresponding to the delay sweep.
        zero_padding: Amount of zeros to be added along the delay sweep dimension to smoothen the FFT spectrum.

    Returns:
        Tuple containing padded data and the padded delay array.

    """
    data_after_subtracting_mean = raw_data - raw_data.mean(delay_array_name)

    padded_data = data_after_subtracting_mean.pad(pad_width={delay_array_name: (0, zero_padding)}, constant_values=0)
    padded_delay_array = padded_data.coords[delay_array_name].fillna(0.0)

    return padded_data, padded_delay_array


def get_frequency_estimate_via_fft(
    raw_data: xr.DataArray,
    delay_array_name: str,
    zero_padding: int,
) -> np.ndarray:
    """Function to estimate the frequency of an oscillation via FFT.

    The following assumptions of the data must hold:

    1. Data has already been rotated to the real part (therefore, complex part is neglected).

    The following steps are taken to extract the dominant oscillation frequency of the raw data:

    1.  First, we subtract the DC component (mean of the raw data along the delay axis).
    2.  The data along the delay axis is padded with zeros at the end. This has the effect of smoothening the FFT
        frequency spectrum.
    3.  Find the integer corresponding to the highest peak in the frequency spectrum.
    4.  With this index, we find the frequency corresponding to the dominant oscillation frequency.

    Args:
        raw_data: Data containing the dimensions corresponding to the delay sweep and some additional sweep, usually
            detuning sweep.
        delay_array_name: The name of the array corresponding to the delay sweep.
        zero_padding: Amount of zeros to be added along the delay sweep dimension to smoothen the FFT spectrum.

    Returns:
        Data array containing the dominant oscillation frequency for each extra sweep point.

    """
    padded_data, padded_delay_array = _add_padding_to_data(raw_data, delay_array_name, zero_padding)

    # assume uniform spacing
    freq_array = np.fft.fftfreq(len(padded_delay_array), float(padded_delay_array[1] - padded_delay_array[0]))

    delay_dim_index = padded_data.dims.index(delay_array_name)
    data_fft = np.fft.fft(np.real(padded_data), axis=delay_dim_index)

    max_frequency_idx = np.argmax(np.abs(data_fft), axis=delay_dim_index)
    frequency_peak_max = np.abs(freq_array[max_frequency_idx])

    return frequency_peak_max
