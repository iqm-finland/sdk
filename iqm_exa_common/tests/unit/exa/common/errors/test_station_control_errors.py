#  ********************************************************************************
#  Copyright (c) 2019-2025 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************

from http import HTTPStatus

from exa.common.errors.station_control_errors import (
    ServiceUnavailableError,
    map_from_error_to_status_code,
    map_from_status_code_to_error,
)


class CustomArbitraryError(Exception):
    pass


class CustomServiceUnavailableError(ServiceUnavailableError):
    pass


def test_map_from_arbitrary_error_to_status_code():
    error = CustomArbitraryError("An arbitrary error")
    status_code = map_from_error_to_status_code(error)
    assert status_code == HTTPStatus.INTERNAL_SERVER_ERROR


def test_map_from_error_to_status_code_service_unavailable():
    error = CustomServiceUnavailableError("A service unavailable error")
    status_code = map_from_error_to_status_code(error)
    assert status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_map_from_status_code_to_error_service_unavailable():
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    error_cls = map_from_status_code_to_error(status_code)
    assert issubclass(error_cls, ServiceUnavailableError)
