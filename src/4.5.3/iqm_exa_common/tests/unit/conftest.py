#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import logging

from mockito import unstub as mockito_unstub
from mockito import verifyNoUnwantedInteractions, verifyStubbedInvocationsAreUsed
import pytest


@pytest.fixture
def tolerances():
    """A tuple with pre-defined tolerances to use in assertions"""
    return (1e-4, 1e-8)


@pytest.fixture
def unstub_():
    """Guarantee that mockito.unstub() is used on teardown."""
    yield mockito_unstub
    mockito_unstub()


@pytest.fixture
def unstub(unstub_):
    """Additionally to mockito.unstub() ensures that stubs are actually used."""
    yield unstub_

    verifyStubbedInvocationsAreUsed()
    verifyNoUnwantedInteractions()


@pytest.fixture
def caplog(caplog):
    """Handle loggers that are configured with propagate=False, which cause the message not to reach the caplog handler.

    https://github.com/eisensheng/pytest-catchlog/issues/44
    """
    restore = []
    for logger in logging.Logger.manager.loggerDict.values():
        try:
            if not logger.propagate:
                logger.propagate = True
                restore += [logger]
        except AttributeError:
            pass
    yield caplog
    for logger in restore:
        logger.propagate = False
