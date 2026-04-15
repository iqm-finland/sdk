#  ********************************************************************************
#  Copyright (c) 2019-2022 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.helpers.software_version_helper import get_all_software_versions

_all_software_versions = get_all_software_versions()


def test_exa_common_exists_in_all_software_versions():
    assert "iqm-exa-common" in _all_software_versions


def test_python_version_exists_in_all_software_versions():
    assert _all_software_versions["python"].startswith("3")


def test_calling_again_does_not_affect_versions():
    versions_again = get_all_software_versions()
    assert _all_software_versions == versions_again
