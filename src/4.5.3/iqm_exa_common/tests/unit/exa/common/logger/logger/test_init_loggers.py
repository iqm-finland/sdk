#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import logging
import re

import pytest

from exa.common.logger import init_loggers


def test_set_invalid_logging_level_raises_error():
    with pytest.raises(ValueError, match="Logger name 'module_x' has an incorrect level value 'TEST'"):
        init_loggers({"module_x": "TESt"})


def test_set_invalid_default_logging_level_raises_error():
    with pytest.raises(ValueError, match="Logger name 'module_x' has an incorrect level value 'TEST'"):
        init_loggers({"module_x": None}, default_level="teSt")


def test_root_logger_is_setup_without_input_args():
    init_loggers()
    assert logging.getLogger() is logging.getLogger("")
    assert logging.getLogger().level == logging.WARNING


def test_default_level_is_ignored_by_root_logger_if_root_is_not_in_input_args():
    init_loggers(default_level="debug")
    assert logging.getLogger() is logging.getLogger("")
    assert logging.getLogger().level == logging.WARNING


def test_logging_level_is_set_correctly_for_root_logger():
    init_loggers({"": "debug"}, default_level="info")
    assert logging.getLogger() is logging.getLogger("")
    assert logging.getLogger().level == logging.DEBUG


def test_logging_level_is_set_correctly_for_named_logger():
    init_loggers({"module_x": "debug"}, default_level="info")
    assert logging.getLogger("module_x").level == logging.DEBUG

    assert logging.getLogger().level == logging.WARNING  # root should use WARNING by default


@pytest.mark.parametrize("none_or_emptystring", [None, ""])
def test_default_logging_level_is_set_correctly(none_or_emptystring):
    init_loggers({"module_x": none_or_emptystring}, default_level="debug")

    assert logging.getLogger("module_x").level == logging.DEBUG

    assert logging.getLogger().level == logging.WARNING  # root should use WARNING by default


def test_log_level_determines_stream(capsys):
    init_loggers({"module_x": "DEBUG"})

    logging.getLogger("module_x").debug("module_x debug")
    logging.getLogger("module_x").info("module_x info")
    logging.getLogger("module_x").warning("module_x warning")
    logging.getLogger("module_x").error("module_x error")
    logging.getLogger("module_x").critical("module_x critical")

    captured = capsys.readouterr()
    assert "module_x debug" in captured.out
    assert "module_x debug" not in captured.err

    assert "module_x info" in captured.out
    assert "module_x info" not in captured.err

    assert "module_x warning" not in captured.out
    assert "module_x warning" in captured.err

    assert "module_x error" not in captured.out
    assert "module_x error" in captured.err

    assert "module_x critical" not in captured.out
    assert "module_x critical" in captured.err


@pytest.mark.parametrize(
    "loggers_dict",
    [
        {"module_x": "DEBUG", "module_x.module_y": "INFO"},
        {"module_x.module_y": "INFO", "module_x": "DEBUG"},
    ],
)
def test_descendant_level_is_overridable_and_input_dict_order_has_no_effect(capsys, loggers_dict):
    init_loggers(loggers_dict)

    assert logging.getLogger("module_x").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("module_x.module_y").getEffectiveLevel() == logging.INFO

    logging.getLogger("module_x").debug("module_x debug")
    logging.getLogger("module_x").info("module_x info")
    logging.getLogger("module_x.module_y").debug("module_x.module_y debug")
    logging.getLogger("module_x.module_y").info("module_x.module_y info")

    captured = capsys.readouterr()
    assert "module_x debug" in captured.out
    assert "module_x info" in captured.out
    assert "module_x.module_y debug" not in captured.out
    assert "module_x.module_y info" in captured.out


@pytest.mark.parametrize(
    "loggers_dict",
    [
        {"module_x": "INFO", "module_x.module_y": "DEBUG"},
        {"module_x.module_y": "DEBUG", "module_x": "INFO"},
    ],
)
def test_parent_level_is_overridable_and_input_dict_order_has_no_effect(capsys, loggers_dict):
    init_loggers(loggers_dict)

    assert logging.getLogger("module_x").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("module_x.module_y").getEffectiveLevel() == logging.DEBUG

    logging.getLogger("module_x").debug("module_x debug")
    logging.getLogger("module_x").info("module_x info")
    logging.getLogger("module_x.module_y").debug("module_x.module_y debug")
    logging.getLogger("module_x.module_y").info("module_x.module_y info")

    captured = capsys.readouterr()
    assert "module_x debug" not in captured.out
    assert "module_x info" in captured.out
    assert "module_x.module_y debug" in captured.out
    assert "module_x.module_y info" in captured.out


def get_brief_fmt_regexp(extra_info: str = ""):
    return (
        r"^"
        r"\["
        r"[0-9]{2}-[0-9]{2}"  # date
        r" [0-9]{2}:[0-9]{2}:[0-9]{2}"  # time
        r";I"  # level
        f"{extra_info}"
        r"\]"
        r" module_x info"  # message
        r"$"
    )


def get_verbose_fmt_regexp(caller_func_name: str, extra_info: str = ""):
    return (
        r"^"
        r"\["
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}"  # date
        r" [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}"  # time,ms
        r";INFO"  # level
        r";.+\([0-9]+\)"  # process(PID)
        r";.+\([0-9]+\)"  # thread(TID)
        r";module_x"
        r";test_init_loggers.py"
        r":[0-9]{1,4}"  # lineno
        f":{caller_func_name}"
        f"{extra_info}"
        r"\]"
        r" module_x info"  # message
        r"$"
    )


def test_logging_format_brief_output(capsys):
    init_loggers({"module_x": "info"}, verbose=False)
    logging.getLogger("module_x").info("module_x info")

    captured = capsys.readouterr()
    assert re.match(get_brief_fmt_regexp(), captured.out)


def test_logging_format_brief_output_with_extra(capsys):
    init_loggers({"module_x": "info"}, verbose=False, extra_info_getter=lambda: ";this is extra")
    logging.getLogger("module_x").info("module_x info")

    captured = capsys.readouterr()
    assert re.match(get_brief_fmt_regexp(";this is extra"), captured.out)


def test_logging_format_verbose_output(capsys):
    init_loggers({"module_x": "info"}, verbose=True)
    logging.getLogger("module_x").info("module_x info")

    captured = capsys.readouterr()
    assert re.match(
        get_verbose_fmt_regexp(
            caller_func_name="test_logging_format_verbose_output",
        ),
        captured.out,
    )


def test_logging_format_verbose_output_with_extra(capsys):
    init_loggers({"module_x": "info"}, verbose=True, extra_info_getter=lambda: ";this is extra")
    logging.getLogger("module_x").info("module_x info")

    captured = capsys.readouterr()
    assert re.match(
        get_verbose_fmt_regexp(
            caller_func_name="test_logging_format_verbose_output_with_extra",
            extra_info=";this is extra",
        ),
        captured.out,
    )
