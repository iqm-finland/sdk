python -m black --check src tests
python -m pytest --verbose --isort --pylint src
python -m pytest --verbose --isort --pylint-rcfile=tests/.pylintrc --pylint --cov iqm.iqm_client --cov-report=term-missing --junitxml=test_report.xml tests
python -m mypy -p iqm.iqm_client
python -m mypy tests