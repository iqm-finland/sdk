#!/usr/bin/env bash
#MISE description="Lint package: iqm-error-reduction-tools"
set -euo pipefail

uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src --config-file pyproject.toml --cache-dir=.mypy_cache
