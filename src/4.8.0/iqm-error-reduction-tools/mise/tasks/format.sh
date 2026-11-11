#!/usr/bin/env bash
#MISE description="Format package: iqm-error-reduction-tools"
set -euo pipefail

uv run --locked ruff format .
uv run --locked ruff check . --fix
