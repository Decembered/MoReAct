#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
moreact_python="${MOREACT_PYTHON:-/data/users/autovla/.envs/remogen-motion-only/bin/python}"
exec "$moreact_python" scripts/run_smoke.py
