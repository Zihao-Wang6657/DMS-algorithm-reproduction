#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/setup_python_env.sh"
"$SCRIPT_DIR/setup_runtime_assets.sh"

echo "bootstrap_complete=1"
